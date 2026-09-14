"""Prepare the next queued job's reference pack while the current one renders.

ComfyUI runs one prompt at a time, node by node, and MiniMaxH3ReferencePack sits at the
head of every R2V graph: every reference is decoded, the VLM's copy is encoded and the
prompt round-trips to the provider before the sampler sees a single step. On a rented
GPU that is 15-45 s of idle silicon per job (pod logs, 2026-09-14: `chat` median
17.7 s, `build_done` median 18.7 s, p75 24 s). Nothing inside the graph can overlap
it - the sockets feed the encoder - but the NEXT job's pack can be built while this one
samples, and a render is minutes long.

So: a daemon thread watches the queue's pending list and, for the first few pending
prompts, builds every reference pack it finds, keyed by the exact string IS_CHANGED
produces for that node. build() looks its own key up first. A full hit returns the
prepared outputs; a prompt-only hit (the decoded media was over the retention budget)
skips just the provider call and decodes again. Anything else runs as it always did.

Fail-safe by construction. A prepare that raises is logged and never retried for that
key: a re-upload changes the key (IS_CHANGED hashes mtime+size), and a provider error is
retried by build() itself when the job runs. A linked input is skipped because its value
is unknowable before execution. build() never trusts a key it did not compute itself,
so the worst case of any mismatch is the old behaviour, not a wrong reference set.

Queue access is read-only and goes through `prompt_queue.get_current_queue`. Queue
Manager hijacks that method to page its sqlite - `(page, page_size)` with page_size > 0
returns pending rows, 0 returns none - while core answers from its heap and takes no
arguments. Both shapes are handled. The one write is wrapping `prompt_queue.get`, so a
job starting wakes the thread at once instead of at the next poll; Queue Manager wraps
the same method and both orders chain correctly.

Environment:
    MINIMAX_REFPACK_PREFETCH         "0"/"off"/"false"/"no" disables the whole thing
    MINIMAX_REFPACK_PREFETCH_GIB     decoded media retained across prepared packs (8)
    MINIMAX_REFPACK_PREFETCH_JOBS    pending prompts prepared ahead (1)
    MINIMAX_REFPACK_PREFETCH_POLL_S  seconds between queue checks (10)
"""

from __future__ import annotations

import collections
import hashlib
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from . import logs

Job = tuple[str, dict[str, Any]]

NODE_CLASS = "MiniMaxH3ReferencePack"

ENABLED_ENV = "MINIMAX_REFPACK_PREFETCH"
BUDGET_ENV = "MINIMAX_REFPACK_PREFETCH_GIB"
JOBS_ENV = "MINIMAX_REFPACK_PREFETCH_JOBS"
POLL_ENV = "MINIMAX_REFPACK_PREFETCH_POLL_S"

# 8 GiB holds one 10 s 1080p plate (~6 GB float32) beside whatever ComfyUI's own output
# cache keeps for the running job. Same order as media's decode ceiling, on purpose: a
# pack that fits the decoder fits here, and a bigger one is kept as prompt-only.
DEFAULT_BUDGET_BYTES = 8.0 * 2**30
DEFAULT_JOBS = 1
DEFAULT_POLL_S = 10.0
_MAX_JOBS = 8
# How many pending prompts are read per pass. Only the first `jobs` are prepared; the
# rest are read so an entry whose job was reordered further down stays alive.
_PENDING_LIMIT = 50

# How long build() waits for a prepare already under way before doing the work itself.
# Generous: a prepare is bounded by the provider timeout plus the decodes, and giving
# up early would run the same decode twice on a box already short of memory.
WAIT_S = 900.0
# A prepared entry outlives its job's disappearance from the queue by this much before
# it is dropped. The worker taking a job moves it pending -> running between two polls,
# and build() may not reach the pack node for a while; without the grace the poll
# thread could evict the entry the worker is about to consume.
EVICT_AFTER_S = 60.0
# The ceiling on any prepared entry's life, whatever the queue says. Renders are minutes
# long, so an hour-old entry is stale by any measure - and this is the bound that holds
# when the queue cannot be read at all, or the thread is gone and only build() is left
# to prune (consume() applies it too).
MAX_AGE_S = 3600.0
_FAILED_KEYS_KEPT = 64
_RECENT_KEYS_KEPT = 32
_OFF_VALUES = {"0", "off", "false", "no"}


@dataclass
class Prepared:
    """One reference pack, built ahead of its job."""

    key: str
    outputs: tuple[Any, ...] | None  # None: the media was dropped for the budget, the prompt kept
    prompt_text: str
    debug_sink: list[str]
    media_bytes: int  # what this entry retains; 0 when outputs is None
    prepared_at: float


def key_digest(key: str) -> str:
    """What the logs carry instead of the key, which embeds the direction text and the
    whole references list."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def enabled(env: Mapping[str, str] | None = None) -> bool:
    source: Mapping[str, str] = os.environ if env is None else env
    return (source.get(ENABLED_ENV) or "").strip().lower() not in _OFF_VALUES


def _env_number(name: str, default: float, minimum: float, env: Mapping[str, str]) -> float:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logs.warn("prefetch_env_ignored", name=name, value=str(raw), reason="not a number")
        return default
    if not math.isfinite(value) or value < minimum:
        logs.warn("prefetch_env_ignored", name=name, value=str(raw), reason=f"below {minimum}")
        return default
    return value


def _number(item) -> float:
    number = item[0] if isinstance(item, (list, tuple)) and item else 0
    return float(number) if isinstance(number, (int, float)) else 0.0


def _jobs(items) -> list[Job]:
    """(prompt_id, prompt) for every well-formed queue item. The native tuple is
    (number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive); Queue
    Manager stores the same shape as a JSON list. Anything else is skipped."""
    out: list[Job] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) > 2 and isinstance(item[2], dict):
            out.append((str(item[1]), item[2]))
    return out


def queue_view(limit: int = _PENDING_LIMIT) -> tuple[list[Job], list[Job]]:
    """(running, pending) as (prompt_id, prompt) pairs, pending in execution order.

    Only meaningful inside a running ComfyUI; `server` is imported here and not at
    module level so the module imports cleanly under pytest.
    """
    from server import PromptServer  # pyright: ignore[reportMissingImports]

    queue = PromptServer.instance.prompt_queue
    try:
        view = queue.get_current_queue(0, limit)  # Queue Manager: (page, page_size)
    except TypeError:
        view = queue.get_current_queue()  # core: no arguments, heap order
    running, pending = view[0], view[1]
    return _jobs(running), _jobs(sorted(pending, key=_number))


def _packs(prompt_dict: dict[str, Any]):
    for node_id, node in prompt_dict.items():
        if isinstance(node, dict) and node.get("class_type") == NODE_CLASS:
            yield str(node_id), node.get("inputs") or {}


class Prefetcher:
    """Owns the prepared packs, the in-flight set and the thread that fills them.

    `key_fn(**inputs)` must be the node's own IS_CHANGED, and `build_fn(**inputs)` the
    node's own builder returning something with `outputs`, `prompt_text`, `debug_sink`
    and `media_bytes` - the two are injected rather than imported so this module never
    imports nodes.py (which imports it) and so a test can drive it with fakes.
    """

    def __init__(
        self,
        *,
        key_fn: Callable[..., str],
        build_fn: Callable[..., Any],
        pending_fn: Callable[[], tuple[list[Job], list[Job]]] | None = None,
        budget_bytes: float = DEFAULT_BUDGET_BYTES,
        jobs: int = DEFAULT_JOBS,
        poll_s: float = DEFAULT_POLL_S,
        evict_after_s: float = EVICT_AFTER_S,
        max_age_s: float = MAX_AGE_S,
        clock: Callable[[], float] | None = None,
    ):
        self._key_fn = key_fn
        self._build_fn = build_fn
        self._pending_fn = pending_fn or queue_view
        self.budget_bytes = float(budget_bytes)
        self.jobs = max(1, min(_MAX_JOBS, int(jobs)))
        self.poll_s = max(1.0, float(poll_s))
        self.evict_after_s = float(evict_after_s)
        self.max_age_s = float(max_age_s)
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._ready: dict[str, Prepared] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._failed: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._recent: collections.deque[str] = collections.deque(maxlen=_RECENT_KEYS_KEPT)
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(cls, *, key_fn, build_fn, env: Mapping[str, str] | None = None) -> Prefetcher:
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            key_fn=key_fn,
            build_fn=build_fn,
            budget_bytes=_env_number(BUDGET_ENV, DEFAULT_BUDGET_BYTES / 2**30, 0.0, source) * 2**30,
            jobs=int(_env_number(JOBS_ENV, DEFAULT_JOBS, 1, source)),
            poll_s=_env_number(POLL_ENV, DEFAULT_POLL_S, 1.0, source),
        )

    # ---- the thread --------------------------------------------------------------

    def poke(self) -> None:
        """A job just started (or the queue changed): look again now, not at the poll."""
        self._wake.set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="MiniMaxRefPack-prefetch", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            self._wake.wait(timeout=self.poll_s)
            self._wake.clear()
            try:
                self.look_ahead()
            except Exception as e:
                # A queue view that raises (a hijacker with a new signature, a closed
                # database) must not kill the thread; the next pass tries again.
                logs.warn("prefetch_pass_failed", error=type(e).__name__)

    # ---- one pass ------------------------------------------------------------------

    def look_ahead(self) -> int:
        """Prepare what the next jobs need and drop what no queued job needs any more.

        Returns how many packs were built in this pass. Synchronous: the caller (the
        thread, or a test) pays for the builds, one at a time, in queue order - one
        decode at a time is the point, on a box that is already rendering.
        """
        try:
            running, pending = self._pending_fn()
        except Exception as e:
            # A queue that cannot be read (a hijacker with a new signature, a closed
            # database) must not turn retained media into a leak: prune by age alone,
            # since nothing can be said about what is live.
            logs.warn("prefetch_queue_unreadable", error=type(e).__name__)
            self._evict(None)
            return 0
        live: set[str] = set()
        todo: list[tuple[str, str, str, dict[str, Any]]] = []
        for prompt_id, prompt_dict in running:
            for node_id, inputs in _packs(prompt_dict):
                key = self._key_of(inputs, prompt_id, node_id)
                if key is not None:
                    live.add(key)
        for position, (prompt_id, prompt_dict) in enumerate(pending):
            for node_id, inputs in _packs(prompt_dict):
                key = self._key_of(inputs, prompt_id, node_id)
                if key is None:
                    continue
                live.add(key)
                if position < self.jobs:
                    todo.append((key, prompt_id, node_id, inputs))
        self._evict(live)
        built = 0
        for key, prompt_id, node_id, inputs in todo:
            if self._begin(key):
                self._prepare(key, prompt_id, node_id, inputs)
                built += 1
        return built

    def _key_of(self, inputs: dict[str, Any], prompt_id: str, node_id: str) -> str | None:
        # A list value is a link to another node's output. Its value exists only once
        # that node has run, so the key cannot be known here - and a key computed from
        # the link's [node_id, slot] pair would never match what build() computes.
        linked = [name for name, value in inputs.items() if isinstance(value, list)]
        if linked:
            logs.debug(
                "prefetch_skipped", prompt_id=prompt_id, node=node_id,
                reason="linked input", inputs=linked,
            )
            return None
        try:
            return self._key_fn(**inputs)
        except Exception as e:
            logs.warn(
                "prefetch_skipped", prompt_id=prompt_id, node=node_id,
                reason="key", error=type(e).__name__,
            )
            return None

    def _begin(self, key: str) -> bool:
        with self._lock:
            if key in self._ready or key in self._inflight or key in self._failed:
                return False
            if key in self._recent:
                # build() produced (or consumed) this key in this process, so ComfyUI's
                # own output cache will serve the job without calling build() at all.
                # This is the re-queue-with-a-seed-change case; preparing it would pay
                # the provider for a prompt nobody reads.
                return False
            self._inflight[key] = threading.Event()
            return True

    def _prepare(self, key: str, prompt_id: str, node_id: str, inputs: dict[str, Any]) -> None:
        digest = key_digest(key)
        try:
            with logs.timed("prefetch", prompt_id=prompt_id, node=node_id, digest=digest) as fields:
                result = self._build_fn(**inputs)
                fields["kept"] = self._store(key, result)
                fields["media_mb"] = result.media_bytes / 2**20
                fields["prompt_chars"] = len(result.prompt_text)
        except Exception as e:
            # timed() has already written the failure line with the exception type; the
            # message is the one build() would show the user for the same job, so it is
            # safe to repeat, cut to a line.
            logs.warn("prefetch_failed", digest=digest, error=type(e).__name__, detail=str(e)[:160])
            with self._lock:
                self._failed[key] = None
                while len(self._failed) > _FAILED_KEYS_KEPT:
                    self._failed.popitem(last=False)
        finally:
            with self._lock:
                event = self._inflight.pop(key, None)
            if event is not None:
                event.set()

    def _store(self, key: str, result: Any) -> str:
        """Keep the whole pack if it fits beside what is already retained, else the
        prompt alone. Returns "full" or "prompt", for the log line."""
        with self._lock:
            retained = sum(p.media_bytes for p in self._ready.values())
            keep_media = result.media_bytes <= max(0.0, self.budget_bytes - retained)
            self._ready[key] = Prepared(
                key=key,
                outputs=tuple(result.outputs) if keep_media else None,
                prompt_text=result.prompt_text,
                debug_sink=list(result.debug_sink),
                media_bytes=int(result.media_bytes) if keep_media else 0,
                prepared_at=self._clock(),
            )
        return "full" if keep_media else "prompt"

    def _evict(self, live: set[str] | None) -> None:
        """Drop entries no queued job needs (after the grace), and any past the age
        ceiling. `live=None` means the queue could not be read: age alone decides."""
        now = self._clock()
        with self._lock:
            stale = []
            for key, entry in self._ready.items():
                age = now - entry.prepared_at
                unneeded = live is not None and key not in live and age >= self.evict_after_s
                if unneeded or age >= self.max_age_s:
                    stale.append(key)
            for key in stale:
                del self._ready[key]
        for key in stale:
            logs.log("prefetch_evicted", digest=key_digest(key))

    # ---- build()'s side ------------------------------------------------------------

    def _acquire(self, key: str) -> tuple[Prepared | None, threading.Event | None, bool]:
        """One lock section: the prepared entry, else the event of a prepare under way,
        else a reservation of the key for the caller (the third value)."""
        with self._lock:
            entry = self._ready.pop(key, None)
            if entry is not None:
                return entry, None, False
            event = self._inflight.get(key)
            if event is not None:
                return None, event, False
            self._inflight[key] = threading.Event()
            return None, None, True

    @contextmanager
    def consume(self, key: str, wait_s: float = WAIT_S):
        """build()'s one call: yields the prepared pack for `key`, or None.

        Lookup and reservation are ONE step under the lock, so the poll thread cannot
        start the same prepare between a miss and the build that follows it (which
        would decode twice and pay the provider twice). A prepare already under way is
        waited for; if it fails, the key is reserved and the caller builds. The entry
        is popped: the outputs go into ComfyUI's own cache from here, and a second
        reference would double what a plate costs in RAM.

        Also prunes by age first, so memory stays bounded even if the thread is gone.
        """
        self._evict(None)
        entry, event, mine = self._acquire(key)
        if event is not None:
            started = time.perf_counter()
            finished = event.wait(wait_s)
            logs.log(
                "prefetch_wait", digest=key_digest(key), finished=finished,
                ms=(time.perf_counter() - started) * 1000.0,
            )
            entry, event, mine = self._acquire(key)
            # Still an event: the prepare outran the wait. Build unreserved rather than
            # wait again; the straggler's entry is evicted once its job has run.
        if entry is not None:
            logs.log(
                "prefetch_hit", digest=key_digest(key),
                kind="full" if entry.outputs is not None else "prompt",
                age_s=self._clock() - entry.prepared_at,
            )
        try:
            yield entry
        finally:
            with self._lock:
                # Remembered whether hit or built: ComfyUI's cache serves this key
                # from now on, so preparing it again would be paid for nothing.
                self._recent.append(key)
                released = self._inflight.pop(key, None) if mine else None
            if released is not None:
                released.set()

    # ---- introspection, for tests and the debug socket -----------------------------

    def ready_keys(self) -> list[str]:
        with self._lock:
            return list(self._ready)

    def retained_bytes(self) -> int:
        with self._lock:
            return sum(p.media_bytes for p in self._ready.values())


# ---- wiring into the running server ---------------------------------------------------

_INSTALLED_FLAG = "_minimax_refpack_prefetch_installed"


def install_on(queue, prefetcher: Prefetcher) -> bool:
    """Wrap `queue.get` so a job starting wakes the thread, then start it. Idempotent."""
    if getattr(queue, _INSTALLED_FLAG, False):
        return False
    previous = queue.get

    def get(timeout=None):
        item = previous(timeout)
        if item is not None:
            prefetcher.poke()
        return item

    queue.get = get
    setattr(queue, _INSTALLED_FLAG, True)
    prefetcher.start()
    logs.log(
        "prefetch_installed", jobs=prefetcher.jobs,
        budget_gib=prefetcher.budget_bytes / 2**30, poll_s=prefetcher.poll_s,
    )
    return True


def install(prefetcher: Prefetcher, env: Mapping[str, str] | None = None) -> bool:
    """Called from the package __init__. Silent no-op outside a running ComfyUI."""
    if not enabled(env):
        logs.log("prefetch_disabled", reason=f"{ENABLED_ENV} is off")
        return False
    try:
        from server import PromptServer  # pyright: ignore[reportMissingImports]
    except ImportError:  # pragma: no cover - pytest, no ComfyUI on sys.path
        return False
    instance = getattr(PromptServer, "instance", None)
    queue = getattr(instance, "prompt_queue", None)
    if queue is None:
        return False
    return install_on(queue, prefetcher)
