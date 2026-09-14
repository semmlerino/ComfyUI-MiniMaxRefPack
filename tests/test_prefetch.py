"""Tests for minimax_refpack.prefetch: the next queued job's pack, built early.

Same seams as test_nodes.py: `folder_paths` is stubbed into sys.modules, the media
loaders and prompt.write_prompt are monkeypatched, so nothing decodes and nothing leaves
the machine. The prefetcher is driven synchronously through look_ahead() with an injected
pending_fn; only the in-flight wait test uses a real thread. The queue-view and install
tests use fake queue objects shaped like core's PromptQueue and Queue Manager's hijack.
"""

import json
import sys
import threading
import time
import types
from typing import Any

import pytest

from minimax_refpack import nodes, prefetch, refs

NODE = "MiniMaxH3ReferencePack"


@pytest.fixture
def fake_folder_paths(tmp_path, monkeypatch):
    # Any object in sys.modules satisfies `import folder_paths`; a namespace types cleanly.
    module = types.SimpleNamespace(get_input_directory=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "folder_paths", module)
    return tmp_path


@pytest.fixture
def stubs(fake_folder_paths, monkeypatch):
    """Loaders return marker strings; write_prompt counts its calls and numbers its
    answer, so a test can tell WHICH call produced the prompt a build returned."""
    (fake_folder_paths / "i1.jpg").write_bytes(b"x")
    (fake_folder_paths / "v1.mp4").write_bytes(b"x")
    calls = {"write_prompt": 0, "load_video": 0}

    def load_video(path, target_fps=24, crop=None, trim=None):
        calls["load_video"] += 1
        return (f"VIDEO:{path}", None)

    def write_prompt(**kwargs):
        calls["write_prompt"] += 1
        if kwargs.get("debug") is not None:
            kwargs["debug"].append("job_type: auto -> standard\n--- payload ---")
        return f"PROMPT#{calls['write_prompt']} for {kwargs['direction']}"

    monkeypatch.setattr(nodes.media, "load_image", lambda path, crop=None, max_edge=0: f"IMG:{path}")
    monkeypatch.setattr(nodes.media, "load_video", load_video)
    monkeypatch.setattr(nodes.prompt, "write_prompt", write_prompt, raising=False)
    return calls


def _inputs(direction: Any = "a girl walks", **over) -> dict[str, Any]:
    """The node's inputs as an API-format prompt carries them (validated values)."""
    base: dict[str, Any] = dict(
        direction=direction, openrouter_api_key="", openrouter_model="m",
        references_json=json.dumps({"references": [
            {"kind": "image", "file": "i1.jpg"},
            {"kind": "video", "file": "v1.mp4"},
        ]}),
        system_prompt="", width=1280, height=720, length_seconds=8.0,
        prompt_provider="openrouter", reasoning_effort="medium", job_type="auto",
        max_reference_edge=2048, api_base="", local_model_slug="", match_video_aspect=False,
    )
    base.update(over)
    return base


def _job(prompt_id, inputs, node_id="7"):
    return (prompt_id, {
        node_id: {"class_type": NODE, "inputs": inputs},
        "8": {"class_type": "SaveVideo", "inputs": {"video": [node_id, 0]}},
    })


def _prefetcher(pending, running=(), **kw):
    state = {"running": list(running), "pending": list(pending)}
    p = prefetch.Prefetcher(
        key_fn=nodes._key_for, build_fn=nodes._prefetch_build,
        pending_fn=lambda: (list(state["running"]), list(state["pending"])), **kw,
    )
    return p, state


def _by_name(out):
    return dict(zip(refs.output_names(), out, strict=True))


# ---- build() and the prepared pack ----------------------------------------------------


def test_full_hit_serves_the_prepared_pack_without_a_second_provider_call(stubs, monkeypatch):
    inputs = _inputs()
    p, _ = _prefetcher([_job("p1", inputs)])
    monkeypatch.setattr(nodes, "PREFETCHER", p)

    assert p.look_ahead() == 1
    assert stubs == {"write_prompt": 1, "load_video": 1}
    assert len(p.ready_keys()) == 1

    out = _by_name(nodes.MiniMaxH3ReferencePack().build(**inputs))
    assert out["prompt"] == "PROMPT#1 for a girl walks"
    assert out["video_1"].startswith("VIDEO:")
    assert "prefetch: prepared while the previous job rendered" in out["debug"]
    # Nothing was redone: the key build() computed from its own arguments matched the
    # one the thread computed from the queued prompt.
    assert stubs == {"write_prompt": 1, "load_video": 1}
    # Popped on hit - ComfyUI's cache holds the outputs from here.
    assert p.ready_keys() == []


def test_prompt_only_hit_decodes_again_but_skips_the_provider(stubs, monkeypatch):
    inputs = _inputs()
    monkeypatch.setattr(nodes, "_media_bytes", lambda outputs: 5 * 2**30)
    p, _ = _prefetcher([_job("p1", inputs)], budget_bytes=1 * 2**30)
    monkeypatch.setattr(nodes, "PREFETCHER", p)

    assert p.look_ahead() == 1
    assert p.retained_bytes() == 0  # over budget: the prompt is kept, the media is not

    out = _by_name(nodes.MiniMaxH3ReferencePack().build(**inputs))
    assert out["prompt"] == "PROMPT#1 for a girl walks"
    assert stubs == {"write_prompt": 1, "load_video": 2}
    assert "prefetch: prompt only" in out["debug"]
    # The payload the prefetcher rendered still reaches the debug socket.
    assert "--- payload sent to OpenRouter ---" in out["debug"]


def test_a_miss_builds_as_before(stubs, monkeypatch):
    p, _ = _prefetcher([])
    monkeypatch.setattr(nodes, "PREFETCHER", p)
    out = _by_name(nodes.MiniMaxH3ReferencePack().build(**_inputs()))
    assert out["prompt"] == "PROMPT#1 for a girl walks"
    assert "prefetch: no" in out["debug"]
    assert stubs == {"write_prompt": 1, "load_video": 1}


def test_a_different_direction_is_a_different_key(stubs, monkeypatch):
    p, _ = _prefetcher([_job("p1", _inputs(direction="one"))])
    monkeypatch.setattr(nodes, "PREFETCHER", p)
    p.look_ahead()
    out = _by_name(nodes.MiniMaxH3ReferencePack().build(**_inputs(direction="two")))
    assert out["prompt"] == "PROMPT#2 for two"
    assert stubs["write_prompt"] == 2


def test_take_waits_for_a_prepare_already_in_flight(stubs, monkeypatch):
    inputs = _inputs()
    slow = nodes.prompt.write_prompt

    def slow_write(**kwargs):
        time.sleep(0.3)
        return slow(**kwargs)

    monkeypatch.setattr(nodes.prompt, "write_prompt", slow_write, raising=False)
    p, _ = _prefetcher([_job("p1", inputs)])
    worker = threading.Thread(target=p.look_ahead)
    worker.start()
    time.sleep(0.05)  # the prepare is under way
    started = time.perf_counter()
    with p.consume(nodes._key_for(**inputs), wait_s=5) as entry:
        assert entry is not None and entry.outputs is not None
    worker.join()
    assert time.perf_counter() - started >= 0.2
    assert stubs["write_prompt"] == 1


# ---- what look_ahead() prepares and what it leaves alone ------------------------------


def test_linked_input_is_not_prepared(stubs):
    p, _ = _prefetcher([_job("p1", _inputs(direction=["3", 0]))])
    assert p.look_ahead() == 0
    assert stubs["write_prompt"] == 0


def test_only_the_first_jobs_pending_are_prepared(stubs):
    p, _ = _prefetcher([_job("p1", _inputs("one")), _job("p2", _inputs("two"))], jobs=1)
    assert p.look_ahead() == 1
    assert stubs["write_prompt"] == 1
    p2, _ = _prefetcher([_job("p1", _inputs("one")), _job("p2", _inputs("two"))], jobs=2)
    assert p2.look_ahead() == 2


def test_a_key_build_produced_itself_is_not_prepared_again(stubs, monkeypatch):
    inputs = _inputs()
    p, state = _prefetcher([])
    monkeypatch.setattr(nodes, "PREFETCHER", p)
    nodes.MiniMaxH3ReferencePack().build(**inputs)
    # The same job re-queued (a seed change elsewhere in the graph): ComfyUI's own
    # cache serves it, so the provider must not be paid for a prompt nobody reads.
    state["pending"].append(_job("p2", inputs))
    assert p.look_ahead() == 0
    assert stubs["write_prompt"] == 1


def test_a_failed_prepare_is_not_cached_and_not_retried(stubs, monkeypatch):
    inputs = _inputs()
    good = nodes.prompt.write_prompt
    attempts = {"n": 0}

    def flaky(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise nodes.prompt.PromptError("openrouter returned 429: rate limited")
        return good(**kwargs)

    monkeypatch.setattr(nodes.prompt, "write_prompt", flaky, raising=False)
    p, _ = _prefetcher([_job("p1", inputs)])
    monkeypatch.setattr(nodes, "PREFETCHER", p)
    assert p.look_ahead() == 1  # attempted
    assert p.ready_keys() == []
    assert p.look_ahead() == 0  # not retried
    assert attempts["n"] == 1
    # The job itself retries when it runs, exactly as before.
    out = _by_name(nodes.MiniMaxH3ReferencePack().build(**inputs))
    assert out["prompt"].startswith("PROMPT#")
    assert attempts["n"] == 2


def test_eviction_spares_running_and_pending_jobs(stubs):
    inputs = _inputs()
    p, state = _prefetcher([_job("p1", inputs)], evict_after_s=0)
    assert p.look_ahead() == 1
    key = p.ready_keys()[0]
    # pending -> running: the worker took it, build() has not reached the node yet.
    state["pending"], state["running"] = [], [_job("p1", inputs)]
    p.look_ahead()
    assert p.ready_keys() == [key]
    # Reordered further down the queue than `jobs` reads: still pending, still kept.
    state["running"], state["pending"] = [], [_job("x", _inputs("other")), _job("p1", inputs)]
    p.look_ahead()
    assert key in p.ready_keys()
    # Gone from the queue altogether: dropped.
    state["pending"] = []
    p.look_ahead()
    assert key not in p.ready_keys()


def test_a_fresh_entry_survives_the_grace_period(stubs):
    now = [100.0]
    p, state = _prefetcher([_job("p1", _inputs())], evict_after_s=60, clock=lambda: now[0])
    p.look_ahead()
    state["pending"] = []
    now[0] += 30
    p.look_ahead()
    assert len(p.ready_keys()) == 1
    now[0] += 31
    p.look_ahead()
    assert p.ready_keys() == []


def test_budget_keeps_the_media_only_while_it_fits(stubs, monkeypatch):
    monkeypatch.setattr(nodes, "_media_bytes", lambda outputs: 3 * 2**30)
    one, two = _inputs("one"), _inputs("two")
    p, _ = _prefetcher([_job("p1", one), _job("p2", two)], jobs=2, budget_bytes=4 * 2**30)
    assert p.look_ahead() == 2
    assert p.retained_bytes() == 3 * 2**30
    with p.consume(nodes._key_for(**one)) as first:
        assert first is not None and first.outputs is not None
    with p.consume(nodes._key_for(**two)) as second:
        assert second is not None and second.outputs is None
        assert second.prompt_text == "PROMPT#2 for two"


def test_lookup_and_reservation_are_one_step(stubs):
    """The poll thread must not start a prepare between build()'s miss and its build."""
    inputs = _inputs()
    p, _ = _prefetcher([_job("p1", inputs)])
    key = nodes._key_for(**inputs)
    with p.consume(key) as entry:
        assert entry is None
        assert p.look_ahead() == 0  # reserved by the foreground build
        assert stubs["write_prompt"] == 0
    assert p.look_ahead() == 0  # and remembered afterwards
    assert stubs["write_prompt"] == 0


def test_a_reference_re_uploaded_while_waiting_discards_the_prepared_pack(stubs, monkeypatch):
    inputs = _inputs()
    p, _ = _prefetcher([_job("p1", inputs)])
    monkeypatch.setattr(nodes, "PREFETCHER", p)
    assert p.look_ahead() == 1
    real = nodes._key_for
    seen = []

    def key_that_moves(**kwargs):
        # First call: the key build() looked up. Second: after the (simulated) wait,
        # the file on disk has a new mtime+size and the key no longer matches.
        seen.append(1)
        return real(**kwargs) if len(seen) == 1 else real(**kwargs) + "-reuploaded"

    monkeypatch.setattr(nodes, "_key_for", key_that_moves)
    out = _by_name(nodes.MiniMaxH3ReferencePack().build(**inputs))
    assert out["prompt"] == "PROMPT#2 for a girl walks"  # rebuilt, not served stale
    assert "prefetch: no" in out["debug"]
    assert stubs == {"write_prompt": 2, "load_video": 2}
    assert p.ready_keys() == []  # the stale entry was popped, not kept


def test_key_is_collision_safe_and_fingerprints_the_credential(fake_folder_paths):
    common = dict(references_json="", openrouter_model="m", prompt_provider="none")
    a = nodes._key_for(direction="x|||y", system_prompt="z", **common)
    b = nodes._key_for(direction="x", system_prompt="y|||z", **common)
    assert a != b
    with_key = nodes._key_for(direction="d", openrouter_api_key="sk-secret-A", **common)
    other_key = nodes._key_for(direction="d", openrouter_api_key="sk-secret-B", **common)
    no_key = nodes._key_for(direction="d", openrouter_api_key="", **common)
    assert len({with_key, other_key, no_key}) == 3
    assert "sk-secret-A" not in with_key


def test_an_unreadable_queue_still_prunes_by_age(stubs):
    now = [0.0]
    p, _ = _prefetcher([_job("p1", _inputs())], max_age_s=100, clock=lambda: now[0])
    assert p.look_ahead() == 1

    def broken():
        raise RuntimeError("sqlite closed")

    p._pending_fn = broken
    now[0] = 50
    assert p.look_ahead() == 0  # logged, not raised
    assert len(p.ready_keys()) == 1  # nothing known about liveness: kept
    now[0] = 101
    p.look_ahead()
    assert p.ready_keys() == []


def test_consume_prunes_by_age_without_the_thread(stubs):
    now = [0.0]
    p, _ = _prefetcher([_job("p1", _inputs())], max_age_s=100, clock=lambda: now[0])
    p.look_ahead()
    now[0] = 500
    with p.consume("some other key") as entry:
        assert entry is None
    assert p.ready_keys() == []


# ---- the queue view, in both shapes ------------------------------------------------------


def _item(number, prompt_id, prompt):
    return (number, prompt_id, prompt, {}, ["8"], {})


def _fake_server(monkeypatch, queue):
    class PromptServer:
        instance = types.SimpleNamespace(prompt_queue=queue)

    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=PromptServer))


def test_queue_view_reads_core_heap_in_execution_order(monkeypatch):
    class Core:
        def get_current_queue(self):
            # A heap list is not sorted; the view must be.
            return ([_item(1, "run", {"a": {}})],
                    [_item(5, "later", {"c": {}}), _item(3, "next", {"b": {}}), "junk"])

    _fake_server(monkeypatch, Core())
    running, pending = prefetch.queue_view()
    assert running == [("run", {"a": {}})]
    assert [pid for pid, _ in pending] == ["next", "later"]


def test_queue_view_reads_queue_manager_pages(monkeypatch):
    seen = {}

    class QM:
        def get_current_queue(self, page=0, page_size=0, route="queue", filters=None, return_meta=False):
            seen["args"] = (page, page_size)
            # Queue Manager stores items as JSON lists, already ordered by number.
            return ([], [list(_item(2, "next", {"b": {}})), list(_item(4, "later", {"c": {}}))])

    _fake_server(monkeypatch, QM())
    running, pending = prefetch.queue_view(limit=7)
    assert seen["args"] == (0, 7)
    assert running == []
    assert [pid for pid, _ in pending] == ["next", "later"]


# ---- installing onto the running queue --------------------------------------------------


def _noop_prefetcher(monkeypatch):
    p = prefetch.Prefetcher(key_fn=lambda **k: "k", build_fn=lambda **k: None, pending_fn=lambda: ([], []))
    monkeypatch.setattr(p, "start", lambda: None)  # no thread in a test
    return p


def test_install_on_wraps_get_and_wakes_the_thread_when_a_job_starts(monkeypatch):
    class Queue:
        def __init__(self):
            self.calls = []

        def get(self, timeout=None):
            self.calls.append(timeout)
            return ("item", 1)

    queue = Queue()
    p = _noop_prefetcher(monkeypatch)
    assert prefetch.install_on(queue, p) is True
    assert queue.get(timeout=5) == ("item", 1)
    assert queue.calls == [5]
    assert p._wake.is_set()
    # Idempotent: a second install (a reload) does not wrap the wrapper.
    assert prefetch.install_on(queue, p) is False


def test_install_is_off_under_the_environment_switch(monkeypatch):
    p = _noop_prefetcher(monkeypatch)
    assert prefetch.install(p, env={prefetch.ENABLED_ENV: "0"}) is False
    assert prefetch.enabled({prefetch.ENABLED_ENV: "off"}) is False
    assert prefetch.enabled({prefetch.ENABLED_ENV: "1"}) is True
    assert prefetch.enabled({}) is True


def test_from_env_parses_and_bounds_the_knobs():
    p = prefetch.Prefetcher.from_env(
        key_fn=lambda **k: "k", build_fn=lambda **k: None,
        env={prefetch.BUDGET_ENV: "2", prefetch.JOBS_ENV: "20", prefetch.POLL_ENV: "0.5"},
    )
    assert p.budget_bytes == 2 * 2**30
    assert p.jobs == 8  # clamped to the ceiling
    assert p.poll_s == prefetch.DEFAULT_POLL_S  # below the floor: ignored, not clamped
    bad = prefetch.Prefetcher.from_env(
        key_fn=lambda **k: "k", build_fn=lambda **k: None,
        env={prefetch.BUDGET_ENV: "lots", prefetch.JOBS_ENV: "-1"},
    )
    assert bad.budget_bytes == prefetch.DEFAULT_BUDGET_BYTES
    assert bad.jobs == prefetch.DEFAULT_JOBS
