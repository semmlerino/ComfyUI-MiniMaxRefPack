"""ComfyUI-MiniMaxRefPack — reference management for MiniMax H3 Reference-to-Video."""

# ComfyUI imports this file as a package, so the relative form is the real one. pytest
# walks the repo root as a package too and imports this file with no parent, hence the
# absolute fallback (the repo root is on sys.path during tests, see conftest.py).
# Importing `routes` is what registers the HTTP routes.
try:
    from .minimax_refpack import prefetch, routes
    from .minimax_refpack.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, PREFETCHER
except ImportError:  # pragma: no cover - test-time import path
    from minimax_refpack import prefetch, routes
    from minimax_refpack.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, PREFETCHER

# Builds the next queued job's reference pack while the current one renders. A no-op
# outside a running ComfyUI (no `server` module) and under MINIMAX_REFPACK_PREFETCH=0.
prefetch.install(PREFETCHER)

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "PREFETCHER", "WEB_DIRECTORY", "routes"]
