"""hermes_laya — native Hermes plugin exposing the Laya System 1 decision engine.

Two-step registration (same pattern as the other standalone Hermes plugins):
1. Importing laya_tool runs its module-level registry.register(), so the tool
   exists in the registry with its canonical schema/handler/check_fn.
2. We re-register the same tool through ctx.register_tool(), which marks it
   plugin-owned and lets the plugin loader auto-enable the `laya` toolset.

No core Hermes files are touched.
"""

from __future__ import annotations

try:
    from . import laya_tool  # noqa: F401 — module-level registry.register() fires here
except Exception as _e:  # noqa: BLE001 — a broken tool must not hide the rest
    print(f"[hermes_laya] WARNING: failed to import laya_tool: {_e}")

_TOOLS = [("laya", "laya")]


def register(ctx) -> None:
    """Register the tool via the plugin API, copying the live registry entry."""
    from tools.registry import registry

    for name, toolset in _TOOLS:
        entry = registry.get_entry(name)
        if entry is None:
            continue
        ctx.register_tool(
            name=name,
            toolset=toolset,
            schema=entry.schema,
            handler=entry.handler,
            check_fn=entry.check_fn,
            is_async=entry.is_async,
            emoji=entry.emoji or "",
        )
