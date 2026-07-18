"""Tests that `hermes mcp serve` skips inline client-side MCP discovery.

`mcp serve` exposes Hermes conversations *as* an MCP server.  It is listed in
`_AGENT_SUBCOMMANDS`, so it reaches `_prepare_agent_startup()`, but its tool
surface is the messaging bridge rather than downstream MCP tools.  Running the
inline (blocking) discovery there dials every configured MCP server before the
stdio handshake can complete, which pushed startup to ~60s locally and past the
30s timeout MCP clients use -- Claude Code reports "Failed to connect".
"""

from __future__ import annotations

import sys
import types

import pytest


def _serve_args(**overrides):
    args = types.SimpleNamespace(
        command="mcp",
        mcp_action="serve",
        safe_mode=False,
        yolo=False,
        tui=False,
        accept_hooks=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_mcp_serve_has_dedicated_mcp_startup():
    import hermes_cli.main as main_mod

    assert main_mod._command_has_dedicated_mcp_startup(_serve_args()) is True


@pytest.mark.parametrize("mcp_action", ["list", "add", "test", "catalog"])
def test_other_mcp_subcommands_are_not_claimed(mcp_action):
    """Only `serve` runs its own server; the rest keep prior behaviour."""
    import hermes_cli.main as main_mod

    args = _serve_args(mcp_action=mcp_action)
    assert main_mod._command_has_dedicated_mcp_startup(args) is False


def test_prepare_agent_startup_skips_inline_discovery_for_mcp_serve(monkeypatch):
    """The regression guard: no blocking discover_mcp_tools() before serving."""
    import hermes_cli.main as main_mod

    plugins = types.ModuleType("hermes_cli.plugins")
    setattr(plugins, "discover_plugins", lambda: None)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", plugins)

    called: list[str] = []
    mcp_tool = types.ModuleType("tools.mcp_tool")

    def discover_mcp_tools() -> None:
        called.append("discover_mcp_tools")

    setattr(mcp_tool, "discover_mcp_tools", discover_mcp_tools)
    monkeypatch.setitem(sys.modules, "tools.mcp_tool", mcp_tool)

    mcp_startup = types.ModuleType("hermes_cli.mcp_startup")

    def start_background_mcp_discovery(**_kwargs):
        called.append("start_background_mcp_discovery")

    setattr(mcp_startup, "start_background_mcp_discovery", start_background_mcp_discovery)
    monkeypatch.setitem(sys.modules, "hermes_cli.mcp_startup", mcp_startup)

    main_mod._prepare_agent_startup(_serve_args())

    assert called == []
