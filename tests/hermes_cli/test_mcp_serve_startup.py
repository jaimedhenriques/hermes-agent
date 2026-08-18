"""Regression coverage for dedicated ``hermes mcp serve`` startup."""

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
    import hermes_cli.main as main_mod

    assert main_mod._command_has_dedicated_mcp_startup(
        _serve_args(mcp_action=mcp_action)
    ) is False


def test_prepare_agent_startup_skips_inline_discovery_for_mcp_serve(monkeypatch):
    import hermes_cli.main as main_mod

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(start_background_plugin_discovery=lambda: None),
    )
    called: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(
            discover_mcp_tools=lambda: called.append("discover_mcp_tools")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.mcp_startup",
        types.SimpleNamespace(
            start_background_mcp_discovery=lambda **_kwargs: called.append(
                "start_background_mcp_discovery"
            )
        ),
    )

    main_mod._prepare_agent_startup(_serve_args())

    assert called == []
