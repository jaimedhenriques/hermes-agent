"""Behavioral coverage for top-level one-shot skill preloading."""

from __future__ import annotations

import sys
import types


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_normalize_skills_flattens_repeated_and_comma_separated_values():
    from hermes_cli.oneshot import _normalize_skills

    assert _normalize_skills(
        [" first-skill, second-skill ", "first-skill", "", "third-skill,"]
    ) == ["first-skill", "second-skill", "third-skill"]


def test_oneshot_fails_closed_when_every_requested_skill_is_missing(
    monkeypatch, capsys
):
    import agent.skill_commands as skill_commands
    import hermes_cli.oneshot as oneshot

    requested = []
    agent_called = []

    def fake_build(skills, task_id=None):
        requested.extend(skills)
        assert task_id is None
        return "", [], list(skills)

    monkeypatch.setattr(skill_commands, "build_preloaded_skills_prompt", fake_build)
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: agent_called.append(True) or ("unexpected", {}),
    )

    assert oneshot.run_oneshot(
        "hello", skills=[" missing-skill,other-missing ", "missing-skill"]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "hermes -z: --skills did not load any requested skills: "
        "missing-skill, other-missing.\n"
    )
    assert requested == ["missing-skill", "other-missing"]
    assert agent_called == []


def test_oneshot_fails_closed_when_loaded_skills_have_no_prompt(monkeypatch, capsys):
    import agent.skill_commands as skill_commands
    import hermes_cli.oneshot as oneshot

    agent_called = []
    monkeypatch.setattr(
        skill_commands,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: ("", ["present-skill"], []),
    )
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: agent_called.append(True) or ("unexpected", {}),
    )

    assert oneshot.run_oneshot("hello", skills=["present-skill"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "hermes -z: --skills loaded no system-prompt content; refusing to run.\n"
    )
    assert agent_called == []


def test_oneshot_warns_deterministically_and_continues_after_partial_skill_load(
    monkeypatch, capsys
):
    import agent.skill_commands as skill_commands
    import hermes_cli.oneshot as oneshot

    exact_skill_prompt = "[skill wrapper]\n\n# Present Skill\n\nEXACT SKILL BODY"
    captured_call = {}
    monkeypatch.setattr(
        skill_commands,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: (
            exact_skill_prompt,
            ["present-skill"],
            ["missing-skill"],
        ),
    )

    def fake_run_agent(prompt, **kwargs):
        captured_call.update({"prompt": prompt, **kwargs})
        return "done", {"failed": False, "partial": False}

    monkeypatch.setattr(oneshot, "_run_agent", fake_run_agent)

    assert oneshot.run_oneshot(
        "hello", skills=["present-skill,missing-skill"]
    ) == 0
    captured = capsys.readouterr()
    assert captured.out == "done\n"
    assert captured.err == (
        "hermes -z: unavailable --skills entries: missing-skill. "
        "Continuing with: present-skill.\n"
    )
    assert captured_call["prompt"] == "hello"
    assert captured_call["system_message"] == exact_skill_prompt


def test_main_oneshot_wrapper_forwards_repeated_skill_flags(monkeypatch):
    import hermes_cli.main as main_mod
    import hermes_cli.oneshot as oneshot

    captured = {}
    monkeypatch.setattr(
        oneshot,
        "run_oneshot",
        lambda prompt, **kwargs: captured.update({"prompt": prompt, **kwargs}) or 0,
    )
    monkeypatch.setattr(main_mod, "_cleanup_oneshot_runtime", lambda: None)
    monkeypatch.setattr(
        main_mod,
        "_exit_after_oneshot",
        lambda rc: captured.update({"exit_code": rc}),
    )

    main_mod._run_and_exit_oneshot(
        "hello",
        skills=["first-skill,second-skill", "first-skill"],
    )

    assert captured["skills"] == [
        "first-skill,second-skill",
        "first-skill",
    ]
    assert captured["exit_code"] == 0


def test_run_agent_persists_exact_skill_prompt_in_cached_system_message(monkeypatch):
    from agent.conversation_loop import _restore_or_build_system_prompt
    from hermes_cli.oneshot import _run_agent

    exact_skill_prompt = "[skill wrapper]\n\n# Loaded Skill\n\nEXACT SKILL BODY"
    persisted = {}
    captured = {}

    class FakeSessionDB:
        def update_system_prompt(self, session_id, system_prompt):
            persisted[session_id] = system_prompt

        def close(self):
            pass

    session_db = FakeSessionDB()

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._session_db = kwargs["session_db"]
            self.session_id = "oneshot-session"
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.platform = kwargs["platform"]
            self._cached_system_prompt = None
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def _build_system_prompt(self, system_message=None):
            return f"BASE SYSTEM\n\n{system_message}" if system_message else "BASE SYSTEM"

        def run_conversation(self, prompt, system_message=None, **_kwargs):
            captured["prompt"] = prompt
            captured["system_message"] = system_message
            _restore_or_build_system_prompt(self, system_message, None)
            captured["cached_system_prompt"] = self._cached_system_prompt
            return {"final_response": "ok", "failed": False, "partial": False}

        def shutdown_memory_provider(self, *_args):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules, "hermes_state", _module("hermes_state", SessionDB=lambda: session_db)
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module(
            "hermes_cli.models",
            detect_provider_for_model=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "requested_provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda *_args, **_kwargs: {"terminal"},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.mcp_startup",
        _module(
            "hermes_cli.mcp_startup",
            ensure_mcp_discovery_before_agent_build=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.lifecycle",
        _module("hermes_cli.lifecycle", invoke_hook=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.credits_tracker",
        _module(
            "agent.credits_tracker",
            seed_credits_at_session_start=lambda _agent: None,
        ),
    )

    text, result = _run_agent("do the work", system_message=exact_skill_prompt)

    assert text == "ok"
    assert result["failed"] is False
    assert captured["system_message"] == exact_skill_prompt
    expected = f"BASE SYSTEM\n\n{exact_skill_prompt}"
    assert captured["cached_system_prompt"] == expected
    assert persisted == {"oneshot-session": expected}
