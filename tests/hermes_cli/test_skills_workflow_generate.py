"""Manual workflow enrollment uses the active profile and existing skill resolver."""

import argparse
from pathlib import Path

from agent.skill_workflow_store import required_records, save_record
from hermes_cli.skills_hub import handle_skills_slash, skills_command
from hermes_cli.subcommands.skills import build_skills_parser


def _skill(home: Path, name: str) -> None:
    directory = home / "skills" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use {name}.\n---\n\n# Steps\n\n1. Inspect.\n",
        encoding="utf-8",
    )


def test_manual_generate_one_then_all_uses_cli_and_slash(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _skill(home, "alpha")
    _skill(home, "beta")
    parser = argparse.ArgumentParser()
    build_skills_parser(parser.add_subparsers(dest="command"), cmd_skills=skills_command)

    args = parser.parse_args(["skills", "workflow", "generate", "alpha"])
    args.func(args)
    assert {record["name"] for _, record in required_records()} == {"alpha"}
    assert "Start a chat turn" in capsys.readouterr().out
    directory, record = required_records()[0]
    save_record(directory, {**record, "status": "failed", "completed_failures": 3})
    args.func(args)
    assert required_records()[0][1]["completed_failures"] == 0

    handle_skills_slash("/skills workflow generate --all")
    assert {record["name"] for _, record in required_records()} == {"alpha", "beta"}
    assert not any((directory / "graph.json").exists() for directory, _ in required_records())


def test_manual_generate_is_profile_scoped_a_b_a(tmp_path, monkeypatch, request):
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.skills_hub import do_workflow_generate

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = [tmp_path / ".hermes", tmp_path / ".hermes" / "profiles" / "b"]
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    for home in homes:
        _skill(home, "same")
    set_multiplex_active(True)
    request.addfinalizer(lambda: set_multiplex_active(False))
    try:
        for home in (homes[0], homes[1], homes[0]):
            token = set_hermes_home_override(home)
            try:
                do_workflow_generate("same")
                assert len(required_records()) == 1
                assert required_records()[0][0].is_relative_to(home / "workflows")
            finally:
                reset_hermes_home_override(token)
    finally:
        set_multiplex_active(False)
