"""Parser for explicit executable skill workflows."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag


def build_workflow_parser(subparsers, *, cmd_workflow: Callable) -> None:
    parser = subparsers.add_parser("workflow", help="Inspect, generate, and run skill workflows")
    sub = parser.add_subparsers(dest="workflow_action")
    listing = sub.add_parser("list", help="List workflows for a skill or profile")
    listing.add_argument("skill", nargs="?")
    listing.add_argument("--all", action="store_true", help="List every existing workflow in the selected profile")
    show = sub.add_parser("show", help="Show a workflow")
    show.add_argument("skill", nargs="?", help="Skill name")
    show.add_argument("--name", default="default")
    show.add_argument("--format", choices=("text", "yaml", "json", "mermaid"), default="text")
    validate = sub.add_parser("validate", help="Validate a workflow")
    validate.add_argument("skill")
    validate.add_argument("--name", default="default")
    generate = sub.add_parser("generate", help="Generate a workflow sidecar for a skill")
    generate.add_argument("skill", nargs="?")
    generate.add_argument("--name", default="default")
    generate.add_argument("--all", action="store_true", help="Generate missing or stale workflows for Hermes-created skills")
    generate.add_argument("--include-local", action="store_true", help="With --all, include user-created and learned profile-local skills")
    generate.add_argument("--include-prebuilt", action="store_true", help="With --all, include bundled, Hub, and external skills")
    generate.add_argument("--force", action="store_true", help="Regenerate workflows that are already valid")
    generate.add_argument("--dry-run", action="store_true", help="List skills that would be generated without calling a model")
    generate.add_argument("--continue-on-error", action="store_true", help="Continue the batch after a generation failure")
    add_json_flag(generate, "Output one result object per skill")
    activate = sub.add_parser("activate", help="Activate a generated workflow candidate")
    activate.add_argument("skill")
    activate.add_argument("--name", default="default")
    runs = sub.add_parser("runs", help="List workflow run receipts")
    runs.add_argument("skill", nargs="?")
    runs.add_argument("--limit", type=int, default=50)
    inspect = sub.add_parser("inspect", help="Inspect a workflow run and its events")
    inspect.add_argument("run_id")
    resume = sub.add_parser("resume", help="Resume a paused workflow from its saved checkpoint")
    resume.add_argument("run_id")
    resume.add_argument("--decision", required=True, choices=("accept", "continue", "stop", "retry"))
    add_json_flag(resume, "Output the updated run receipt as JSON")
    run = sub.add_parser("run", help="Run an explicit skill workflow")
    run.add_argument("skill")
    run.add_argument("--name", default="default")
    run.add_argument("--input-json", default="{}")
    run.add_argument("--step-model", action="append", default=[], metavar="STEP=PROVIDER/MODEL")
    run.add_argument("--step-reasoning", action="append", default=[], metavar="STEP=EFFORT")
    add_json_flag(run, "Output the run receipt as JSON")
    parser.set_defaults(func=cmd_workflow)
