---
title: "Executable Skill Workflows"
description: "Run, inspect, and visualize versioned skill workflows."
---

# Executable Skill Workflows

Skills without an active workflow remain flexible instructions. A workflow is an explicit,
versioned execution graph for a skill: each step receives structured data, can select a different
model, and records a JSONL receipt for later evaluation.

```bash
hermes workflow list research
hermes workflow generate research
hermes workflow activate research
hermes workflow show research --format mermaid
hermes workflow run research --input-json '{"request":"summarize this topic"}' --json
hermes workflow runs research --limit 10
hermes workflow inspect <run-id>
hermes workflow resume <run-id> --decision accept
```

Workflow files live in `workflows/*.yaml` beside a skill. Generated workflows use a profile-local
overlay, so Hermes never changes the installed package. An explicit single `/<skill-name> …`
invocation checks for a workflow before the parent model runs. For ordinary chat, Hermes performs
the same check after the parent model successfully selects one skill with `skill_view`; the workflow
then receives the current request plus bounded prior user context. Both paths prefer a fresh active
overlay, then a valid packaged workflow. If neither is usable, Hermes immediately follows the prose
skill and queues profile-scoped compilation in the background. Hermes-created and learned skills
activate that result automatically; workflows generated for user-owned, bundled, Hub, or external
skills remain candidates until you activate them. Linked-file `skill_view` calls, auto-loaded skills,
bundles, and stacked skills keep their existing prose behavior. Set `skills.workflow_execution: off`
to disable both preference and background compilation.

Workflows use `hermes/workflow@2` and are authored as flowcharts: every node is a `task`,
`decision`, or `loop`. Tasks have one `next` target. Decisions have named `branches`, and their
`output_schema.properties.transition.enum` must exactly name every branch. Loops additionally
declare `repeat`, `max_iterations`, and `on_exhausted`, making retries and their stopping path
visible and bounded. Repeated loop calls receive the preceding result as `previous_output` so they
can accumulate work. Do not bind `$input.previous_output` in a loop's declared input: it is absent
on the first visit and supplied automatically on later visits. Step bindings must name required
top-level inputs or outputs from steps guaranteed to have run.
Side-effect steps may declare `preconditions` mapping input fields to required scalar values;
the runner checks them before starting the step.
An empty top-level `output` returns the terminal branch's structured output. The renderer draws
tasks as rectangles, decisions as diamonds, and loops as labelled loop nodes; undeclared cycles are
rejected.

Agent-created and learned skills can declare an executable flow skeleton in `SKILL.md`:

~~~~markdown
## Workflow contract
```yaml
entry: collect
steps:
  collect:
    kind: task
    next: check
    max_retries: 1
    on_error: report_blocker
  check:
    kind: decision
    branches: {enough: finish, shortfall: report_blocker}
  finish: {kind: task, next: END, terminal_outcome: success}
  report_blocker: {kind: task, next: END, terminal_outcome: blocked}
```
~~~~

The compiler fills in each node's instructions, bindings, schemas, and toolsets, but preserves
the declared nodes and edges. `skill_manage` requires this contract when an agent creates a skill
and preserves it during later edits. New workflow generation and explicit workflow writes need this
contract, including for older skills when they are regenerated. Generated graphs get up to three
attempts to satisfy it. An explicitly written workflow
is checked against the same contract before acceptance. Existing skills without a contract keep
their previous workflow files, but new starts require the current reliability contract. Every
generated step declares `effect` (`read`, `write`, or `external_action`), `max_retries` of at least
one, and a `recovery` method. The compiler chooses the bounded retry count for each step. Reads
can retry; writes and external actions must reconcile a stable operation ID and observed state
before another attempt. An unknown state pauses the run. `on_error` is a recovery/reporting edge
after attempts fail, separate from a loop's `on_exhausted` path.

Skills with a measurable target declare `goal_gate` with target and actual bindings. The runner
compares those values before delivery. A shortfall pauses with a structured `partial` outcome and
offers `accept`, `continue`, or `stop`. A repairable `blocked` result offers `retry` at its saved
check step. These decisions persist in `checkpoint.json` beside the receipt and can be submitted
later through `hermes workflow resume <run-id> --decision <choice>`, chat with
`workflow resume <run-id> <choice>`, or the Desktop workflow view. Resume uses the saved workflow
snapshot, including after skill regeneration or an app restart.
In chat, “try again” continues the most recent paused partial run or retries the most recent
repairable blocked run in the same session.

Use `--step-model step=provider/model` and `--step-reasoning step=high` to compare model choices
per step. Each invocation is stored under
`~/.hermes/workflow-runs/YYYY-MM/<run-id>/` with an append-only `events.jsonl`, derived `run.json`,
and the exact normalized `workflow.yaml` snapshot. Receipts record routes, allowed and used tools,
redacted structured data, transitions, timing, schema results, and final conformance. Prose
fallbacks receive terminal receipts too, including the reason and background generation job ID.
Retention defaults to 30 days and 1,000 terminal runs and is configurable with
`skills.workflow_receipt_retention_days` and `skills.workflow_receipt_max_runs`.

The Desktop app shows the same graph and source YAML in **Capabilities → Workflows**, polls active
runs, colors the executed path, separates fallback/generation state, and can activate candidates.
