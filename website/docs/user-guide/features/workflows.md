---
title: "Executable Skill Workflows"
description: "Run, inspect, and visualize versioned skill workflows."
---

# Executable Skill Workflows

Skills remain flexible instructions during ordinary conversations. A workflow is an explicit,
versioned execution graph for a skill: each step receives structured data, can select a different
model, and records a JSONL receipt for later evaluation.

```bash
hermes workflow list research
hermes workflow generate research
hermes workflow show research --format mermaid
hermes workflow run research --input-json '{"request":"summarize this topic"}' --json
```

Workflow files live in `workflows/*.yaml` beside a writable skill. Hub and external skills use a
profile-local overlay instead, so Hermes never changes the installed package. A missing or stale
workflow is generated and validated before an explicit run; ordinary skill loading never generates
or runs one.

Workflows use `hermes/workflow@2` and are authored as flowcharts: every node is a `task`,
`decision`, or `loop`. Tasks have one `next` target. Decisions have named `branches`, and their
`output_schema.properties.transition.enum` must exactly name every branch. Loops additionally
declare `repeat`, `max_iterations`, and `on_exhausted`, making retries and their stopping path
visible and bounded. The renderer draws tasks as rectangles, decisions as diamonds, and loops as
labelled loop nodes; undeclared cycles are rejected.

Use `--step-model step=provider/model` and `--step-reasoning step=high` to compare model choices
per step. Receipts are written under `~/.hermes/workflow-runs/` and include the selected route,
structured input/output, transitions, and timing. They are intended as stable input to a later
evaluation harness, not a scoring system by themselves.

The Desktop app shows the same Mermaid flowchart and its source YAML in **Capabilities → Workflows**.
Viewing it is read-only.
