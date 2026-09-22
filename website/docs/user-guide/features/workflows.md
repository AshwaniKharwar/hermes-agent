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

Use `--step-model step=provider/model` and `--step-reasoning step=high` to compare model choices
per step. Receipts are written under `~/.hermes/workflow-runs/` and include the selected route,
structured input/output, transitions, and timing. They are intended as stable input to a later
evaluation harness, not a scoring system by themselves.

The Desktop app shows the same Mermaid graph in **Capabilities → Skills**. Viewing it is read-only.
