---
name: tldrgraph-init
description: Build or refresh source-backed feature workflows
---

# TLDRGraph: build the source-backed workflow catalog

For a first catalog, run:

```bash
tldrgraph init
```

When the status is `needs_feature_workflows`:

1. Read the `source_hash` and validation status returned by the command.
2. Identify feature outcomes, define areas, and write the v4
   `.tldrgraph/features.yaml` index before delegating workflow research.
3. Spawn one fresh source-reading subagent for each indexed feature. Never assign the entire catalog to one subagent.
4. Have each worker write only its evidence-backed v4 workflow to its assigned
   `.tldrgraph/workflows/<feature_id>.yaml` file; do not collect or rewrite it.
5. Run the same command again. Use `tldrgraph refresh` when updating an existing catalog after source changes.

Continue until the status is `done`.
