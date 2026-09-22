---
name: tldrgraph-refresh
description: Refresh source-backed feature workflows after repository source changes
---

# TLDRGraph: refresh the source-backed workflow catalog

Run exactly:

```bash
tldrgraph refresh
```

When the status is `needs_feature_workflows`:

1. Read the `source_hash` and validation status returned by `tldrgraph refresh`.
2. Identify feature outcomes, define areas, and write the v4
   `.tldrgraph/features.yaml` index before delegating workflow research.
3. Spawn one fresh source-reading subagent for each indexed feature. Never assign the entire catalog to one subagent.
4. Have each worker write only its evidence-backed v4 workflow to its assigned
   `.tldrgraph/workflows/<feature_id>.yaml` file; do not collect or rewrite it.
5. Run `tldrgraph refresh` again.

Continue until the status is `done`.
