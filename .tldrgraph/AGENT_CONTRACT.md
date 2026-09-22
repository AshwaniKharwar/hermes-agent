# TLDRGraph Agent Contract

TLDRGraph builds a source-backed feature catalog and Workflow Explorer. It does
not construct an architecture graph and does not launch an AI process itself.

## Direct artifact workflow

Run `tldrgraph init` to create a catalog. Run `tldrgraph refresh` to update an
existing catalog after source changes. A missing, invalid, or stale catalog returns
`needs_feature_workflows` and prints the current `source_hash`. The active coding
agent identifies feature outcomes and writes the catalog index first. It then
delegates each indexed outcome to a separate source-reading subagent. Each
worker writes only its assigned workflow file; the active agent does not collect
worker objects or write workflows:

```text
.tldrgraph/features.yaml
.tldrgraph/workflows/<feature_id>.yaml
```

Each final artifact must use the reported source hash. Run the command that
reported the status again; TLDRGraph validates the direct artifacts against the current source
inventory and generates the visualizer.

## Evidence

Every feature and every displayed workflow step must have at least one evidence
record:

```yaml
file: relative/path/to/source.py
symbol: verified_symbol
line: 12
code_start: 12
code_end: 28
```

Paths must be repository-relative and present in the current source inventory.
Line ranges must exist in the current file. Never invent evidence.

When one workflow step has mutually exclusive paths, modes, or choices, add an
`options` list to that step. Each option becomes its own flow-chart node and must
include its own `title`, `text`, and verified `evidence`. For example, a runtime
bootstrap step with Docker and Kubernetes paths should use two options, not one
step with both paths hidden as references.

## Catalog rules

- Areas have a `product` or `technical` perspective and an order within it.
- Features describe user, admin, developer, or operator outcomes—not code symbols.
- A `generated` workflow proves the complete journey through its final response
  or visible result.
- A `partial` workflow contains only proven steps and explains the absent portion
  in `missing_coverage`.
- A `pending` workflow has no steps and explains what could not be established.
- User-facing flows begin at the initiating action and include the client request,
  backend work, response, and UI update when those stages exist.
- Alternate paths are represented as step `options` so the explorer can draw
  separate branch nodes that reconnect to the following step.

`features.yaml` uses schema `tldrgraph/features@4`, generator
`feature-catalog-agent@4`, the reported `source_hash`, areas, and lightweight
feature index metadata. Each worker-owned workflow uses schema
`tldrgraph/feature-workflow@4`, generator `feature-workflow-subagent@4`, the
reported source hash, and all status and evidence. Continue until `tldrgraph
init` or `tldrgraph refresh` reports `done`, then run `tldrgraph ui --serve`.
