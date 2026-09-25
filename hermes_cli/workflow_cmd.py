"""CLI implementation for explicit, executable skill workflows."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from agent.workflows import (WorkflowError, activate_workflow, load_workflow, render_mermaid, save_overlay,
                             TERMINAL_REASON_CODES, validate_workflow, validate_workflow_contract, workflow_contract,
                             workflow_source_fingerprint, workflow_sources)
from agent.workflow_receipts import inspect_workflow_run, list_workflow_runs


def _content(response: Any) -> str:
    try:
        return str(response.choices[0].message.content or "")
    except (AttributeError, IndexError, TypeError):
        return ""


def _json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise WorkflowError("Model response must be a JSON object.")
    return value


def _overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise WorkflowError(f"Expected STEP=VALUE, got {value!r}.")
        step, setting = value.split("=", 1)
        if not step or not setting:
            raise WorkflowError(f"Expected STEP=VALUE, got {value!r}.")
        result[step] = setting
    return result


def _workflow_response_format(skill: str, contract: Mapping[str, Any]) -> dict:
    """Constrain the compiler to JSON with exactly the declared node IDs."""
    steps = contract["steps"]
    return {"type": "json_schema", "json_schema": {
        "name": "skill_workflow", "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "schema": {"type": "string", "enum": ["hermes/workflow@2"]},
                "id": {"type": "string"},
                "skill": {"type": "string", "enum": [skill]},
                "entry": {"type": "string", "enum": [contract["entry"]]},
                "max_transitions": {"type": "integer"},
                "input_schema": {"type": "object"},
                "defaults": {"type": "object"},
                "steps": {"type": "object", "properties": {name: {"type": "object"} for name in steps},
                          "required": list(steps), "additionalProperties": False},
                "goal_gate": {"type": "object"},
                "output": {"type": "object"},
            },
            "required": ["schema", "id", "skill", "entry", "max_transitions", "input_schema", "steps"],
            "additionalProperties": False,
        },
    }}


def _normalize_generated_contract_fields(workflow: dict, contract: Mapping[str, Any]) -> None:
    """Apply contract-owned labels and safety floors to a model draft."""
    for name, declared in contract["steps"].items():
        step = workflow.get("steps", {}).get(name)
        if not isinstance(step, dict):
            continue
        if step.get("max_retries", 0) == 0 and "max_retries" not in declared:
            step["max_retries"] = 1
        if declared.get("kind") == "decision":
            step["effect"] = "read"
            step["recovery"] = {"method": "retry"}
        outcome = declared.get("terminal_outcome")
        reason = step.get("terminal_reason")
        if outcome in TERMINAL_REASON_CODES and isinstance(reason, dict):
            if reason.get("code") not in TERMINAL_REASON_CODES[outcome]:
                reason["code"] = "goal_unmet" if outcome == "partial" else "blocked_condition"
                reason["action"] = "Review the observed evidence and repair the blocking condition."
        if declared.get("kind") not in {"decision", "loop"}:
            continue
        schema = step.get("output_schema")
        if not isinstance(schema, dict):
            continue
        properties = schema.setdefault("properties", {})
        if not isinstance(properties, dict):
            continue
        properties["transition"] = {"type": "string", "enum": list(declared["branches"])}
        required = schema.setdefault("required", [])
        if isinstance(required, list) and "transition" not in required:
            required.append("transition")


def _generate(skill: str, name: str, *, activation: str = "active",
              expected_fingerprint: str | None = None, expected_job_id: str | None = None,
              progress: Callable[[str], None] | None = None) -> dict:
    from agent.auxiliary_client import call_llm
    from tools.skill_manager_tool import _find_skill

    found = _find_skill(skill)
    if not found:
        raise WorkflowError(f"Skill {skill!r} was not found.")
    if progress:
        progress("Reading skill and workflow contract")
    skill_dir = Path(found["path"])
    contract_data = workflow_contract(skill_dir)
    if contract_data is None:
        raise WorkflowError("Workflow generation needs a ## Workflow contract YAML block in SKILL.md.")
    skill_md = skill_dir / "SKILL.md"
    # Compile the actual procedure, including bounded textual progressive-
    # disclosure files. Assets and existing workflows cannot improve a first
    # generation and would only turn this into an unbounded prompt.
    sources = [("SKILL.md", skill_md)]
    for directory in ("references", "templates", "scripts"):
        sources.extend((str(path.relative_to(skill_dir)), path) for path in sorted((skill_dir / directory).rglob("*"))
                       if path.is_file() and path.suffix.lower() in {".md", ".txt", ".yaml", ".yml", ".json", ".py", ".sh"})
    remaining, fragments = 100_000, []
    for label, path in sources:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not remaining:
            break
        fragments.append(f"\n\n--- {label} ---\n{text[:remaining]}")
        remaining -= len(text)
    contract = """Required shape (all keys are top-level):
{"schema":"hermes/workflow@2","id":"default","skill":"SKILL_NAME","entry":"first_step","max_transitions":100,
 "input_schema":{"type":"object","properties":{"request":{"type":"string"}}},
 "steps":{"first_step":{"kind":"task","instructions":"...","input":{"request":"$input.request"},
 "effect":"read","max_retries":1,"recovery":{"method":"retry"},
 "output_schema":{"type":"object","properties":{}},"next":"END"}},"output":{}}
Every step MUST have kind task, decision, or loop. Tasks use next. EVERY decision AND loop must have output_schema {"type":"object","properties":{"transition":{"type":"string","enum":[EXACT BRANCH LABELS]}},"required":["transition"]}; the enum must contain all and only the keys of branches. Other output fields may be added to properties and required. Loops also use branches, repeat, max_iterations, and on_exhausted. A loop's repeat value is the LABEL of its repeating branch, never a step id; for example branches {"again":"collect","done":"next"} uses repeat:"again".
The runtime supplies operation_id beside input for every step and reconciliation request. Never bind $input.operation_id or declare it in input_schema.
If SKILL.md contains a Workflow contract, preserve its entry, exact step ids, kinds, edges, max_iterations, declared max_retries, on_error, terminal_outcome, and goal_gate. Never merge two declared steps. Choose 1-5 extra recovery attempts for EVERY step; use more when failures are plausibly transient. Declare effect read, write, or external_action and recovery.method retry for reads or reconcile for writes/external actions. A reconcile recovery needs instructions to check the stable operation_id supplied in the step request (run ID, step ID, visit number) and report done, not_done, or unknown. Never blindly repeat a side effect; unknown blocks. on_error routes exhausted execution failures to a recovery step, while loop on_exhausted handles the repeat branch.
Automatic chat execution supplies exactly one top-level input: request. The input_schema MUST require only request. Extract recipient, subject, location, and every other task field in the first workflow step from request; do not add them as required top-level inputs.
Never hardcode a quantity, geography, recipient, filename, or other value the request may override. Extract requested values such as target_count in the first step, mark them required in that step's output, bind them into every step that uses them, and use the skill's value only as a documented default when the request omitted it.
Every step that reads external data or performs an action MUST declare the smallest required toolsets. Use web for web_search/web_extract, terminal for command execution, and file for file operations. A step with no toolsets can only reason over its structured input; never instruct a tool-less step to research, read, write, validate, execute, send, or inspect external state.
Make output schemas checkable: require fields consumed downstream, give meaningful text minLength:1 and required collections minItems:1 where an empty result is invalid. Leave legitimately empty fields (for example recipient when email is not requested, or an honest empty shortlist) empty-compatible. For action nodes, return observed evidence rather than a bare success claim.
Branch terminal steps may return different output shapes. Prefer a common final result shape, but never make top-level output depend exclusively on a step that runs on only one branch.
Every $steps.X.output.field binding must refer to a field listed in X.output_schema.required, and X must run on every path before the consuming step. Every $input.field binding must name a required field in input_schema. A loop node should normally perform the repeated action itself, carry the needed toolsets, require every downstream output field, and point its repeat branch back to itself. On repeated iterations the runner adds previous_output to the loop input; use it to preserve and accumulate prior results. Never bind $input.previous_output: it is absent on the first visit. Do not add a $steps.<same-loop>.output binding because that value does not exist on the first iteration. Do not create a tool-less loop controller followed by a separate action step: that loses the action output across iterations. A citation or document step needs the skills and terminal/file toolsets it actually uses. For a side-effect step, add preconditions mapping its declared input fields to LITERAL required scalar values, such as sent_precheck: clear and validation_passed: true. Never put a $steps binding in preconditions; bind the field in the step input instead. The runner checks literal preconditions before calling the step. An empty top-level output returns the actual terminal step output, which is preferred when branches have different terminal results.
Set max_transitions high enough for every declared step plus all bounded loop repetitions; use 100 unless the graph provably needs a different safe bound.
If the skill declares a measurable target, preserve its goal_gate: bind target and actual, use operator gte or exists, and route a shortfall through its partial terminal before delivery. Give partial and blocked terminals a terminal_reason with a fixed code and an evidence list naming ONLY fields in that terminal step's input. Partial code must be goal_unmet; blocked code must be blocked_condition, external_unreadable, verification_unreadable, or recovery_exhausted. Use an unreadable code only when evidence actually equals unreadable. The engine renders this reason; do not ask the model to invent a reason. Never claim the user supplied facts unless they appear in the locked scope. A partial terminal must support accept, continue, and stop; a repairable blocked terminal declares resume_step. Do not fabricate missing conditions.
Reads that check whether an external action is safe or already happened must return a typed state such as clear, duplicate, or unreadable in their output schema. Route unreadable to a blocked terminal with the observed state and the concrete repair action; do not guess from ambiguous prose.
"""
    prompt = (
        "Compile this Hermes skill into a real executable workflow. Return ONLY a JSON object matching "
        "hermes/workflow@2. Model the actual flowchart: use decisions for every meaningful if/else, loops "
        "for every retry cycle, and explicit exhausted paths. Include concrete agent steps, JSON input/output "
        "schemas, and only safe $input.* or $steps.<id>.output.* bindings. Never emit credentials. Do not "
        "invent a fallback step; represent the actual procedure.\n\n" + contract + "\nSKILL PACKAGE:" + "".join(fragments))
    messages = [
        {"role": "system", "content": "You compile safe, deterministic workflow graphs."},
        {"role": "user", "content": prompt},
    ]
    last_error, preview = "", ""
    for attempt in range(3):
        if progress:
            progress(f"Attempt {attempt + 1}/3: waiting for model response")
        done = threading.Event()
        ticker = None
        if progress:
            def report_wait() -> None:
                elapsed = 0
                while not done.wait(30):
                    elapsed += 30
                    progress(f"Attempt {attempt + 1}/3: still waiting for model ({elapsed}s)")

            ticker = threading.Thread(target=report_wait, daemon=True)
            ticker.start()
        try:
            response = call_llm(task="workflow_generation", messages=messages,
                                extra_body={"response_format": _workflow_response_format(skill, contract_data)})
        finally:
            done.set()
            if ticker:
                ticker.join()
        if progress:
            progress(f"Attempt {attempt + 1}/3: validating draft")
        raw = _content(response)
        try:
            workflow = _json_object(raw)
            _normalize_generated_contract_fields(workflow, contract_data)
            validate_workflow(workflow, skill=skill, reliable=True)
            validate_workflow_contract(workflow, skill_dir, require=True)
            current_fingerprint = workflow_source_fingerprint(skill)
            if expected_fingerprint is not None and current_fingerprint != expected_fingerprint:
                raise WorkflowError("Skill source changed while its workflow was being generated.")
            if expected_job_id is not None:
                from tools.skill_usage import load_usage
                current_job = load_usage().get(skill, {}).get("workflow_generation", {}).get("job_id")
                if current_job != expected_job_id:
                    raise WorkflowError("Workflow generation job was superseded before publication.")
            target = save_overlay(skill, workflow, name=name, activation=activation)
            if progress:
                progress(f"Workflow saved: {target}")
            return {"success": True, "path": str(target), "workflow": workflow,
                    "fingerprint": current_fingerprint, "activation": activation}
        except (WorkflowError, json.JSONDecodeError) as exc:
            last_error, preview = str(exc), raw[:500].replace("\n", " ")
            if progress:
                progress(f"Attempt {attempt + 1}/3 rejected: {last_error}")
            if raw:
                messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Your workflow was rejected: {last_error}. Repair the JSON draft above and return a "
                    "complete replacement JSON object only. A step reached from multiple paths may bind only "
                    "outputs produced on every incoming path; use distinct contract nodes when the paths need "
                    "different evidence. Preserve every declared node and edge.")})
    raise WorkflowError(f"{last_error}. Model response preview: {preview!r}")


def _bulk_generate(*, force: bool, dry_run: bool, continue_on_error: bool,
                   include_local: bool = False, include_prebuilt: bool = False) -> tuple[list[dict], bool]:
    """Generate workflow gaps for Hermes-created skills, unless explicitly widened."""
    from tools.skills_tool import _find_all_skills
    from tools.skill_usage import workflow_eligibility

    results, failed = [], False
    for row in _find_all_skills():
        skill = row["name"]
        eligibility = workflow_eligibility(skill)
        selected = (
            eligibility == "eligible"
            or (eligibility == "user_owned" and include_local)
            or (eligibility in {"prebuilt", "external"} and include_prebuilt)
        )
        if not selected:
            results.append({"skill": skill, "status": f"skipped_{eligibility}"})
            continue
        try:
            workflow, source = load_workflow(skill)
            if not force:
                results.append({"skill": skill, "status": "skipped", "reason": f"valid {source.source} workflow"})
                continue
        except WorkflowError:
            pass
        if dry_run:
            results.append({"skill": skill, "status": "would_generate"})
            continue
        try:
            generated = _generate(skill, "default", progress=lambda message: print(
                f"{skill}: {message}", file=sys.stderr, flush=True))
            results.append({"skill": skill, "status": "generated", "path": generated["path"]})
        except Exception as exc:
            failed = True
            results.append({"skill": skill, "status": "failed", "error": str(exc)})
            print(f"Workflow generation failed for {skill}: {exc}", file=sys.stderr, flush=True)
            if not continue_on_error:
                break
    return results, failed


def _summary(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def _list_all_workflows() -> list[dict]:
    """Existing sidecars only; this inspection never generates a workflow."""
    from tools.skills_tool import _find_all_skills

    rows = []
    for skill in _find_all_skills():
        for item in workflow_sources(skill["name"]):
            rows.append({
                "skill": skill["name"], "name": item.path.stem, "source": item.source,
                "stale": item.stale, "activation": item.activation, "path": str(item.path),
            })
    return rows


def _run(skill: str, name: str, inputs: Mapping[str, Any], model_specs: list[str], reasoning_specs: list[str]) -> dict:
    model_overrides, reasoning_overrides = _overrides(model_specs), _overrides(reasoning_specs)
    from agent.workflow_runner import run_workflow
    return run_workflow(skill, name, inputs, model_overrides=model_overrides,
                        reasoning_overrides=reasoning_overrides)


def cmd_workflow(args) -> int:
    action = getattr(args, "workflow_action", None) or "list"
    try:
        if action == "list":
            if args.all:
                if args.skill:
                    raise WorkflowError("Use either a skill name or --all, not both.")
                rows = _list_all_workflows()
                print(json.dumps(rows, indent=2) if rows else "No workflows found in this profile.")
                return 0
            if not args.skill:
                print("Pass a skill name or use `hermes workflow list --all`.")
                return 2
            rows = [{"name": item.path.stem, "source": item.source, "stale": item.stale,
                     "activation": item.activation, "path": str(item.path)}
                    for item in workflow_sources(args.skill)]
            print(json.dumps(rows, indent=2) if rows else f"No workflows found for {args.skill}.")
            return 0
        if action == "generate":
            if args.all:
                results, failed = _bulk_generate(
                    force=args.force, dry_run=args.dry_run,
                    continue_on_error=args.continue_on_error,
                    include_local=args.include_local, include_prebuilt=args.include_prebuilt)
                no_candidates = not any(not result["status"].startswith("skipped_") for result in results)
                message = "No Hermes-created skills need workflow generation." if no_candidates else None
                if args.json:
                    payload = {"results": results, "summary": _summary(results)}
                    if message:
                        payload["message"] = message
                    print(json.dumps(payload, indent=2, ensure_ascii=False))
                else:
                    for result in results:
                        detail = result.get("path") or result.get("reason") or result.get("error") or ""
                        print(f"{result['skill']}: {result['status']} {detail}".rstrip())
                    summary = _summary(results)
                    print("Summary: " + ", ".join(f"{status}={count}" for status, count in sorted(summary.items())))
                    if message:
                        print(message)
                return 1 if failed else 0
            if not args.skill:
                raise WorkflowError("Pass a skill name or --all.")
            result = _generate(args.skill, args.name, progress=lambda message: print(
                f"{args.skill}: {message}", file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if action == "activate":
            print(json.dumps({"success": True, "path": str(activate_workflow(args.skill, args.name))}, indent=2))
            return 0
        if action == "runs":
            print(json.dumps(list_workflow_runs(args.skill, limit=args.limit), indent=2, ensure_ascii=False))
            return 0
        if action == "inspect":
            print(json.dumps(inspect_workflow_run(args.run_id), indent=2, ensure_ascii=False))
            return 0
        if action == "resume":
            from agent.workflow_runner import resume_workflow
            result = resume_workflow(args.run_id, args.decision)
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else
                  f"Workflow {result['run_id']}: {result['status']} ({result['outcome']})")
            return 0
        if not args.skill:
            print("Pass a skill name. Use `hermes workflow list <skill>` to inspect available workflows.")
            return 2
        workflow, source = load_workflow(args.skill, args.name)
        if action == "validate":
            print(f"Valid workflow: {workflow['skill']}/{workflow['id']} ({source.source})")
            return 0
        if action == "show":
            if args.format == "yaml":
                import yaml
                print(yaml.safe_dump(workflow, allow_unicode=True, sort_keys=False))
            elif args.format == "json":
                print(json.dumps(workflow, indent=2, ensure_ascii=False))
            else:
                print(render_mermaid(workflow))
            return 0
        inputs = json.loads(args.input_json)
        if not isinstance(inputs, dict):
            raise WorkflowError("--input-json must be an object.")
        result = _run(args.skill, args.name, inputs, args.step_model, args.step_reasoning)
        print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else json.dumps(result["output"], indent=2, ensure_ascii=False))
        return 0
    except (WorkflowError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Workflow error: {exc}")
        return 2
