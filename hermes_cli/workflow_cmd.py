"""CLI implementation for explicit, executable skill workflows."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from agent.workflows import (WorkflowError, execute_workflow, load_workflow, render_mermaid, save_overlay,
                             validate_workflow, workflow_source_fingerprint, workflow_sources)
from hermes_constants import get_hermes_home


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


def _validate_output(value: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    if schema.get("type") == "object" and not isinstance(value, Mapping):
        raise WorkflowError("Step output must be an object.")
    for name in schema.get("required", []) or []:
        if name not in value:
            raise WorkflowError(f"Step output is missing required field {name!r}.")
    for name, rule in (schema.get("properties", {}) or {}).items():
        if name in value and isinstance(rule, Mapping) and "enum" in rule and value[name] not in rule["enum"]:
            raise WorkflowError(f"Step output field {name!r} is outside its enum.")


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


def _generate(skill: str, name: str) -> dict:
    from agent.auxiliary_client import call_llm
    from tools.skill_manager_tool import _find_skill

    found = _find_skill(skill)
    if not found:
        raise WorkflowError(f"Skill {skill!r} was not found.")
    skill_dir = Path(found["path"])
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
{"schema":"hermes/workflow@1","id":"default","skill":"SKILL_NAME","entry":"first_step","max_transitions":12,
 "input_schema":{"type":"object","properties":{"request":{"type":"string"}}},
 "steps":{"first_step":{"instructions":"...","input":{"request":"$input.request"},
 "output_schema":{"type":"object","properties":{}},"next":"END"}},"output":{}}
`steps` MUST be a non-empty object keyed by step ids. `entry` MUST equal one of those keys.
"""
    prompt = (
        "Compile this Hermes skill into a real executable workflow. Return ONLY a JSON object matching "
        "hermes/workflow@1. Include concrete agent steps, JSON input/output schemas, and only safe $input.* "
        "or $steps.<id>.output.* bindings. Never emit credentials. Do not invent a fallback step; represent "
        "the actual procedure.\n\n" + contract + "\nSKILL PACKAGE:" + "".join(fragments))
    messages = [
        {"role": "system", "content": "You compile safe, deterministic workflow graphs."},
        {"role": "user", "content": prompt},
    ]
    last_error, preview = "", ""
    for attempt in range(2):
        response = call_llm(task="workflow_generation", messages=messages)
        raw = _content(response)
        try:
            workflow = _json_object(raw)
            # Models commonly follow JSON Schema's `$schema` convention and
            # call the workflow id `name`; accept those harmless aliases only
            # on generated output, while authored workflow files stay strict.
            if workflow.get("$schema") == "hermes/workflow@1":
                workflow["schema"] = workflow["$schema"]
            workflow.setdefault("id", workflow.get("name", name))
            workflow.setdefault("schema", "hermes/workflow@1")
            workflow["skill"] = skill
            if not workflow.get("entry") and isinstance(workflow.get("steps"), dict) and workflow["steps"]:
                workflow["entry"] = next(iter(workflow["steps"]))
            validate_workflow(workflow, skill=skill)
            target = save_overlay(skill, workflow, name=name)
            return {"success": True, "path": str(target), "workflow": workflow,
                    "fingerprint": workflow_source_fingerprint(skill)}
        except (WorkflowError, json.JSONDecodeError) as exc:
            last_error, preview = str(exc), raw[:500].replace("\n", " ")
            messages.append({
                "role": "user",
                "content": (
                    f"Your workflow was rejected: {last_error}. Return a complete replacement JSON object only. "
                    "It must include a non-empty top-level steps mapping with real agent steps.")})
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
            print(f"Generating workflow: {skill}", file=sys.stderr, flush=True)
            generated = _generate(skill, "default")
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


def _run(skill: str, name: str, inputs: Mapping[str, Any], model_specs: list[str], reasoning_specs: list[str]) -> dict:
    try:
        workflow, source = load_workflow(skill, name)
    except WorkflowError as exc:
        if "has no workflow" not in str(exc) and "is stale" not in str(exc):
            raise
        _generate(skill, name)
        workflow, source = load_workflow(skill, name)
    model_overrides, reasoning_overrides = _overrides(model_specs), _overrides(reasoning_specs)

    def run_step(step_id: str, step: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        route = dict(context["route"])
        if step_id in model_overrides:
            provider, separator, model = model_overrides[step_id].partition("/")
            if not separator or not provider or not model:
                raise WorkflowError("--step-model must be STEP=PROVIDER/MODEL.")
            route.update({"provider": provider, "model": model})
        if step_id in reasoning_overrides:
            route["reasoning_effort"] = reasoning_overrides[step_id]
        request = {
            "skill": workflow["skill"], "step": step_id, "instructions": step["instructions"],
            "input": context["input"], "output_schema": step.get("output_schema", {}),
            "toolsets": step.get("toolsets", workflow.get("defaults", {}).get("toolsets", [])),
        }
        last_error = ""
        for attempt in range(2):
            from run_agent import AIAgent
            child = AIAgent(
                provider=route.get("provider"), model=route.get("model", ""), max_iterations=20,
                enabled_toolsets=request["toolsets"], quiet_mode=True, platform="workflow",
                skip_context_files=True, skip_memory=True, skip_background_review=True,
                ephemeral_system_prompt="Execute one workflow step. Return only JSON matching output_schema.",
                reasoning_config={"effort": route["reasoning_effort"]} if route.get("reasoning_effort") else None,
            )
            try:
                text = child.chat(json.dumps(request, ensure_ascii=False))
            finally:
                activity = child.get_activity_summary() if hasattr(child, "get_activity_summary") else {}
                child.close()
            try:
                output = _json_object(text)
                _validate_output(output, step.get("output_schema", {}))
                output["__workflow_meta__"] = {
                    "provider": getattr(child, "provider", route.get("provider")),
                    "model": getattr(child, "model", route.get("model")),
                    "toolsets": request["toolsets"], "activity": activity,
                }
                return output
            except (WorkflowError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                request["correction"] = f"Your prior output was invalid: {last_error}. Return corrected JSON only."
        raise WorkflowError(f"Step {step_id!r} returned invalid structured output: {last_error}")

    receipt = get_hermes_home() / "workflow-runs" / f"{skill}-{name}.jsonl"
    result = execute_workflow(workflow, inputs, run_step, receipt_path=receipt)
    result.update({"source": source.source, "receipt": str(receipt)})
    return result


def cmd_workflow(args) -> int:
    action = getattr(args, "workflow_action", None) or "list"
    try:
        if action == "list":
            if not args.skill:
                print("Pass a skill name. Use `hermes skills list` to see installed skills.")
                return 2
            rows = [{"name": item.path.stem, "source": item.source, "stale": item.stale, "path": str(item.path)}
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
            print(json.dumps(_generate(args.skill, args.name), indent=2, ensure_ascii=False))
            return 0
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
    except (WorkflowError, json.JSONDecodeError) as exc:
        print(f"Workflow error: {exc}")
        return 2
