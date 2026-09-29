"""Host-side workflow route before the main agent's first model call."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def route_skill_workflow(agent, state) -> bool:
    """Return True when the graph owns this turn; otherwise leave the direct loop runnable."""
    if not isinstance(state.user_message, str) or getattr(agent, "platform", "") == "subagent":
        return False
    from agent.skill_workflow_generation import resume_pending, schedule_repair
    from agent.skill_workflow_runtime import run_workflow, select_workflow
    from agent.skill_workflow_store import refresh_source

    resume_pending(agent)
    try:
        selected = select_workflow(state.user_message, agent)
    except Exception as exc:
        logger.exception("Skill workflow selection failed")
        state.final_response = f"I could not check whether a required skill workflow applies: {exc}"
        state.failed = True
        state._turn_exit_reason = "skill_workflow_route_failed"
        return True
    if selected is None:
        return False
    directory, record, graph_status, graph = selected
    if graph_status == "ready":
        try:
            from agent.skill_commands import extract_user_instruction_from_skill_message

            request = extract_user_instruction_from_skill_message(state.user_message) or record["description"]
            state.final_response, succeeded = run_workflow(graph, request, agent, directory)
            state.failed = not succeeded
            state._turn_exit_reason = "skill_workflow_completed" if succeeded else "skill_workflow_failed"
        except Exception as exc:
            # An earlier node may already have changed the world. Do not retry
            # the same task in the direct agent loop.
            logger.exception("Skill workflow failed for %s", record.get("name"))
            state.final_response = f"Workflow for '{record.get('name')}' failed: {exc}"
            state.failed = True
            state._turn_exit_reason = "skill_workflow_failed"
        return True
    record = refresh_source(directory, record)
    if graph_status == "awaiting_approval":
        state.final_response = f"Workflow for skill '{record['name']}' awaits approval. Approve it before this task can run."
        state.failed = True
        state._turn_exit_reason = "skill_workflow_unavailable"
        return True
    elif graph_status == "rejected":
        state.final_response = f"Workflow for skill '{record['name']}' was rejected. Generate and approve a new workflow before this task can run."
        state.failed = True
        state._turn_exit_reason = "skill_workflow_unavailable"
        return True
    elif graph_status == "missing_skill":
        state.final_response = f"Skill '{record['name']}' is missing, so its workflow cannot run."
        state.failed = True
        state._turn_exit_reason = "skill_workflow_unavailable"
        return True
    else:
        exhausted = record.get("completed_failures", 0) >= 3
        if exhausted:
            notice = (f"Workflow for skill '{record['name']}' is {graph_status}; automatic repair attempts are exhausted. "
                      "I will complete this request directly using the skill. "
                      f"Run '/skills workflow generate {record['name']}' in this profile to retry generation.")
        else:
            prior = "A previous repair attempt failed; I will retry. " if record.get("status") == "failed" else ""
            notice = (
                f"Workflow for skill '{record['name']}' is {graph_status}. {prior}"
                "I will repair it in a parallel subagent and complete this request directly using the skill."
            )
    agent._emit_status(notice)
    if not exhausted:
        schedule_repair(record, agent)
    # Newest user tail only: the system prompt and prior messages remain stable.
    if state.messages and state.messages[-1].get("role") == "user":
        tail = state.messages[-1]
        tail["api_content"] = str(tail.get("api_content") or tail.get("content") or "") + (
            f"\n\n[Workflow unavailable: call skill_view(name='{record['name']}') "
            "and perform this task directly from that skill.]")
    return False
