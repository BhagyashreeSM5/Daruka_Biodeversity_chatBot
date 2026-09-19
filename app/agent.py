"""
LangGraph state machine implementing the conversational-intelligence
requirement: parse -> check completeness -> (ask clarifying question OR
explain what wasn't understood) OR (retrieve evidence -> multi-metric
reasoning -> format structured output).

    START
      |
   parse_input
      |
  check_completeness
    /          \\
 [missing]   [complete]
    |             |
ask_clarifying  reason
 (or retry msg)   |
    |         format_output
   END             |
                   END
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.models import SiteInput, Recommendation
from app.parsing import (
    parse_message,
    next_clarifying_question,
    humanize_known_fields,
    clarification_retry_message,
    is_greeting,
    is_conversational_filler,
    is_user_question,
    SITE_FIELDS,
)
from app.reasoning import generate_recommendations, answer_question_with_evidence
from app.llm import get_llm
from app.relationships import infer_primary_metrics


class AgentState(TypedDict):
    session_id: str
    turn_count: int
    message: Optional[str]
    incoming_data: Optional[dict]
    site_input: SiteInput
    missing_fields: list[str]
    newly_filled: list[str]
    understood: bool
    recommendations: list[Recommendation]
    reply: str
    status: str


def node_parse_input(state: AgentState) -> AgentState:
    site_input = state["site_input"]
    newly_filled: list[str] = []
    understood = True

    if state.get("incoming_data"):
        merged = site_input.model_copy()
        for k, v in state["incoming_data"].items():
            if k in SITE_FIELDS and v is not None:
                setattr(merged, k, v)
                newly_filled.append(k)
        site_input = merged

    if state.get("message"):
        result = parse_message(state["message"], site_input)
        site_input = result.site_input
        newly_filled.extend(f for f in result.newly_filled if f not in newly_filled)
        understood = result.understood

    state["site_input"] = site_input
    state["newly_filled"] = newly_filled
    state["understood"] = understood
    return state


def node_check_completeness(state: AgentState) -> str:
    # 1. If newly filled site fields were provided (e.g. updated parameters, soil pH, moisture, edge-deforestation)
    if state.get("newly_filled"):
        missing = state["site_input"].missing_required()
        state["missing_fields"] = missing
        return "ask_clarifying" if missing else "reason"

    # 2. If assessment was completed and no new site parameters were parsed in this message:
    if state["site_input"].assessment_completed:
        msg = (state.get("message") or "").strip()
        if is_greeting(msg):
            return "handle_greeting_reset"
        elif is_user_question(msg):
            return "handle_user_question"
        else:
            return "post_assessment"

    # 3. Assessment in progress
    missing = state["site_input"].missing_required()
    state["missing_fields"] = missing
    return "ask_clarifying" if missing else "reason"


def node_handle_greeting_reset(state: AgentState) -> AgentState:
    site_input = SiteInput()
    question = next_clarifying_question(site_input, is_first_turn=True)
    state["site_input"] = site_input
    state["reply"] = question
    state["status"] = "awaiting_input"
    state["recommendations"] = []
    return state


def node_handle_user_question(state: AgentState) -> AgentState:
    msg = state.get("message") or ""
    recs = answer_question_with_evidence(msg, state["site_input"])
    state["recommendations"] = recs
    state["status"] = "complete"
    return state


def node_post_assessment(state: AgentState) -> AgentState:
    state["reply"] = (
        "You're welcome! Let me know if you'd like to explore a different scenario or share updated site details!"
    )
    state["status"] = "complete"
    state["recommendations"] = []
    return state


def node_ask_clarifying(state: AgentState) -> AgentState:
    site_input = state["site_input"]
    turn_count = state.get("turn_count", 1)
    is_first_turn = (turn_count <= 1)
    question = next_clarifying_question(site_input, is_first_turn=is_first_turn)

    if is_first_turn:
        # On turn 1, NEVER show retry message. Always respond with warm intro + first question
        if state.get("newly_filled"):
            summary = humanize_known_fields(site_input, state["newly_filled"])
            state["reply"] = f"Got it — noted {summary}. {question}"
        else:
            state["reply"] = question
    elif not state.get("understood", True) and not is_greeting(state.get("message") or ""):
        # Explain what went wrong instead of silently repeating the same question
        still_pending = site_input.missing_required()
        target = still_pending[0] if still_pending else None
        retry = clarification_retry_message(target) if target else question
        state["reply"] = retry
    elif state.get("newly_filled"):
        summary = humanize_known_fields(site_input, state["newly_filled"])
        state["reply"] = f"Got it — noted {summary}. {question}"
    else:
        state["reply"] = question

    state["status"] = "awaiting_input"
    state["recommendations"] = []
    return state


def node_reason(state: AgentState) -> AgentState:
    recs = generate_recommendations(state["site_input"])
    state["recommendations"] = recs
    state["status"] = "complete"
    state["site_input"].assessment_completed = True
    return state


def node_format_output(state: AgentState) -> AgentState:
    recs = state["recommendations"]
    llm = get_llm()
    has_urgent = bool(infer_primary_metrics(state["site_input"]))

    if llm is not None:
        try:
            state["reply"] = _llm_phrase(recs, llm, has_urgent=has_urgent)
            return state
        except Exception:
            pass

    if not has_urgent:
        lu_label = state["site_input"].land_use or "your site"
        lines = [
            f"Nothing in your inputs indicates an urgent environmental issue. "
            f"Here's a general resilience recommendation for {lu_label}:\n"
        ]
    else:
        lines = ["Here's what the evidence supports for this site:\n"]

    for i, rec in enumerate(recs, start=1):
        lines.append(f"{i}. {rec.action}")
        lines.append(f"   Why: {rec.reasoning}")
        lines.append(f"   Metrics impacted: {', '.join(rec.metrics_impacted)}")
        lines.append(f"   Time horizon: {rec.time_horizon} | Confidence: {rec.confidence}")
        lines.append(f"   Source: {rec.source}\n")
    state["reply"] = "\n".join(lines)
    return state


def _llm_phrase(recommendations: list[Recommendation], llm, has_urgent: bool = True) -> str:
    summary = "\n".join(
        f"- {r.action} (impacts: {', '.join(r.metrics_impacted)}; "
        f"horizon: {r.time_horizon}; confidence: {r.confidence}; source: {r.source})\n"
        f"  reasoning: {r.reasoning}"
        for r in recommendations
    )
    framing = (
        "The site metrics do not indicate an urgent environmental crisis; explicitly state "
        "that nothing in the inputs indicates an urgent issue and frame these as general resilience recommendations. "
        if not has_urgent else ""
    )
    prompt = (
        f"Rewrite the following structured recommendations as a clear, well-organized "
        f"reply for a land manager. {framing}Keep every fact, number, metric, time horizon, "
        f"confidence level and source exactly as given — do not invent or drop anything. "
        f"Use short numbered sections.\n\n" + summary
    )
    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("parse_input", node_parse_input)
    graph.add_node("ask_clarifying", node_ask_clarifying)
    graph.add_node("handle_greeting_reset", node_handle_greeting_reset)
    graph.add_node("handle_user_question", node_handle_user_question)
    graph.add_node("post_assessment", node_post_assessment)
    graph.add_node("reason", node_reason)
    graph.add_node("format_output", node_format_output)

    graph.set_entry_point("parse_input")
    graph.add_conditional_edges(
        "parse_input",
        node_check_completeness,
        {
            "ask_clarifying": "ask_clarifying",
            "reason": "reason",
            "handle_greeting_reset": "handle_greeting_reset",
            "handle_user_question": "handle_user_question",
            "post_assessment": "post_assessment",
        },
    )
    graph.add_edge("ask_clarifying", END)
    graph.add_edge("handle_greeting_reset", END)
    graph.add_edge("handle_user_question", "format_output")
    graph.add_edge("post_assessment", END)
    graph.add_edge("reason", "format_output")
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
