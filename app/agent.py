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
    site = state["site_input"]
    llm = get_llm()
    has_urgent = bool(infer_primary_metrics(site))

    if llm is not None:
        try:
            state["reply"] = _llm_phrase(recs, llm, site, has_urgent=has_urgent)
            return state
        except Exception:
            pass

    known = []
    if site.soil_organic_carbon_pct is not None:
        known.append(f"Soil Organic Carbon: {site.soil_organic_carbon_pct}%")
    if site.rainfall:
        known.append(f"Rainfall: {site.rainfall}")
    if site.land_use:
        known.append(f"Land Use: {site.land_use}")
    if site.region:
        known.append(f"Region: {site.region}")
    if site.soil_ph is not None:
        known.append(f"Soil pH: {site.soil_ph}")
    if site.soil_moisture:
        known.append(f"Soil Moisture: {site.soil_moisture}")
    if site.pollution_level:
        known.append(f"Pollution Level: {site.pollution_level}")
    if site.deforestation_trend:
        known.append(f"Deforestation Trend: {site.deforestation_trend}")

    summary_text = (
        f"Site profile evaluated: {', '.join(known) if known else 'general ecosystem'}. "
        f"{'Urgent environmental pressure(s) identified requiring targeted ecological intervention.' if has_urgent else 'Metrics indicate healthy base conditions; general resilience recommendations apply.'}"
    )

    lines = [f"**Site assessment:**\n{summary_text}\n"]

    for i, rec in enumerate(recs, start=1):
        lines.append(f"**{i}. {rec.action}**")
        lines.append(f"**What to do:** Implement {rec.action.lower()}. {rec.reasoning.split('.')[0]}.")
        lines.append(f"**Why it fits this site:** Designed for {site.land_use or 'agricultural/natural land'} in a {site.region or 'stated'} region under {site.rainfall or 'observed'} rainfall conditions.")
        lines.append(f"**Ecological mechanism:** {rec.reasoning}")
        lines.append(f"**Metrics impacted:** {', '.join(rec.metrics_impacted)}")
        lines.append(f"**Time horizon:** {rec.time_horizon.capitalize()} | **Confidence:** {rec.confidence.capitalize()}")
        lines.append(f"**Evidence:** {rec.source}")
        lines.append(f"**Limitation:** Site-specific soil testing and multi-year monitoring recommended to confirm long-term outcomes.\n")

    state["reply"] = "\n".join(lines)
    return state


SYSTEM_PROMPT = """
You are Darukaa.Earth's AI Biodiversity Intelligence system. Behave like an evidence-driven environmental scientist, not a generic chatbot.

1. Use the actual site data:
Extract and preserve all provided parameters (soil pH, organic carbon, moisture, rainfall/climate/region, land use, biodiversity indicators, pollution and its source, deforestation and habitat fragmentation). Do not invent or silently change values. Distinguish Known (explicitly provided), Unknown (not provided), and Inferred (scientifically derived). Never present an inference as a user-provided fact.

2. Check applicability:
Every recommendation must fit the user's specific ecosystem, climate and land use. Reject recommendations that are generally valid but unsuitable for the stated site.

3. Multi-metric reasoning:
Do not give single-variable recommendations. Build a causal chain: Site conditions -> ecological mechanism -> biodiversity effect -> intervention -> measurable metric. Connect at least 2-3 relevant environmental variables where scientifically appropriate.

4. Handle missing information:
Ask a clarifying question when missing information could materially change the recommendation. If the available information is insufficient for a site-specific recommendation, say what is missing instead of guessing.

5. Evidence grounding:
Retrieve evidence before recommending an intervention. Prefer peer-reviewed studies, systematic reviews, IPCC, IPBES, FAO and other authoritative sources. The source must support the specific claim, not merely the general topic. Never invent sources, statistics, percentages, or timeframes. Only provide quantitative estimates when the retrieved evidence supports them.

6. Select relevant recommendations:
Give 2-4 strong recommendations, prioritizing interventions that directly address the site's major pressures, match the land use, involve multiple environmental variables, have credible evidence, and have measurable outcomes.

7. Output format:
**Site assessment:**
Briefly summarize the main ecological conditions and interacting pressures.

For each recommendation:

**1. [Specific intervention]**
**What to do:** Concrete action.
**Why it fits this site:** Connect the user's actual parameters.
**Ecological mechanism:** Explain the causal chain.
**Metrics impacted:** List measurable metrics.
**Time horizon:** Short / Medium / Long
**Confidence:** High / Medium / Low
**Evidence:** Source supporting the recommendation.
**Limitation:** Important uncertainty or missing information.
"""


def _llm_phrase(recommendations: list[Recommendation], llm, site_input: SiteInput, has_urgent: bool = True) -> str:
    summary = "\n".join(
        f"- {r.action} (impacts: {', '.join(r.metrics_impacted)}; "
        f"horizon: {r.time_horizon}; confidence: {r.confidence}; source: {r.source})\n"
        f"  reasoning: {r.reasoning}"
        for r in recommendations
    )
    known = [f"{k}: {v}" for k, v in site_input.model_dump().items() if v is not None and k not in ("intro_shown", "assessment_completed")]
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Known site parameters: {', '.join(known)}\n\n"
        f"Retrieved structured recommendations to rephrase:\n"
        f"{summary}\n\n"
        f"Format your response using the exact Section 7 layout (**Site assessment:** followed by numbered recommendations with bold field names)."
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
