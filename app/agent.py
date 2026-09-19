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
from app.relationships import infer_primary_metrics, is_healthy_site


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
        inc = state["incoming_data"]
        core_fields = ["soil_organic_carbon_pct", "rainfall", "land_use", "region"]
        inc_core_count = sum(1 for f in core_fields if inc.get(f) is not None)
        # Fresh site isolation: if assessment completed or incoming data has >=3 core fields
        if site_input.assessment_completed or inc_core_count >= 3:
            merged = SiteInput()
            merged.intro_shown = site_input.intro_shown
        else:
            merged = site_input.model_copy()
        for k, v in inc.items():
            if k in SITE_FIELDS and v is not None:
                setattr(merged, k, v)
                merged.set_known(k)
                newly_filled.append(k)
        merged.validate_user_values()
        site_input = merged

    if state.get("message"):
        result = parse_message(state["message"], site_input)
        site_input = result.site_input
        newly_filled.extend(f for f in result.newly_filled if f not in newly_filled)
        understood = result.understood

    # Enforce Section 5: User-provided values must NEVER be corrupted
    site_input.validate_user_values()

    state["site_input"] = site_input
    state["newly_filled"] = newly_filled
    state["understood"] = understood
    return state


def node_check_completeness(state: AgentState) -> str:
    # 1. If newly filled site fields were provided
    if state.get("newly_filled"):
        missing = state["site_input"].missing_required()
        state["missing_fields"] = missing
        return "ask_clarifying" if missing else "reason"

    # 2. If assessment was completed and no new site parameters were parsed
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
        if state.get("newly_filled"):
            summary = humanize_known_fields(site_input, state["newly_filled"])
            state["reply"] = f"Got it — noted {summary}. {question}"
        else:
            state["reply"] = question
    elif not state.get("understood", True) and not is_greeting(state.get("message") or ""):
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


def _build_site_assessment(site: SiteInput, is_healthy: bool) -> str:
    """Build the Site Assessment section (spec §15, §20, §22).
    Shows current conditions, main pressures, and unknowns."""
    lines = ["## Site assessment\n"]

    # Current conditions — only show KNOWN values
    conditions = []
    if site.region:
        conditions.append(f"Region: {site.region.title()}")
    if site.land_use:
        conditions.append(f"Land use: {site.land_use.title()}")
    if site.soil_organic_carbon_pct is not None:
        conditions.append(f"Soil organic carbon: {site.soil_organic_carbon_pct}%")
    if site.soil_ph is not None:
        conditions.append(f"Soil pH: {site.soil_ph}")
    if site.soil_moisture:
        conditions.append(f"Soil moisture: {site.soil_moisture.title()}")
    if site.rainfall:
        conditions.append(f"Rainfall: {site.rainfall.title()}")
    if site.temperature:
        conditions.append(f"Temperature: {site.temperature.title()}")
    if site.pollution_level:
        conditions.append(f"Pollution: {site.pollution_level.title()}")
    if site.deforestation_trend:
        conditions.append(f"Deforestation trend: {site.deforestation_trend.title()}")
    if site.human_impact:
        conditions.append(f"Human impact: {site.human_impact.title()}")
    if site.biodiversity_status:
        conditions.append(f"Biodiversity: {site.biodiversity_status.title()}")

    lines.append("Current conditions:")
    for c in conditions:
        lines.append(f"- {c}")

    # Main pressures (spec §7, §9, §19, §22)
    lines.append("\nMain environmental pressures:")
    if is_healthy:
        lines.append("- None identified — baseline conditions indicate relatively healthy ecological function. No urgent pressure is evident from the supplied data; recommendations focus on general resilience and maintenance.")
    else:
        pressures = infer_primary_metrics(site)
        if pressures:
            pressure_labels = {
                "soil_organic_carbon": f"Low soil organic carbon ({site.soil_organic_carbon_pct}%)" if site.soil_organic_carbon_pct is not None else "Low soil organic carbon",
                "rainfall": f"Water stress ({site.rainfall} rainfall)" if site.rainfall else "Water stress",
                "land_use": f"Land-use pressure ({site.land_use})" if site.land_use else "Land-use pressure",
                "pollution_level": f"Pollution pressure ({site.pollution_level})" if site.pollution_level else "Pollution pressure",
                "deforestation_trend": f"Active deforestation ({site.deforestation_trend})" if site.deforestation_trend else "Active deforestation",
                "grazing_pressure": f"Grazing pressure ({site.human_impact})" if site.human_impact else "Grazing pressure",
                "pesticide_exposure": f"Pesticide exposure risk ({site.human_impact})" if site.human_impact else "Pesticide exposure risk",
                "habitat_fragmentation": "Habitat fragmentation and reduced connectivity",
                "habitat_loss": "Habitat loss and structural degradation",
                "water_quality": "Water quality degradation and agricultural runoff",
            }
            for p in pressures:
                lines.append(f"- {pressure_labels.get(p, p.replace('_', ' ').title())}")
        else:
            lines.append("- General environmental resilience and conservation monitoring")

    # Unknowns — list fields that were not provided (spec §2, §6, §22)
    unknown_fields = []
    field_display = {
        "soil_organic_carbon_pct": "Soil organic carbon",
        "rainfall": "Rainfall",
        "land_use": "Land use",
        "region": "Region",
        "soil_ph": "Soil pH",
        "soil_moisture": "Soil moisture",
        "temperature": "Temperature",
        "pollution_level": "Pollution level",
        "deforestation_trend": "Deforestation trend",
        "human_impact": "Human impact",
        "biodiversity_status": "Biodiversity status",
    }
    for f in ["soil_ph", "soil_moisture", "temperature", "pollution_level", "deforestation_trend",
              "human_impact", "biodiversity_status"]:
        if getattr(site, f, None) is None:
            unknown_fields.append(field_display.get(f, f))
    if unknown_fields:
        lines.append(f"\nUnknowns:\n- {', '.join(unknown_fields)} (not measured/provided)")

    return "\n".join(lines)


def _format_recommendation(i: int, rec: Recommendation) -> str:
    """Format a single recommendation in the spec §15/§20/§22 layout."""
    evidence_text = rec.source
    if rec.evidence_excerpt and rec.evidence_excerpt != rec.source:
        evidence_text += f'\n*Retrieved scientific evidence:* "{rec.evidence_excerpt}"'

    lines = [
        f"\n## Recommendation {i}\n",
        f"**{rec.action}**\n",
        f"### What to do\n{rec.what_to_do}\n",
        f"### Why it fits this site\n{rec.why_it_fits}\n",
        f"### Ecological mechanism\n{rec.ecological_mechanism or rec.reasoning}\n",
        f"### Impacted metrics\n{', '.join(m.replace('_', ' ').title() for m in rec.metrics_impacted)}\n",
        f"### Time horizon\n{rec.time_horizon.capitalize()}\n",
        f"### Confidence\n{rec.confidence.capitalize()}\n",
        f"### Evidence\n{evidence_text}\n",
        f"### Limitation\n{rec.limitation}",
    ]
    return "\n".join(lines)


def _validate_output(recs: list[Recommendation], site: SiteInput) -> list[Recommendation]:
    """Final validation check before returning (spec §21, §26).
    Removes any recommendation that fails integrity checks."""
    validated = []
    lu = (site.land_use or "").lower()

    for rec in recs:
        # Check: must have concrete action, site fit, mechanism, limitation, >= 2 metrics
        if not rec.what_to_do or not rec.why_it_fits or not rec.limitation:
            continue
        if len(rec.metrics_impacted) < 2:
            continue
        if "implement the recommended practice" in rec.what_to_do.lower():
            continue

        reasoning_lower = f"{rec.action} {rec.what_to_do} {rec.reasoning}".lower()

        # Check: no orchard getting annual crop rotation or tillage
        if "orchard" in lu:
            if ("crop rotation" in reasoning_lower or "no-till" in reasoning_lower) and "orchard" not in reasoning_lower:
                continue
            if any(w in reasoning_lower for w in ["rotational grazing", "overgrazing", "pasture", "rangeland"]):
                continue

        # Check: no forest getting grazing recommendations
        if any(w in lu for w in ["forest", "woodland"]):
            if any(w in reasoning_lower for w in ["rotational grazing", "stocking", "rangeland", "pasture"]):
                continue

        # Check: no pasture getting crop-specific recommendations
        if "pasture" in lu or "grassland" in lu:
            if "crop rotation" in reasoning_lower:
                if "pasture" not in reasoning_lower and "grassland" not in reasoning_lower:
                    continue

        validated.append(rec)

    return validated


def node_format_output(state: AgentState) -> AgentState:
    recs = state["recommendations"]
    site = state["site_input"]
    llm = get_llm()
    healthy = is_healthy_site(site)

    # Ensure user-provided values remain uncorrupted (spec §5)
    site.validate_user_values()

    # Run final validation (spec §21, §26)
    recs = _validate_output(recs, site)
    state["recommendations"] = recs

    if llm is not None:
        try:
            state["reply"] = _llm_phrase(recs, llm, site, healthy=healthy)
            return state
        except Exception:
            pass

    # Rule-based output formatting (spec §15, §20)
    assessment = _build_site_assessment(site, healthy)
    parts = [assessment]

    for i, rec in enumerate(recs, start=1):
        parts.append(_format_recommendation(i, rec))

    if not recs:
        if healthy:
            parts.append(
                "\nNo specific intervention is strongly indicated by the evidence for this site's current conditions. "
                "Continue current management practices and monitor key indicators periodically."
            )
        else:
            parts.append(
                "\nInsufficient compatible evidence was found for specific recommendations. "
                "Consider providing additional site details for more targeted guidance."
            )

    state["reply"] = "\n".join(parts)
    return state


SYSTEM_PROMPT = """
You are Darukaa.Earth's AI Biodiversity Intelligence system. Behave like an evidence-driven environmental scientist, not a generic chatbot.

1. Use the actual site data:
Extract and preserve all provided parameters (soil pH, organic carbon, moisture, rainfall/climate/region, land use, biodiversity indicators, pollution and its source, deforestation and habitat fragmentation, human impact). Do not invent or silently change values. Distinguish Known (explicitly provided), Unknown (not provided), and Inferred (scientifically derived). Never present an inference as a user-provided fact. List any unknown parameters explicitly.

2. Check applicability:
Every recommendation must fit the user's specific ecosystem, climate and land use. Reject recommendations that are generally valid but unsuitable for the stated site. Orchard ≠ annual cropland. Forest ≠ pasture. Pasture ≠ cropland.

3. Multi-metric reasoning:
Do not give single-variable recommendations. Build a causal chain: Site conditions -> ecological interaction -> environmental mechanism -> biodiversity consequence -> intervention -> measurable outcomes. Connect at least 2-3 relevant environmental variables dynamically selected from the site.

4. Handle missing information:
List unknown parameters explicitly. Never invent values for parameters the user did not provide. If the available information is insufficient for a site-specific recommendation, say what is missing instead of guessing.

5. Evidence grounding:
The source must support the specific claim, not merely the general topic. Never invent sources, statistics, percentages, or timeframes. Only provide quantitative estimates when the retrieved evidence supports them.

6. Confidence scoring:
High = strong evidence + good context match + sufficient data. Medium = credible evidence but contextual uncertainty. Low = indirect/limited evidence or substantial missing information. Do NOT automatically assign High to every recommendation.

7. Healthy sites:
If no major pressure is evident, state that baseline conditions indicate relatively healthy ecological function. Do NOT manufacture urgent problems. Return only maintenance/resilience recommendations if useful (0-2 recommendations).

8. Output format:
## Site assessment
Current conditions:
- Region: ...
- Land use: ...
Main environmental pressures:
- ...
Unknowns:
- ...

For each recommendation:
## Recommendation N
**[Specific intervention title]**
### What to do
Concrete action.
### Why it fits this site
Connect the user's actual site parameters.
### Ecological mechanism
Explain the causal chain connecting conditions, interaction, mechanism, and outcomes.
### Impacted metrics
List measurable true outcomes (never input conditions).
### Time horizon
Short / Medium / Long
### Confidence
High / Medium / Low
### Evidence
Source title and excerpt.
### Limitation
Context-specific limitation.
"""


def _llm_phrase(recommendations: list[Recommendation], llm, site_input: SiteInput, healthy: bool = False) -> str:
    rec_details = []
    for r in recommendations:
        rec_details.append(
            f"- Action: {r.action}\n"
            f"  What to do: {r.what_to_do}\n"
            f"  Why it fits: {r.why_it_fits}\n"
            f"  Ecological mechanism: {r.ecological_mechanism or r.reasoning}\n"
            f"  Metrics impacted: {', '.join(r.metrics_impacted)}\n"
            f"  Time horizon: {r.time_horizon}\n"
            f"  Confidence: {r.confidence}\n"
            f"  Evidence source: {r.source}\n"
            f"  Evidence excerpt: {r.evidence_excerpt}\n"
            f"  Limitation: {r.limitation}"
        )
    summary = "\n".join(rec_details)

    # Build known/unknown lists for the LLM
    known = []
    unknown = []
    field_display = {
        "soil_organic_carbon_pct": "Soil organic carbon",
        "rainfall": "Rainfall", "land_use": "Land use", "region": "Region",
        "soil_ph": "Soil pH", "soil_moisture": "Soil moisture", "temperature": "Temperature",
        "pollution_level": "Pollution", "deforestation_trend": "Deforestation trend",
        "human_impact": "Human impact", "biodiversity_status": "Biodiversity status",
    }
    for f in SITE_FIELDS:
        v = getattr(site_input, f, None)
        label = field_display.get(f, f.replace("_", " ").title())
        if v is not None:
            known.append(f"{label}: {v}")
        elif f not in ("latitude", "longitude"):
            unknown.append(label)

    healthy_note = ""
    if healthy:
        healthy_note = (
            "\n\nIMPORTANT: This is a HEALTHY site with no urgent pressures. "
            "Do NOT claim 'Urgent environmental pressures identified'. "
            "State that conditions are relatively healthy and provide only maintenance/resilience recommendations."
        )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Known site parameters: {', '.join(known)}\n"
        f"Unknown parameters: {', '.join(unknown) if unknown else 'None'}\n"
        f"{healthy_note}\n\n"
        f"Retrieved structured recommendations to format:\n"
        f"{summary}\n\n"
        f"Format your response using the exact output format from section 8. "
        f"Do NOT invent any statistics, percentages, or sources not listed above. "
        f"Do NOT add recommendations beyond those listed above. "
        f"List unknown parameters explicitly in the site assessment."
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
