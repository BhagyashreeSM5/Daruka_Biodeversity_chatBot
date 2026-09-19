"""
Explicit multi-metric relationship graph.

This is what stops the system from giving single-variable, generic answers.
Every recommendation generated in app/reasoning.py must touch at least two
linked metrics from this graph and say so in its `metrics_impacted` field —
that linkage is enforced in code, not left to an LLM's discretion.
"""

RELATIONSHIPS: dict[str, list[str]] = {
    "soil_organic_carbon": ["microbial_diversity", "water_availability", "pollinator_support"],
    "land_use": ["habitat_fragmentation", "species_richness", "habitat_diversity"],
    "rainfall": ["water_availability", "species_survival"],
    "deforestation_trend": ["habitat_fragmentation", "species_richness", "microclimate"],
    "pollution_level": ["pollinator_support", "microbial_diversity", "water_quality"],
    "water_availability": ["species_survival", "habitat_diversity"],
    "habitat_fragmentation": ["species_richness", "species_survival"],
}


def linked_metrics(primary_metric: str) -> list[str]:
    """Return the metrics causally linked to a primary metric, per the graph."""
    return RELATIONSHIPS.get(primary_metric, [])


def infer_primary_metrics(site_input) -> list[str]:
    """Map a SiteInput's populated fields onto graph node names, in priority
    order (weakest / most concerning signal first)."""
    primaries: list[tuple[float, str]] = []

    if site_input.soil_organic_carbon_pct is not None and site_input.soil_organic_carbon_pct < 1.5:
        # lower SOC = more urgent (< 1.5% is degraded)
        urgency = max(0.0, 1.0 - (site_input.soil_organic_carbon_pct / 1.5))
        primaries.append((urgency, "soil_organic_carbon"))

    if site_input.rainfall == "low":
        primaries.append((0.9, "rainfall"))

    if site_input.land_use and "mono" in site_input.land_use.lower():
        primaries.append((0.85, "land_use"))

    if site_input.pollution_level in ("medium", "high"):
        weight = 0.6 if site_input.pollution_level == "medium" else 0.9
        primaries.append((weight, "pollution_level"))

    if site_input.deforestation_trend == "increasing":
        primaries.append((0.95, "deforestation_trend"))

    primaries.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in primaries]
