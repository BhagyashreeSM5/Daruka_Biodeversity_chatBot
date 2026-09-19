"""
Explicit multi-metric relationship graph.

This is what stops the system from giving single-variable, generic answers.
Every recommendation generated in app/reasoning.py must touch at least two
linked metrics from this graph and say so in its `metrics_impacted` field —
that linkage is enforced in code, not left to an LLM's discretion.
"""

RELATIONSHIPS: dict[str, list[str]] = {
    "soil_organic_carbon": ["microbial_diversity", "water_availability", "pollinator_support", "soil_structure"],
    "land_use": ["habitat_fragmentation", "species_richness", "habitat_diversity", "pollinator_support"],
    "rainfall": ["water_availability", "species_survival", "soil_moisture_retention"],
    "deforestation_trend": ["habitat_fragmentation", "species_richness", "microclimate", "habitat_connectivity"],
    "pollution_level": ["pollinator_support", "microbial_diversity", "water_quality"],
    "water_availability": ["species_survival", "habitat_diversity", "soil_moisture_retention"],
    "habitat_fragmentation": ["species_richness", "species_survival", "habitat_connectivity"],
    # Entries for human-impact-driven reasoning (spec §8, §18)
    "grazing_pressure": ["vegetation_cover", "soil_degradation", "species_richness", "soil_organic_carbon"],
    "pesticide_exposure": ["pollinator_support", "microbial_diversity", "water_quality", "species_richness"],
    "habitat_loss": ["species_richness", "habitat_fragmentation", "species_survival", "habitat_connectivity"],
    "water_quality": ["species_survival", "microbial_diversity", "habitat_diversity"],
}


def linked_metrics(primary_metric: str) -> list[str]:
    """Return the metrics causally linked to a primary metric, per the graph."""
    return RELATIONSHIPS.get(primary_metric, [])


def is_healthy_site(site_input) -> bool:
    """Multi-indicator healthy-site detection (spec §6, §7, §19).

    A site is considered healthy when no strong environmental pressure
    is evident from the supplied data. This uses MULTIPLE indicators,
    not a single hardcoded SOC threshold.

    Returns True only when no urgent intervention need is detected."""

    pressures_found = 0

    # --- Check each indicator ---

    # 1. SOC: very low SOC indicates degradation (<1.5% degraded, <2.0% mild)
    if site_input.soil_organic_carbon_pct is not None:
        if site_input.soil_organic_carbon_pct < 1.5:
            pressures_found += 2  # strong signal
        elif site_input.soil_organic_carbon_pct < 2.0:
            pressures_found += 1  # mild concern

    # 2. Pollution
    if site_input.pollution_level == "high":
        pressures_found += 2
    elif site_input.pollution_level == "medium":
        pressures_found += 1

    # 3. Deforestation
    if site_input.deforestation_trend == "increasing":
        pressures_found += 2

    # 4. Human impact — check for serious pressures
    hi = (site_input.human_impact or "").lower()
    if hi:
        high_pressure_words = ["overgrazing", "over-grazing", "heavy", "high",
                               "mining", "logging", "illegal", "slash"]
        moderate_words = ["moderate", "pesticide", "runoff", "road construction",
                          "urbanization", "drainage"]
        low_words = ["low", "minimal", "limited", "light"]
        if any(w in hi for w in high_pressure_words):
            pressures_found += 2
        elif any(w in hi for w in moderate_words):
            pressures_found += 1
        elif any(w in hi for w in low_words):
            pass  # low impact doesn't add pressure
        else:
            pressures_found += 1  # unknown impact type, mild concern

    # 5. Biodiversity status — explicitly degraded
    bio = (site_input.biodiversity_status or "").lower()
    sr = (site_input.species_richness or "").lower()
    hd = (site_input.habitat_diversity or "").lower()
    all_bio = f"{bio} {sr} {hd}".strip()
    if all_bio:
        if any(w in all_bio for w in ["low", "declining", "degraded", "fragmented", "poor"]):
            pressures_found += 1
        # High biodiversity is a positive signal — do NOT add pressure

    # 6. Monoculture land use has inherent pressure
    lu = (site_input.land_use or "").lower()
    if "mono" in lu:
        pressures_found += 1

    # 7. Low rainfall alone doesn't make a site unhealthy, but combined
    # with low moisture it's worth noting
    if site_input.rainfall == "low" and (site_input.soil_moisture or "").lower() == "low":
        pressures_found += 1

    # Healthy = no strong pressures detected
    return pressures_found == 0


def _human_impact_to_metrics(human_impact: str) -> list[tuple[float, str]]:
    """Map specific human-impact descriptions to the relevant primary metrics
    for reasoning. Returns (urgency, metric_name) tuples.

    Spec §5, §18: Do NOT map every impact to pollution. Use impact-specific interpretation."""
    hi = human_impact.lower()
    metrics: list[tuple[float, str]] = []

    if any(w in hi for w in ["overgrazing", "over-grazing", "over grazing"]):
        metrics.append((0.90, "grazing_pressure"))

    if any(w in hi for w in ["pesticide", "herbicide", "insecticide"]):
        metrics.append((0.85, "pesticide_exposure"))

    if any(w in hi for w in ["runoff", "nutrient", "fertilizer"]):
        metrics.append((0.80, "water_quality"))

    if any(w in hi for w in ["road", "construction", "urbanization", "urban"]):
        metrics.append((0.85, "habitat_fragmentation"))

    if any(w in hi for w in ["deforestation", "clearing", "logging", "forest loss"]):
        metrics.append((0.90, "deforestation_trend"))

    if any(w in hi for w in ["mining", "quarrying"]):
        metrics.append((0.85, "pollution_level"))
        metrics.append((0.80, "habitat_loss"))

    if any(w in hi for w in ["fire", "burning", "slash"]):
        metrics.append((0.85, "habitat_loss"))

    if any(w in hi for w in ["drainage", "wetland"]):
        metrics.append((0.80, "water_availability"))

    return metrics


def infer_primary_metrics(site_input) -> list[str]:
    """Map a SiteInput's populated fields onto graph node names, in priority
    order (weakest / most concerning signal first).

    Spec §8, §9: Metrics derived from actual current site conditions, including
    human_impact-specific interpretation."""

    # --- Healthy site check (spec §7, §19) ---
    if is_healthy_site(site_input):
        return []  # no urgent metrics → resilience/maintenance path

    primaries: list[tuple[float, str]] = []

    if site_input.soil_organic_carbon_pct is not None and site_input.soil_organic_carbon_pct < 1.5:
        # lower SOC = more urgent (<1.5% is degraded)
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

    # --- Human impact → specific metrics (spec §5, §8, §18) ---
    if site_input.human_impact:
        impact_metrics = _human_impact_to_metrics(site_input.human_impact)
        for urgency, metric in impact_metrics:
            if metric not in [m for _, m in primaries]:
                primaries.append((urgency, metric))

    # --- Biodiversity pressure signals ---
    bio = (site_input.biodiversity_status or "").lower()
    sr = (site_input.species_richness or "").lower()
    hd = (site_input.habitat_diversity or "").lower()
    all_bio = f"{bio} {sr} {hd}".strip()

    if all_bio and any(w in all_bio for w in ["low", "declining", "degraded", "fragmented", "poor"]):
        if "pollinator" in all_bio or "pesticide" in all_bio:
            if "pesticide_exposure" not in [m for _, m in primaries]:
                primaries.append((0.80, "pesticide_exposure"))
        elif "fragment" in all_bio or "connectivity" in all_bio:
            if "habitat_fragmentation" not in [m for _, m in primaries]:
                primaries.append((0.80, "habitat_fragmentation"))
        elif "loss" in all_bio or "diversity" in all_bio:
            if "habitat_loss" not in [m for _, m in primaries]:
                primaries.append((0.75, "habitat_loss"))

    primaries.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in primaries]
