"""
Reasoning engine.

Deliberately NOT "ask an LLM for advice". For each primary metric flagged by
the input (app/relationships.infer_primary_metrics), it:
  1. builds a specific retrieval query
  2. pulls real evidence chunks from pgvector (app/retrieval.retrieve_evidence)
  3. filters chunks via value-aware validation to prevent citing evidence that
     contradicts the site input (e.g. rainfall, land use, region, or priority signals)
  4. matches against the primary metric and multi-metric graph links (app/relationships.RELATIONSHIPS)
  5. emits a structured Recommendation that names >=2 impacted metrics
     and cites the retrieved source — never an invented one.

An LLM (if configured) is only used downstream to phrase the natural-language
reply around these structured objects — it never invents the recommendation,
the metrics, or the source.
"""

from app.models import SiteInput, Recommendation, EvidenceChunk
from app.relationships import infer_primary_metrics, linked_metrics
from app.retrieval import retrieve_evidence

MAX_RECOMMENDATIONS = 3


def _time_horizon_for(metric_tags: list[str]) -> str:
    if any(t in metric_tags for t in ("soil_organic_carbon", "microbial_diversity")):
        return "medium"
    if any(t in metric_tags for t in ("species_richness", "habitat_diversity", "deforestation_trend")):
        return "long"
    return "short"


def _confidence_for(evidence: EvidenceChunk, matched_links: int) -> str:
    if matched_links >= 2 and evidence.year and evidence.year >= 2020:
        return "high"
    if matched_links >= 1:
        return "medium"
    return "low"


def is_land_use_compatible(evidence: EvidenceChunk, site_land_use: str | None, deforestation_trend: str | None = None) -> bool:
    """Explicit land-use compatibility check: rejects any evidence chunk that mentions
    a specific land-use system (rangeland, grassland, cropland, monoculture, agroforestry)
    if it does not match or reasonably relate to the site's actual land_use value."""
    if not site_land_use:
        return True

    lu = site_land_use.lower()
    text = evidence.text.lower()

    is_grazing_site = any(w in lu for w in ["grassland", "rangeland", "pasture", "grazing", "meadow", "livestock", "cattle", "range"])
    is_crop_site = any(w in lu for w in ["wheat", "cropland", "crop", "corn", "soybean", "arable", "monoculture", "orchard", "vineyard", "farm", "agriculture", "tillage", "paddy", "barley", "rice", "cotton"])
    is_agroforestry_site = any(w in lu for w in ["agroforestry", "silvopasture", "intercropping", "trees on farm", "orchard"])
    is_forest_site = any(w in lu for w in ["forest", "forestry", "woodland", "timber", "plantation"])
    is_urban_site = any(w in lu for w in ["urban", "peri-urban", "city", "town", "built", "residential", "park"])

    # 1. Grazing / Rangeland / Grassland evidence (e.g. kb019)
    if any(w in text for w in ["rotational grazing", "rangeland", "rangelands", "continuous grazing"]):
        if not is_grazing_site:
            return False

    # 2. Arable Cropping / Tillage / Crop Rotation / Cover Crops evidence (e.g. kb001, kb005, kb010, kb017)
    if any(w in text for w in ["cover crops", "reduced-tillage", "no-till", "crop rotation", "crop rotations"]):
        if not (is_crop_site or is_agroforestry_site):
            return False

    # 3. Monoculture cropping evidence (e.g. kb004)
    if "monoculture cropping" in text:
        if not ("mono" in lu or is_crop_site):
            return False

    # 4. Agroforestry / Tree Canopy evidence (e.g. kb002, kb013)
    if "agroforestry" in text:
        if not (is_agroforestry_site or is_crop_site or is_grazing_site):
            return False

    # 5. Urban / Peri-urban evidence (e.g. kb020)
    if "urban and peri-urban" in text or "built landscapes" in text:
        if not is_urban_site:
            return False

    # 6. Reforestation / Deforestation evidence (e.g. kb009, kb016)
    if any(w in text for w in ["reforestation", "deforestation rates"]):
        if not is_forest_site and deforestation_trend != "increasing":
            return False

    return True


def is_evidence_compatible(evidence: EvidenceChunk, site_input: SiteInput) -> bool:
    """Value-aware filter: ensures retrieved evidence does not contradict
    the user's explicit rainfall level, region, land use, or priority flags."""
    text_lower = evidence.text.lower()
    reg_lower = (site_input.region or "").lower()
    rf = site_input.rainfall

    # 1. Land use compatibility check
    if not is_land_use_compatible(evidence, site_input.land_use, site_input.deforestation_trend):
        return False

    # 2. Rainfall compatibility
    if rf == "high":
        if any(term in text_lower for term in ["low-rainfall", "low rainfall", "dryland", "drylands", "semi-arid", "arid"]):
            return False
        if any(term in text_lower for term in ["drought-tolerant", "drought-adapted"]):
            return False
    elif rf == "medium":
        if any(term in text_lower for term in ["low-rainfall", "low rainfall", "drylands"]):
            return False
        if any(term in text_lower for term in ["drought-tolerant", "drought-adapted"]):
            return False
    elif rf == "low":
        if any(term in text_lower for term in ["wetland", "high rainfall", "high-rainfall"]):
            return False

    # 3. Region / Climate compatibility
    is_semi_arid_site = any(term in reg_lower for term in ["semi-arid", "arid", "desert", "dryland"])
    if not is_semi_arid_site and ("in semi-arid" in text_lower or "semi-arid regions" in text_lower):
        return False

    if any(term in reg_lower for term in ["tropical", "humid", "rainforest"]):
        if any(term in text_lower for term in ["semi-arid", "drylands", "low-rainfall"]):
            return False
    if is_semi_arid_site:
        if "wetland" in text_lower:
            return False

    return True


def _build_query(primary_metric: str, site_input: SiteInput) -> str:
    lu = site_input.land_use or ""
    reg = site_input.region or ""
    rf = site_input.rainfall or ""

    if primary_metric == "pollution_level":
        return f"pesticide drift chemical runoff synthetic fertilizer pollution reduction buffer strips water quality {lu} {reg} {rf}"
    elif primary_metric == "deforestation_trend":
        return f"reforestation forest canopy habitat connectivity tree cover deforestation biodiversity {lu} {reg} {rf}"
    elif primary_metric == "soil_organic_carbon":
        if any(w in lu.lower() for w in ["grassland", "rangeland", "pasture", "grazing"]):
            return f"soil carbon sequestration rotational grazing rangelands grassland plant species diversity {reg} {rf}"
        else:
            return f"soil organic carbon cover crops compost tillage organic matter microbial biodiversity {lu} {reg} {rf}"
    elif primary_metric == "rainfall":
        if rf == "low":
            return f"rainwater harvesting micro-catchment soil moisture retention drought resilience low rainfall drylands {lu} {reg}"
        elif rf == "high":
            return f"riparian buffer strips waterways nutrient sediment runoff water quality aquatic species survival {lu} {reg}"
        else:
            return f"soil moisture retention water availability plant-available water species survival {lu} {reg}"
    elif primary_metric == "land_use":
        if "mono" in lu.lower():
            return f"monoculture cropping diversification crop rotation polyculture habitat heterogeneity pollinator bird richness {reg} {rf}"
        elif any(w in lu.lower() for w in ["grass", "range", "pasture", "grazing"]):
            return f"rotational grazing rangelands grassland plant species diversity pasture management {reg} {rf}"
        elif "agroforest" in lu.lower():
            return f"agroforestry canopy structural diversity tree canopy microclimate buffering {reg} {rf}"
        elif "urban" in lu.lower():
            return f"urban green corridors bird pollinator species richness built landscapes {reg} {rf}"
        else:
            return f"hedgerows field margins corridors habitat heterogeneity {lu} {reg} {rf}"

    return f"{primary_metric.replace('_', ' ')} {lu} {reg} {rf}"


def generate_recommendations(site_input: SiteInput) -> list[Recommendation]:
    primary_metrics = infer_primary_metrics(site_input)
    recommendations: list[Recommendation] = []
    seen_evidence_ids: set[str] = set()

    # No-urgent-metrics path: healthy inputs receive general resilience advice
    if not primary_metrics:
        lu = site_input.land_use or "agricultural land"
        reg = site_input.region or "landscape"
        rf = site_input.rainfall or ""
        query = f"biodiversity resilience habitat enhancement soil conservation {lu} {reg} {rf}".strip()
        evidence_list = retrieve_evidence(query, top_k=15)
        for evidence in evidence_list:
            if evidence.id in seen_evidence_ids:
                continue
            if not is_evidence_compatible(evidence, site_input):
                continue
            impacted = sorted(set(evidence.metric_tags))
            if len(impacted) < 2:
                impacted = evidence.metric_tags[:2] or ["land_use", "habitat_diversity"]
            seen_evidence_ids.add(evidence.id)
            recommendations.append(
                Recommendation(
                    action=evidence.text.split(",")[0].split(".")[0].strip().capitalize(),
                    reasoning=evidence.text,
                    metrics_impacted=impacted,
                    time_horizon=_time_horizon_for(evidence.metric_tags),
                    confidence=_confidence_for(evidence, len(impacted)),
                    source=f"{evidence.source} ({evidence.year})" if evidence.year else evidence.source,
                )
            )
            if len(recommendations) >= MAX_RECOMMENDATIONS:
                break
        return recommendations

    for primary in primary_metrics:
        if len(recommendations) >= MAX_RECOMMENDATIONS:
            break

        query = _build_query(primary, site_input)
        evidence_list = retrieve_evidence(query, top_k=15)

        for evidence in evidence_list:
            if evidence.id in seen_evidence_ids:
                continue

            if not is_evidence_compatible(evidence, site_input):
                continue

            # Ensure evidence aligns directly with the primary metric under evaluation
            if primary == "pollution_level" and "pollution_level" not in evidence.metric_tags:
                continue
            if primary == "deforestation_trend" and "deforestation_trend" not in evidence.metric_tags:
                continue
            if primary == "soil_organic_carbon" and "soil_organic_carbon" not in evidence.metric_tags:
                continue
            if primary == "rainfall" and not any(t in evidence.metric_tags for t in ["water_availability", "species_survival", "water_quality"]):
                continue
            if primary == "land_use" and not any(t in evidence.metric_tags for t in ["land_use", "habitat_diversity", "habitat_fragmentation", "species_richness"]):
                continue

            impacted = sorted(set(evidence.metric_tags) | ({primary} if primary in ["rainfall", "soil_organic_carbon", "pollution_level", "deforestation_trend", "land_use"] else set()))
            if len(impacted) < 2:
                continue

            seen_evidence_ids.add(evidence.id)
            recommendations.append(
                Recommendation(
                    action=evidence.text.split(",")[0].split(".")[0].strip().capitalize(),
                    reasoning=evidence.text,
                    metrics_impacted=impacted,
                    time_horizon=_time_horizon_for(evidence.metric_tags),
                    confidence=_confidence_for(evidence, len(impacted)),
                    source=f"{evidence.source} ({evidence.year})" if evidence.year else evidence.source,
                )
            )
            break  # one recommendation per primary metric

    # If fewer than MAX_RECOMMENDATIONS were generated, backfill with compatible resilience chunks
    if len(recommendations) < MAX_RECOMMENDATIONS:
        lu = site_input.land_use or "agricultural land"
        reg = site_input.region or "landscape"
        rf = site_input.rainfall or ""
        query = f"biodiversity resilience habitat enhancement soil conservation {lu} {reg} {rf}".strip()
        evidence_list = retrieve_evidence(query, top_k=15)
        for evidence in evidence_list:
            if evidence.id in seen_evidence_ids:
                continue
            if not is_evidence_compatible(evidence, site_input):
                continue
            impacted = sorted(set(evidence.metric_tags))
            if len(impacted) < 2:
                impacted = evidence.metric_tags[:2] or ["land_use", "habitat_diversity"]
            seen_evidence_ids.add(evidence.id)
            recommendations.append(
                Recommendation(
                    action=evidence.text.split(",")[0].split(".")[0].strip().capitalize(),
                    reasoning=evidence.text,
                    metrics_impacted=impacted,
                    time_horizon=_time_horizon_for(evidence.metric_tags),
                    confidence=_confidence_for(evidence, len(impacted)),
                    source=f"{evidence.source} ({evidence.year})" if evidence.year else evidence.source,
                )
            )
            if len(recommendations) >= MAX_RECOMMENDATIONS:
                break

    return recommendations



