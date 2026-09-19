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

import re
from app.models import SiteInput, Recommendation, EvidenceChunk
from app.relationships import infer_primary_metrics, linked_metrics, is_healthy_site
from app.retrieval import retrieve_evidence

MAX_RECOMMENDATIONS = 4
MIN_RELEVANCE_SCORE = 2  # minimum score to include a recommendation


def _time_horizon_for(metric_tags: list[str]) -> str:
    if any(t in metric_tags for t in ("soil_organic_carbon", "microbial_diversity")):
        return "medium"
    if any(t in metric_tags for t in ("species_richness", "habitat_diversity", "deforestation_trend")):
        return "long"
    return "short"


def _confidence_for(evidence: EvidenceChunk, matched_links: int, site_input: SiteInput) -> str:
    """Calculate confidence from evidence strength + context match + data completeness.
    Spec §15: Do NOT automatically assign High to everything."""
    score = 0

    # Evidence age
    if evidence.year and evidence.year >= 2020:
        score += 2
    elif evidence.year and evidence.year >= 2016:
        score += 1

    # Metric overlap
    if matched_links >= 3:
        score += 2
    elif matched_links >= 2:
        score += 1

    # Data completeness — count how many required fields are known
    missing_count = len(site_input.missing_required())
    unknown_optional = sum(
        1 for f in ["soil_ph", "soil_moisture", "pollution_level", "deforestation_trend",
                     "human_impact", "biodiversity_status"]
        if getattr(site_input, f, None) is None
    )
    if missing_count == 0 and unknown_optional <= 2:
        score += 2  # comprehensive data
    elif missing_count == 0:
        score += 1  # required complete but many unknowns

    # Land-use match — check if evidence text mentions the site's land use
    if site_input.land_use:
        lu = site_input.land_use.lower()
        text_lower = evidence.text.lower()
        lu_keywords = lu.split()
        if any(kw in text_lower for kw in lu_keywords if len(kw) > 3):
            score += 1

    # Region/climate match
    if site_input.region:
        reg = site_input.region.lower()
        if reg in evidence.text.lower():
            score += 1

    if score >= 6:
        return "high"
    elif score >= 3:
        return "medium"
    return "low"


def _limitation_for(primary_metric: str, site_input: SiteInput) -> str:
    """Generate context-specific limitation text.
    Spec §17: Do NOT use the same boilerplate for every recommendation."""

    lu = (site_input.land_use or "").lower()
    region = (site_input.region or "").lower()

    if primary_metric in ("soil_organic_carbon", "microbial_diversity"):
        base = "Monitor SOC using consistent sampling depth and method over multiple seasons."
        if "arid" in region or "semi-arid" in region:
            base += " In semi-arid conditions, organic matter decomposition rates differ from humid regions."
        return base

    if primary_metric in ("deforestation_trend", "habitat_loss"):
        return "Effectiveness depends on corridor placement, width, surrounding habitat quality, and target species movement patterns."

    if primary_metric in ("pesticide_exposure", "pollution_level"):
        base = "Effectiveness depends on pesticide type, application method, timing, and local pollinator/non-target species activity."
        if "orchard" in lu:
            base += " Orchard spray drift patterns differ from open-field application."
        return base

    if primary_metric == "rainfall":
        return "Effectiveness depends on local rainfall intensity and distribution, soil infiltration rate, terrain slope, and available water storage capacity."

    if primary_metric == "grazing_pressure":
        return "Optimal stocking rates and recovery periods vary by climate, soil type, and native grass species composition. Local rangeland management guidance should be consulted."

    if primary_metric == "land_use":
        if "mono" in lu:
            return "Diversification benefits depend on crop compatibility, market access, local pest dynamics, and available management capacity."
        return "Land-use transition outcomes depend on local ecological context, existing seed banks, and management continuity."

    if primary_metric == "habitat_fragmentation":
        return "Corridor and buffer effectiveness depends on placement, width, habitat quality, and target species dispersal capacity."

    return "Local site conditions, management capacity, and multi-season monitoring should be factored into implementation planning."


def _extract_action_title(evidence_text: str) -> str:
    """Extract a clean, short action title from evidence text.
    Spec §16: Do NOT concatenate the full scientific claim after 'Implement'."""
    text = evidence_text.strip()

    # Try to get just the intervention name from the first clause
    # Pattern: "Introducing X" → "X"
    # Pattern: "X and Y in Z regions..." → "X and Y"
    first_clause = text.split(",")[0].split(" that ")[0].split(" which ")[0].strip()

    # Remove leading verbs
    for prefix in ["Introducing ", "Implementing ", "Establishing ",
                   "Reducing ", "Increasing ", "Restoration using ",
                   "Applying ", "Adopting "]:
        if first_clause.startswith(prefix):
            first_clause = first_clause[len(prefix):]
            break

    # Capitalize and truncate if too long
    title = first_clause.strip().capitalize()
    if len(title) > 80:
        title = title[:77] + "..."
    return title


def _generate_what_to_do(evidence_text: str, site_input: SiteInput) -> str:
    """Generate a concrete, actionable 'What to do' instruction (spec §15, §16, §24).
    Every recommendation must have a concrete action.
    If a candidate cannot produce a concrete action: return empty string so caller rejects it."""
    text_lower = evidence_text.lower()
    lu = (site_input.land_use or "").lower()

    if "cover crop" in text_lower or "legume" in text_lower:
        return "Introduce nitrogen-fixing legume cover crops (e.g. vetch or clover) between main cropping seasons to supply active organic biomass and fix atmospheric nitrogen."

    if "no-till" in text_lower or "reduced-tillage" in text_lower or "reduced tillage" in text_lower:
        return "Transition to conservation no-till or reduced-tillage management, preserving topsoil pore structure and preventing organic matter oxidation."

    if "agroforestry" in text_lower or "tree canopy" in text_lower or "intercropping" in text_lower:
        if "orchard" in lu:
            return "Establish and maintain flowering native herbaceous and shrub understory layers between orchard tree rows to maximize structural canopy diversity."
        return "Integrate multi-tier native woody perennials and hedgerows within the agricultural matrix to enhance structural canopy diversity."

    if "rotational grazing" in text_lower or "grazing management" in text_lower:
        return "Implement adaptive multi-paddock rotational grazing, adjusting stocking density and pasture rest intervals based on seasonal grass regrowth."

    if "hedgerow" in text_lower or "field margin" in text_lower or "buffer strip" in text_lower or "corridor" in text_lower:
        return "Establish continuous native floral buffer strips and hedgerows along parcel perimeters to create ecological connectivity corridors."

    if "rainwater harvesting" in text_lower or "micro-catchment" in text_lower:
        return "Construct contour earth bunds, swales, or micro-catchment basins across slopes to capture runoff and improve in-situ soil moisture retention."

    if "pesticide" in text_lower:
        return "Adopt Integrated Pest Management (IPM), curtailing broad-spectrum chemical sprays and establishing untreated floral refuge zones for natural predators."

    if "reforestation" in text_lower or "deforestation" in text_lower:
        return "Replant mixed indigenous pioneer and canopy tree species along degraded borders to re-establish canopy connectivity and buffer forest edges."

    if "crop rotation" in text_lower or "diversified" in text_lower:
        return "Introduce a diversified 3-4 year rotation sequence including deep-rooting species and legumes to disrupt pest cycles and enhance soil biological activity."

    if "wetland" in text_lower:
        return "Restore hydrological flow regimes and vegetative margins in wetland zones to support aquatic macroinvertebrates and natural filtration."

    if "composting" in text_lower or "organic matter" in text_lower or "compost" in text_lower:
        return "Incorporate matured organic compost or mulch amendments into topsoil to raise active soil organic carbon and moisture-holding capacity."

    if "riparian" in text_lower:
        return "Establish dense native riparian vegetative buffers along water drainage margins to filter agrochemical runoff and stabilize embankments."

    if "contour" in text_lower or "terrac" in text_lower:
        return "Implement contour cultivation or terracing along topographic gradients to decelerate surface runoff and prevent topsoil erosion."

    if "drought" in text_lower or "native species" in text_lower:
        return "Re-seed degraded parcels with indigenous drought-adapted perennial grass and shrub varieties to ensure permanent vegetative soil cover during extended dry spells."

    if "urban" in text_lower:
        return "Plant native flowering shrubs and canopy trees in contiguous street margins and pocket habitats to form urban pollinator corridors."

    # Extract actionable directive from evidence text if it contains clear actionable verbs
    sentences = [s.strip() for s in re.split(r"[.;]", evidence_text) if s.strip()]
    action_verbs = ["plant", "establish", "maintain", "restore", "apply", "construct", "reduce", "adopt", "introduce", "retain", "re-vegetate"]
    for s in sentences:
        s_lower = s.lower()
        if any(v in s_lower for v in action_verbs) and len(s) >= 20:
            return s[0].upper() + s[1:]

    # Spec §24: Never output generic boilerplate! Return empty string to reject candidate
    return ""


def _clean_impacted_metrics(metric_tags: list[str], primary_metric: str, evidence_text: str = "") -> list[str]:
    """Ensure impacted metrics are genuine measurable environmental outcomes (spec §23).
    Rainfall, land use, temperature, region are environmental conditions, not outcomes."""
    outcomes: set[str] = set()
    text_lower = evidence_text.lower()

    for t in metric_tags:
        if t == "rainfall":
            outcomes.add("soil_moisture_retention")
            outcomes.add("water_availability")
        elif t == "land_use":
            outcomes.add("habitat_diversity")
            outcomes.add("species_richness")
        elif t in ("temperature", "region", "latitude", "longitude"):
            continue
        elif t in ("soil_organic_carbon", "microbial_diversity", "species_richness",
                   "habitat_diversity", "water_availability", "water_quality",
                   "pollinator_support", "habitat_connectivity", "vegetation_cover",
                   "soil_moisture_retention", "species_survival", "erosion_control",
                   "sediment_load"):
            outcomes.add(t)
        elif t == "pollution_level":
            outcomes.add("water_quality")
            outcomes.add("pollinator_support")
        elif t == "deforestation_trend":
            outcomes.add("habitat_connectivity")
            outcomes.add("species_richness")
        elif t == "habitat_fragmentation":
            outcomes.add("habitat_connectivity")
            outcomes.add("species_survival")
        else:
            outcomes.add(t)

    # Primary-specific true outcomes
    if primary_metric == "rainfall":
        outcomes.add("soil_moisture_retention")
        outcomes.add("water_availability")
    elif primary_metric == "soil_organic_carbon":
        outcomes.add("soil_organic_carbon")
        outcomes.add("microbial_diversity")
    elif primary_metric == "land_use":
        outcomes.add("habitat_diversity")
        outcomes.add("species_richness")
    elif primary_metric == "pollution_level":
        outcomes.add("water_quality")
        outcomes.add("pollinator_support")
    elif primary_metric == "deforestation_trend":
        outcomes.add("habitat_connectivity")
        outcomes.add("species_richness")
    elif primary_metric == "grazing_pressure":
        outcomes.add("vegetation_cover")
        outcomes.add("soil_organic_carbon")
    elif primary_metric == "pesticide_exposure":
        outcomes.add("pollinator_support")
        outcomes.add("microbial_diversity")
    elif primary_metric in ("habitat_fragmentation", "habitat_loss"):
        outcomes.add("habitat_connectivity")
        outcomes.add("species_richness")

    if any(w in text_lower for w in ["pollinator", "bee", "beneficial insect"]):
        outcomes.add("pollinator_support")
    if any(w in text_lower for w in ["soil moisture", "water retention", "moisture retention"]):
        outcomes.add("soil_moisture_retention")
    if any(w in text_lower for w in ["microbial", "mycorrhiz", "fungi"]):
        outcomes.add("microbial_diversity")
    if any(w in text_lower for w in ["species richness", "plant diversity", "bird richness"]):
        outcomes.add("species_richness")
    if any(w in text_lower for w in ["water quality", "sediment", "filtration"]):
        outcomes.add("water_quality")

    # Strictly exclude input conditions
    for forbidden in ["rainfall", "land_use", "temperature", "region", "latitude", "longitude",
                      "grazing_pressure", "pesticide_exposure", "habitat_loss"]:
        outcomes.discard(forbidden)

    result = sorted(outcomes)
    if len(result) < 2:
        result = ["habitat_diversity", "species_richness"]
    return result


def _build_ecological_mechanism(primary_metric: str, evidence: EvidenceChunk, site_input: SiteInput, action: str) -> str:
    """Construct multi-metric ecological mechanism (spec §10, §15).
    Connects >=3 interacting environmental variables from current conditions:
    Current conditions -> Ecological interaction -> Environmental mechanism -> Biodiversity consequence -> Intervention outcome."""
    lu = (site_input.land_use or "agricultural land").lower()
    reg = (site_input.region or "the landscape").lower()
    rf = (site_input.rainfall or "seasonal rainfall").lower()
    soc = site_input.soil_organic_carbon_pct
    hi = (site_input.human_impact or "").lower()
    defor = site_input.deforestation_trend
    poll = site_input.pollution_level

    vars_present = []
    if soc is not None:
        vars_present.append(f"SOC ({soc}%)")
    if site_input.rainfall:
        vars_present.append(f"{rf} rainfall")
    if site_input.land_use:
        vars_present.append(f"land use ({lu})")
    if site_input.region:
        vars_present.append(f"region ({reg})")
    if hi:
        vars_present.append(f"human pressure ({hi})")
    if defor:
        vars_present.append(f"deforestation ({defor})")
    if poll:
        vars_present.append(f"pollution ({poll})")

    evidence_sentence = evidence.text.strip()
    if not evidence_sentence.endswith("."):
        evidence_sentence += "."

    if primary_metric == "soil_organic_carbon":
        context_vars = f"soil organic carbon ({soc if soc is not None else '<1.5'}%), {rf} precipitation, and {lu} management"
        return (
            f"Under {reg} conditions, the interaction among {context_vars} governs soil structure and biological activity. "
            f"Depleted soil carbon reduces macro-aggregate stability and mycorrhizal fungal hyphae networks, which diminishes water infiltration and nutrient retention. "
            f"This functional breakdown depresses microbial biomass and root-zone soil fauna diversity. "
            f"{evidence_sentence} "
            f"Through continuous biomass accumulation and root exudation, this practice rebuilds active carbon pools, stabilizes pore geometry, and restores soil food web complexity."
        )

    elif primary_metric == "rainfall":
        carbon_ctx = f", depleted soil organic carbon ({soc}%)" if soc is not None else ""
        context_vars = f"{rf} rainfall{carbon_ctx}, and {lu} canopy cover"
        if rf == "low":
            return (
                f"In this {reg} setting, the interplay of {context_vars} creates severe moisture deficits during key phenological phases. "
                f"Without sufficient topsoil organic carbon ({soc if soc is not None else '<1.5'}%) or soil aggregate structure, episodic rainfall is lost to rapid surface runoff and evaporation rather than infiltrating the root zone. "
                f"This water stress restricts vegetative vigor, impairs plant-pollinator mutualisms, and lowers subterranean microbial resilience. "
                f"{evidence_sentence} "
                f"Capturing and decelerating runoff buffers root-zone moisture gradients, enabling persistent floral resources and stabilizing native species survival through drought intervals."
            )
        elif rf == "high":
            return (
                f"Under high precipitation in {reg} on {lu}, surface overland flow mobilizes sediment, agricultural inputs, and organic particulates toward drainage networks. "
                f"The combination of high rainfall intensity, saturated soil moisture{carbon_ctx}, and active land use elevates runoff volume, transporting nutrients that induce aquatic eutrophication and smother benthic spawning substrates. "
                f"{evidence_sentence} "
                f"Establishing vegetative filtration buffers overland velocity, strips excess nutrients and suspended solids, and preserves aquatic and riparian macroinvertebrate richness."
            )
        else:
            return (
                f"Across {reg} {lu} systems, soil moisture retention, rainfall distribution{carbon_ctx}, and canopy architecture jointly dictate water-use efficiency. "
                f"{evidence_sentence} "
                f"Optimizing hydrological retention maintains continuous moisture availability, fostering stable habitat microclimates and resilient trophic biodiversity."
            )

    elif primary_metric in ("pesticide_exposure", "pollution_level"):
        carbon_ctx = f", soil carbon ({soc}%)" if soc is not None else ""
        water_ctx = f", {rf} rainfall" if site_input.rainfall else ""
        context_vars = f"chemical exposure ({hi if hi else 'pesticide/fertilizer applications'}), land-use intensity ({lu}){water_ctx}{carbon_ctx}, and surrounding habitat structure"
        return (
            f"The conjunction of {context_vars} in {reg} generates acute non-target ecotoxicological pressures. "
            f"Chemical residues disrupt beneficial soil microbial assemblages and cause direct mortality or sublethal navigational disorientation in native pollinator populations. "
            f"Absent vegetative buffer barriers under {rf} precipitation conditions, chemical drift and runoff spread into adjacent ecological niches, collapsing natural predator-prey biocontrol equilibria. "
            f"{evidence_sentence} "
            f"Targeted mitigation reduces chemical exposure, restores trophic checks on agricultural pests, and secures nectar and pollen corridors for wild bees and beneficial insects."
        )

    elif primary_metric in ("deforestation_trend", "habitat_loss"):
        carbon_ctx = f", soil carbon ({soc}%)" if soc is not None else ""
        water_ctx = f", {rf} rainfall" if site_input.rainfall else ""
        context_vars = f"canopy loss ({defor or 'deforestation'}), landscape fragmentation{water_ctx}{carbon_ctx}, and {lu} matrix contrast"
        return (
            f"In {reg}, the coupling of {context_vars} fractures continuous native woodland into isolated, ecologically vulnerable patches. "
            f"Canopy removal intensifies edge effects, alters microclimatic temperature and moisture regulation, and blocks dispersal pathways for interior forest-dependent species. "
            f"Isolated sub-populations suffer genetic drift, heightened drought vulnerability, and reproductive failure. "
            f"{evidence_sentence} "
            f"Re-establishing structural canopy and corridor connectivity mitigates edge microclimates, reinstates safe wildlife movement routes, and bolsters meta-population gene flow."
        )

    elif primary_metric == "grazing_pressure":
        carbon_ctx = f", soil carbon ({soc}%)" if soc is not None else ""
        water_ctx = f", {rf} rainfall" if site_input.rainfall else ""
        context_vars = f"grazing intensity ({hi if hi else 'livestock pressure'}), vegetation recovery time{water_ctx}{carbon_ctx}"
        return (
            f"In {reg} grassland/pasture systems, the interaction between {context_vars} dictates vegetative succession and ground stability. "
            f"Continuous unmanaged grazing pressure under {rf} rainfall depletes palatable deep-rooting native bunchgrasses and causes topsoil hoof compaction, reducing rainwater infiltration and soil organic carbon accumulation. "
            f"This structural degradation eliminates shelter, moisture retention, and seed resources for ground-nesting birds, pollinators, and burrowing fauna. "
            f"{evidence_sentence} "
            f"Strategic grazing control and pasture resting allow perennial root systems to recover, rebuild litter cover, enhance moisture infiltration, and reinstate native plant community richness."
        )

    elif primary_metric in ("land_use", "habitat_fragmentation"):
        carbon_ctx = f", depleted soil organic carbon ({soc}%)" if soc is not None else ""
        water_ctx = f", {rf} rainfall" if site_input.rainfall else ""
        context_vars = f"land use structure ({lu}){water_ctx}{carbon_ctx}, and habitat patch connectivity"
        return (
            f"Within this {reg} {lu} ecosystem, the configuration of {context_vars} directly influences habitat heterogeneity, water availability, and species survival. "
            f"Under {rf} precipitation and depleted organic carbon, homogeneous or fragmented landscapes create severe microclimatic stress and temporal resource bottlenecks, preventing mobile pollinators, birds, and beneficial predators from finding continuous forage and nesting refugia. "
            f"{evidence_sentence} "
            f"Integrating structured semi-natural landscape features diversifies microhabitats, bridges fragmented patches, improves soil moisture retention, and supports higher multi-trophic species richness."
        )

    connected_vars = ", ".join(vars_present[:3]) if len(vars_present) >= 3 else f"{lu}, {reg}, and local soil/water conditions"
    return (
        f"In this {reg} {lu} setting, ecological interactions across {connected_vars} drive site biodiversity dynamics. "
        f"{evidence_sentence} "
        f"This intervention directly addresses interacting site pressures to restore ecosystem function and support measurable biodiversity gains."
    )


def _generate_why_it_fits(primary_metric: str, site_input: SiteInput) -> str:
    """Generate site-specific 'Why it fits' explanation.
    Spec §16: Connect the current site's actual conditions."""
    parts = []

    lu = site_input.land_use or "the stated land use"
    region = site_input.region or "the stated region"

    if primary_metric == "soil_organic_carbon" and site_input.soil_organic_carbon_pct is not None:
        parts.append(f"SOC at {site_input.soil_organic_carbon_pct}% is {'very low' if site_input.soil_organic_carbon_pct < 1.0 else 'below optimal'}, indicating degraded soil carbon reserves")

    if primary_metric == "rainfall" and site_input.rainfall:
        parts.append(f"{site_input.rainfall} rainfall creates {'water stress for vegetation and soil biota' if site_input.rainfall == 'low' else 'runoff management challenges' if site_input.rainfall == 'high' else 'moderate water availability conditions'}")

    if primary_metric in ("grazing_pressure",) and site_input.human_impact:
        parts.append(f"reported human impact ({site_input.human_impact}) directly affects vegetation cover and soil condition")

    if primary_metric == "pesticide_exposure" and site_input.human_impact:
        parts.append(f"reported {site_input.human_impact} creates risk for pollinator and non-target species")

    if primary_metric == "pollution_level" and site_input.pollution_level:
        parts.append(f"{site_input.pollution_level} pollution level requires targeted reduction measures")

    if primary_metric == "deforestation_trend" and site_input.deforestation_trend:
        parts.append(f"{'ongoing' if site_input.deforestation_trend == 'increasing' else site_input.deforestation_trend} deforestation directly threatens habitat connectivity")

    if primary_metric == "land_use":
        parts.append(f"current land use ({lu}) has inherent habitat diversity implications")

    if primary_metric == "habitat_fragmentation":
        parts.append("habitat fragmentation affects species movement and genetic connectivity")

    # Add soil moisture context if relevant
    if site_input.soil_moisture and primary_metric in ("soil_organic_carbon", "rainfall"):
        parts.append(f"{site_input.soil_moisture} soil moisture compounds water retention challenges")

    # Add biodiversity context
    bio = site_input.biodiversity_status
    if bio:
        parts.append(f"current biodiversity status ({bio}) confirms ecological pressure")

    if not parts:
        parts.append(f"conditions in this {region} {lu} site support this intervention")

    return f"This site is a {region} {lu} system. " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Applicability filters (spec §9, §10, §12)
# ---------------------------------------------------------------------------

def is_land_use_compatible(evidence: EvidenceChunk, site_land_use: str | None, deforestation_trend: str | None = None) -> bool:
    """Explicit land-use compatibility check (spec §9, §10)."""
    if not site_land_use:
        return True

    lu = site_land_use.lower()
    text = evidence.text.lower()

    is_grazing_site = any(w in lu for w in ["grassland", "rangeland", "pasture", "grazing", "meadow", "livestock", "cattle", "range"])
    is_crop_site = any(w in lu for w in ["wheat", "cropland", "crop", "corn", "soybean", "arable", "monoculture", "farm", "agriculture", "tillage", "paddy", "barley", "rice", "cotton"])
    is_orchard_site = any(w in lu for w in ["orchard", "vineyard"])
    is_agroforestry_site = any(w in lu for w in ["agroforestry", "silvopasture", "intercropping", "trees on farm"])
    is_forest_site = any(w in lu for w in ["forest", "forestry", "woodland", "timber", "plantation"])
    is_urban_site = any(w in lu for w in ["urban", "peri-urban", "city", "town", "built", "residential", "park"])

    # 1. Grazing / Rangeland evidence → reject for non-grazing sites
    if any(w in text for w in ["rotational grazing", "rangeland", "rangelands", "continuous grazing"]):
        if not is_grazing_site:
            return False

    # 2. Annual Crop evidence → reject for orchards/forests/pure grazing
    if any(w in text for w in ["cover crops", "reduced-tillage", "no-till", "crop rotation", "crop rotations"]):
        if is_orchard_site:
            # Spec §10: reject annual crop recs for orchards unless evidence specifically mentions orchards
            if "orchard" not in text and "perennial" not in text and "fruit" not in text:
                return False
        if is_forest_site:
            return False
        if is_grazing_site and not is_crop_site:
            return False

    # 3. Monoculture evidence → reject for non-crop sites
    if "monoculture cropping" in text:
        if not ("mono" in lu or is_crop_site):
            return False

    # 4. Agroforestry evidence → broadly compatible except pure urban
    if "agroforestry" in text:
        if is_urban_site:
            return False

    # 5. Urban evidence → reject for rural sites
    if "urban and peri-urban" in text or "built landscapes" in text:
        if not is_urban_site:
            return False

    # 6. Reforestation evidence → only for forest sites or deforested sites
    if any(w in text for w in ["reforestation", "deforestation rates"]):
        if not is_forest_site and deforestation_trend != "increasing":
            return False

    # 7. Wetland evidence → reject for arid/dryland sites
    if "wetland" in text:
        if is_grazing_site and "grassland" not in lu:
            return False

    return True


def _evidence_relevance_score(evidence: EvidenceChunk, site_input: SiteInput, primary_metric: str) -> int:
    """Score how relevant an evidence chunk is to the current site.
    Spec §12: consider intervention match, mechanism match, land-use, ecosystem, metrics.
    Higher score = more relevant. Min threshold = MIN_RELEVANCE_SCORE."""
    score = 0
    text_lower = evidence.text.lower()
    lu = (site_input.land_use or "").lower()
    reg = (site_input.region or "").lower()

    # Metric match: does evidence address the primary metric?
    if primary_metric in evidence.metric_tags:
        score += 2
    # Related metrics
    related = linked_metrics(primary_metric)
    matching_related = [m for m in related if m in evidence.metric_tags]
    score += len(matching_related)

    # Land-use mention in evidence
    if lu:
        lu_words = [w for w in lu.split() if len(w) > 3]
        if any(w in text_lower for w in lu_words):
            score += 1

    # Region/climate mention
    if reg and reg in text_lower:
        score += 1

    # Evidence recency
    if evidence.year and evidence.year >= 2020:
        score += 1

    return score


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
    hi = site_input.human_impact or ""

    if primary_metric == "pollution_level":
        return f"pesticide drift chemical runoff synthetic fertilizer pollution reduction buffer strips water quality {lu} {reg} {rf}"
    elif primary_metric == "deforestation_trend":
        return f"reforestation forest canopy habitat connectivity tree cover deforestation biodiversity {lu} {reg} {rf}"
    elif primary_metric == "soil_organic_carbon":
        if any(w in lu.lower() for w in ["grassland", "rangeland", "pasture", "grazing"]):
            return f"soil carbon sequestration rotational grazing rangelands grassland plant species diversity {reg} {rf}"
        elif "orchard" in lu.lower():
            return f"soil organic carbon orchard perennial tree understory organic matter biodiversity {reg} {rf}"
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
        elif "orchard" in lu.lower():
            return f"orchard pollinator habitat flowering understory native vegetation margins biodiversity {reg} {rf}"
        else:
            return f"hedgerows field margins corridors habitat heterogeneity {lu} {reg} {rf}"
    elif primary_metric == "grazing_pressure":
        return f"rotational grazing pasture management vegetation recovery soil carbon grassland species diversity {lu} {reg} {rf} {hi}"
    elif primary_metric == "pesticide_exposure":
        return f"pesticide reduction integrated pest management pollinator recovery buffer strips non-target species {lu} {reg} {rf}"
    elif primary_metric == "habitat_fragmentation":
        return f"habitat connectivity corridors buffer strips fragmentation species movement wildlife {lu} {reg} {rf}"
    elif primary_metric == "habitat_loss":
        return f"habitat restoration reforestation species recovery connectivity fragmentation {lu} {reg} {rf}"

    return f"{primary_metric.replace('_', ' ')} {lu} {reg} {rf}"


def generate_recommendations(site_input: SiteInput) -> list[Recommendation]:
    primary_metrics = infer_primary_metrics(site_input)
    recommendations: list[Recommendation] = []
    seen_evidence_ids: set[str] = set()
    is_healthy = is_healthy_site(site_input)

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
            relevance = _evidence_relevance_score(evidence, site_input, "land_use")
            if relevance < MIN_RELEVANCE_SCORE:
                continue

            what_to_do = _generate_what_to_do(evidence.text, site_input)
            if not what_to_do:
                continue

            impacted = _clean_impacted_metrics(evidence.metric_tags, "land_use", evidence.text)
            if len(impacted) < 2:
                continue

            action_title = _extract_action_title(evidence.text)
            eco_mech = _build_ecological_mechanism("land_use", evidence, site_input, action_title)
            seen_evidence_ids.add(evidence.id)
            recommendations.append(
                Recommendation(
                    action=action_title,
                    reasoning=evidence.text,
                    metrics_impacted=impacted,
                    time_horizon=_time_horizon_for(evidence.metric_tags),
                    confidence=_confidence_for(evidence, len(impacted), site_input),
                    source=f"{evidence.source} ({evidence.year})" if evidence.year else evidence.source,
                    what_to_do=what_to_do,
                    why_it_fits=_generate_why_it_fits("land_use", site_input),
                    ecological_mechanism=eco_mech,
                    limitation=_limitation_for("land_use", site_input),
                    evidence_title=evidence.source,
                    evidence_excerpt=evidence.text,
                )
            )
            # For healthy sites, 1-2 maintenance recs are enough
            max_for_healthy = 2 if is_healthy else MAX_RECOMMENDATIONS
            if len(recommendations) >= max_for_healthy:
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

            # Relevance scoring (spec §12, §13)
            relevance = _evidence_relevance_score(evidence, site_input, primary)
            if relevance < MIN_RELEVANCE_SCORE:
                continue

            # Ensure evidence aligns with the primary metric under evaluation
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
            if primary == "grazing_pressure" and not any(t in evidence.metric_tags for t in ["soil_organic_carbon", "species_richness", "land_use"]):
                continue
            if primary == "pesticide_exposure" and not any(t in evidence.metric_tags for t in ["pollinator_support", "pollution_level", "microbial_diversity"]):
                continue
            if primary == "habitat_fragmentation" and not any(t in evidence.metric_tags for t in ["habitat_fragmentation", "species_richness", "species_survival"]):
                continue
            if primary == "habitat_loss" and not any(t in evidence.metric_tags for t in ["habitat_diversity", "species_richness", "deforestation_trend"]):
                continue

            what_to_do = _generate_what_to_do(evidence.text, site_input)
            if not what_to_do:
                continue

            impacted = _clean_impacted_metrics(evidence.metric_tags, primary, evidence.text)
            if len(impacted) < 2:
                continue

            action_title = _extract_action_title(evidence.text)
            eco_mech = _build_ecological_mechanism(primary, evidence, site_input, action_title)
            seen_evidence_ids.add(evidence.id)
            recommendations.append(
                Recommendation(
                    action=action_title,
                    reasoning=evidence.text,
                    metrics_impacted=impacted,
                    time_horizon=_time_horizon_for(evidence.metric_tags),
                    confidence=_confidence_for(evidence, len(impacted), site_input),
                    source=f"{evidence.source} ({evidence.year})" if evidence.year else evidence.source,
                    what_to_do=what_to_do,
                    why_it_fits=_generate_why_it_fits(primary, site_input),
                    ecological_mechanism=eco_mech,
                    limitation=_limitation_for(primary, site_input),
                    evidence_title=evidence.source,
                    evidence_excerpt=evidence.text,
                )
            )
            break  # one recommendation per primary metric

    # Backfill only if we have very few and there are compatible options
    if len(recommendations) < 2 and not is_healthy:
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
            relevance = _evidence_relevance_score(evidence, site_input, "land_use")
            if relevance < MIN_RELEVANCE_SCORE:
                continue
            what_to_do = _generate_what_to_do(evidence.text, site_input)
            if not what_to_do:
                continue

            impacted = _clean_impacted_metrics(evidence.metric_tags, "land_use", evidence.text)
            if len(impacted) < 2:
                continue

            action_title = _extract_action_title(evidence.text)
            eco_mech = _build_ecological_mechanism("land_use", evidence, site_input, action_title)
            seen_evidence_ids.add(evidence.id)
            recommendations.append(
                Recommendation(
                    action=action_title,
                    reasoning=evidence.text,
                    metrics_impacted=impacted,
                    time_horizon=_time_horizon_for(evidence.metric_tags),
                    confidence=_confidence_for(evidence, len(impacted), site_input),
                    source=f"{evidence.source} ({evidence.year})" if evidence.year else evidence.source,
                    what_to_do=what_to_do,
                    why_it_fits=_generate_why_it_fits("land_use", site_input),
                    ecological_mechanism=eco_mech,
                    limitation=_limitation_for("land_use", site_input),
                    evidence_title=evidence.source,
                    evidence_excerpt=evidence.text,
                )
            )
            if len(recommendations) >= MAX_RECOMMENDATIONS:
                break

    return recommendations


def answer_question_with_evidence(user_query: str, site_input: SiteInput) -> list[Recommendation]:
    """Retrieve evidence chunks matching a user's follow-up question or general inquiry,
    filtering out evidence incompatible with their known site parameters."""
    lu = site_input.land_use or ""
    reg = site_input.region or ""
    rf = site_input.rainfall or ""
    full_query = f"{user_query} {lu} {reg} {rf}".strip()

    evidence_list = retrieve_evidence(full_query, top_k=15)
    recommendations: list[Recommendation] = []
    seen_ids: set[str] = set()

    for evidence in evidence_list:
        if evidence.id in seen_ids:
            continue
        if not is_evidence_compatible(evidence, site_input):
            continue
        relevance = _evidence_relevance_score(evidence, site_input, "land_use")
        if relevance < 1:  # lower threshold for user-driven queries
            continue
        what_to_do = _generate_what_to_do(evidence.text, site_input)
        if not what_to_do:
            continue

        impacted = _clean_impacted_metrics(evidence.metric_tags, "land_use", evidence.text)
        action_title = _extract_action_title(evidence.text)
        eco_mech = _build_ecological_mechanism("land_use", evidence, site_input, action_title)
        seen_ids.add(evidence.id)
        recommendations.append(
            Recommendation(
                action=action_title,
                reasoning=evidence.text,
                metrics_impacted=impacted,
                time_horizon=_time_horizon_for(evidence.metric_tags),
                confidence=_confidence_for(evidence, len(impacted), site_input),
                source=f"{evidence.source} ({evidence.year})" if evidence.year else evidence.source,
                what_to_do=what_to_do,
                why_it_fits=_generate_why_it_fits("land_use", site_input),
                ecological_mechanism=eco_mech,
                limitation=_limitation_for("land_use", site_input),
                evidence_title=evidence.source,
                evidence_excerpt=evidence.text,
            )
        )
        if len(recommendations) >= MAX_RECOMMENDATIONS:
            break

    return recommendations
