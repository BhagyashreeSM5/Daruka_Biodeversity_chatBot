"""
Comprehensive tests for:
1. Site state / memory contamination prevention (spec §1)
2. Field origin tracking (KNOWN / INFERRED / UNKNOWN) (spec §2)
3. Multi-metric reasoning requirements (spec §3)
4. Applicability filtering across ecosystems (spec §4)
5. Structured recommendation output format (spec §5)
"""

import pytest
from app.models import SiteInput, Recommendation
from app.parsing import parse_message, rule_based_parse
from app.reasoning import generate_recommendations
from app.relationships import is_healthy_site, infer_primary_metrics
from app.agent import get_graph


def test_site_state_contamination_fix():
    """Spec §1: Previous site Tropical/Pasture/High pollution/Increasing deforestation
    followed by new site Mediterranean/Orchard/Moderate pesticide use.
    Must NOT retain Tropical, Pasture, High pollution, or Increasing deforestation."""

    # Previous site state
    site1 = SiteInput(
        region="tropical",
        land_use="pasture",
        pollution_level="high",
        deforestation_trend="increasing",
        soil_organic_carbon_pct=1.0,
        rainfall="high",
        assessment_completed=True,
    )
    site1.set_known("region")
    site1.set_known("land_use")
    site1.set_known("pollution_level")
    site1.set_known("deforestation_trend")

    # New message with new ecosystem
    new_message = "Region: Mediterranean, Land use: Orchard, Human impact: Moderate pesticide use"
    res = parse_message(new_message, site1)
    new_site = res.site_input

    # Must have new values
    assert new_site.region == "mediterranean"
    assert new_site.land_use == "orchard"
    assert new_site.human_impact is not None
    assert "pesticide" in new_site.human_impact.lower()

    # Must NOT retain old polluted/deforested state
    assert new_site.pollution_level is None, f"Expected None but got {new_site.pollution_level}"
    assert new_site.deforestation_trend is None, f"Expected None but got {new_site.deforestation_trend}"
    assert new_site.field_origin.get("pollution_level") == "UNKNOWN"
    assert new_site.field_origin.get("deforestation_trend") == "UNKNOWN"


def test_field_origin_tracking():
    """Spec §2: Origins must be explicitly distinguished: KNOWN, INFERRED, UNKNOWN."""
    site = SiteInput(
        soil_organic_carbon_pct=0.4,
        rainfall="low",
        land_use="monoculture wheat",
        region="semi-arid",
    )
    site.set_known("soil_organic_carbon_pct")
    site.set_known("rainfall")
    site.set_known("land_use")
    site.set_known("region")

    assert site.get_origin("soil_organic_carbon_pct") == "KNOWN"
    assert site.get_origin("rainfall") == "KNOWN"
    assert site.get_origin("soil_ph") == "UNKNOWN"
    assert site.get_origin("pollution_level") == "UNKNOWN"

    site.set_inferred("soil_moisture")
    assert site.get_origin("soil_moisture") == "INFERRED"


def test_recommendation_multi_metric_requirement():
    """Spec §3: Every recommendation must touch at least two distinct metrics."""
    site = SiteInput(
        soil_organic_carbon_pct=0.35,
        rainfall="low",
        land_use="monoculture wheat",
        region="semi-arid",
    )
    recs = generate_recommendations(site)
    assert len(recs) > 0

    for r in recs:
        assert len(r.metrics_impacted) >= 2, f"Recommendation '{r.action}' has fewer than 2 metrics impacted: {r.metrics_impacted}"
        assert r.what_to_do, f"Recommendation '{r.action}' missing what_to_do"
        assert r.why_it_fits, f"Recommendation '{r.action}' missing why_it_fits"
        assert r.ecological_mechanism, f"Recommendation '{r.action}' missing ecological_mechanism"
        assert r.confidence.lower() in ["high", "medium", "low"], f"Invalid confidence {r.confidence}"
        assert r.limitation, f"Recommendation '{r.action}' missing limitation"


def test_orchard_applicability_filtering():
    """Spec §4: Orchard ecosystem must reject annual crop rotation and broadacre tillage."""
    site = SiteInput(
        soil_organic_carbon_pct=1.8,
        rainfall="medium",
        land_use="orchard",
        region="mediterranean",
        human_impact="moderate pesticide use",
    )
    recs = generate_recommendations(site)
    assert len(recs) > 0

    forbidden_in_orchards = [
        "crop rotation",
        "rotational grazing",
        "overgrazing",
        "rangeland",
        "pasture",
    ]
    for r in recs:
        text = f"{r.action} {r.what_to_do} {r.why_it_fits} {r.ecological_mechanism}".lower()
        for forbidden in forbidden_in_orchards:
            assert forbidden not in text, f"Found forbidden '{forbidden}' in orchard rec: {text}"


def test_healthy_site_does_not_manufacture_urgent_pressures():
    """Spec §5: Sites with good conditions must not report false urgent crises."""
    healthy_site = SiteInput(
        soil_organic_carbon_pct=3.5,
        rainfall="high",
        land_use="agroforestry",
        region="temperate",
        pollution_level="low",
        deforestation_trend="stable",
    )
    assert is_healthy_site(healthy_site) is True
    pressures = infer_primary_metrics(healthy_site)
    assert len(pressures) == 0

    graph = get_graph()
    result = graph.invoke(
        {
            "session_id": "test_healthy_resilience",
            "turn_count": 2,
            "message": None,
            "incoming_data": None,
            "site_input": healthy_site,
        }
    )
    reply_lower = result["reply"].lower()
    assert "no urgent" in reply_lower or "general resilience" in reply_lower


def test_user_value_preservation():
    """Spec §5: User-provided values must NEVER be corrupted.
    If user says Rainfall: High in a semi-arid region, it MUST remain high."""
    site = SiteInput(
        soil_organic_carbon_pct=0.35,
        rainfall="high",
        land_use="monoculture wheat",
        region="semi-arid",
    )
    site.set_known("rainfall", "high")
    site.set_known("soil_organic_carbon_pct", 0.35)
    site.set_known("region", "semi-arid")
    site.set_known("land_use", "monoculture wheat")

    # Simulate downstream processes and validation
    site.validate_user_values()
    assert site.rainfall == "high"
    assert site.region == "semi-arid"
    assert site.get_origin("rainfall") == "KNOWN"

    # Even if someone attempts to mutate it, validate_user_values restores it
    site.rainfall = "low"
    site.validate_user_values()
    assert site.rainfall == "high"


def test_unknown_fields_remain_unknown():
    """Spec §6: Never invent rainfall, temperature, soil moisture, pollution,
    deforestation, biodiversity status, human impact when not provided."""
    msg = "Analyze this site: Land use is pasture, region is temperate, soil organic carbon is 1.2%, rainfall is medium."
    res = parse_message(msg, SiteInput())
    site = res.site_input

    # Known fields
    assert site.land_use == "pasture"
    assert site.region == "temperate"
    assert site.soil_organic_carbon_pct == 1.2
    assert site.rainfall == "medium"

    # Unknown fields must strictly remain None and UNKNOWN
    for unknown_f in ["soil_ph", "soil_moisture", "temperature", "pollution_level",
                      "deforestation_trend", "human_impact", "biodiversity_status"]:
        assert getattr(site, unknown_f) is None, f"Field {unknown_f} was invented!"
        assert site.get_origin(unknown_f) == "UNKNOWN", f"Field {unknown_f} origin is not UNKNOWN!"


def test_multi_turn_correction():
    """Spec §7: Distinguish correcting information from new site or simple addition."""
    site = SiteInput(
        soil_organic_carbon_pct=1.0,
        rainfall="medium",
        land_use="cropland",
        region="temperate",
    )
    site.set_known("land_use", "cropland")

    correction_msg = "Actually, the land is an orchard, not cropland."
    res = parse_message(correction_msg, site)
    updated = res.site_input

    assert res.is_correction is True
    assert updated.land_use == "orchard"
    # Other fields must remain intact
    assert updated.soil_organic_carbon_pct == 1.0
    assert updated.rainfall == "medium"
    assert updated.region == "temperate"


def test_nested_structured_json_normalization():
    """Spec §27: Structured nested JSON must normalize into the standard SiteInput
    and drive the same reasoning pipeline."""
    payload = {
        "region": "mediterranean",
        "land_use": "orchard",
        "soil": {
            "organic_carbon": "1.4%",
            "ph": 6.8,
            "moisture": "low",
        },
        "climate": {
            "rainfall": "420mm",
            "temperature": "warm",
        },
        "human_impact": "moderate pesticide use",
        "biodiversity": {
            "species_richness": "moderate",
            "habitat_diversity": "low",
        },
    }
    site = SiteInput.from_dict_or_nested(payload)

    assert site.region == "mediterranean"
    assert site.land_use == "orchard"
    assert site.soil_organic_carbon_pct == 1.4
    assert site.soil_ph == 6.8
    assert site.soil_moisture == "low"
    assert site.rainfall == "low"  # 420mm -> low
    assert site.temperature == "warm"
    assert site.human_impact == "moderate pesticide use"
    assert site.species_richness == "moderate"
    assert site.habitat_diversity == "low"
    assert site.get_origin("soil_organic_carbon_pct") == "KNOWN"
    assert site.get_origin("rainfall") == "KNOWN"

    # Drives genuine recommendations
    recs = generate_recommendations(site)
    assert len(recs) > 0
    for r in recs:
        assert r.what_to_do
        assert r.why_it_fits
        assert r.ecological_mechanism


def test_impacted_metrics_are_true_outcomes():
    """Spec §23: Impacted metrics must be genuine environmental outcomes,
    NEVER input conditions like rainfall, land use, temperature, or region."""
    site = SiteInput(
        soil_organic_carbon_pct=0.35,
        rainfall="low",
        land_use="monoculture wheat",
        region="semi-arid",
    )
    recs = generate_recommendations(site)
    assert len(recs) > 0

    forbidden_metrics = {"rainfall", "land_use", "temperature", "region", "latitude", "longitude"}
    for r in recs:
        for m in r.metrics_impacted:
            assert m.lower() not in forbidden_metrics, f"Forbidden input condition '{m}' found in metrics_impacted!"
        assert len(r.metrics_impacted) >= 2, f"Expected >=2 impacted metrics, got {r.metrics_impacted}"


def test_no_generic_filler_in_recommendations():
    """Spec §24: Never output generic filler boilerplate like
    'Implement the recommended practice according to local conditions'."""
    site = SiteInput(
        soil_organic_carbon_pct=0.35,
        rainfall="low",
        land_use="monoculture wheat",
        region="semi-arid",
    )
    recs = generate_recommendations(site)
    assert len(recs) > 0

    forbidden_filler = "implement the recommended practice according to local conditions"
    for r in recs:
        assert forbidden_filler not in r.what_to_do.lower(), f"Found generic filler in what_to_do: {r.what_to_do}"
        assert len(r.what_to_do) > 20, f"what_to_do is too short/vague: {r.what_to_do}"


def test_multi_metric_ecological_reasoning_connects_variables():
    """Spec §10: Multi-metric reasoning must connect >= 3 environmental variables
    dynamically from the site in the ecological mechanism."""
    site = SiteInput(
        soil_organic_carbon_pct=0.35,
        rainfall="low",
        land_use="monoculture wheat",
        region="semi-arid",
    )
    recs = generate_recommendations(site)
    assert len(recs) > 0

    for r in recs:
        mech = r.ecological_mechanism.lower()
        # Check that mechanism addresses site conditions, interactions, and intervention
        assert any(term in mech for term in ["soil organic carbon", "carbon", "soc", "0.35"]), f"Mechanism missing carbon context: {mech}"
        assert any(term in mech for term in ["rainfall", "moisture", "water", "precipitation"]), f"Mechanism missing water context: {mech}"
        assert any(term in mech for term in ["wheat", "monoculture", "semi-arid", "land use"]), f"Mechanism missing land-use context: {mech}"


def test_structured_output_format_spec_compliance():
    """Spec §22: Ensure the output follows the exact required structure:
    ## Site assessment
    Current conditions:
    Main environmental pressures:
    Unknowns:
    ## Recommendation 1
    ### What to do
    ### Why it fits this site
    ### Ecological mechanism
    ### Impacted metrics
    ### Time horizon
    ### Confidence
    ### Evidence
    ### Limitation"""
    site = SiteInput(
        soil_organic_carbon_pct=0.35,
        rainfall="low",
        land_use="monoculture wheat",
        region="semi-arid",
    )
    site.set_known("soil_organic_carbon_pct", 0.35)
    site.set_known("rainfall", "low")
    site.set_known("land_use", "monoculture wheat")
    site.set_known("region", "semi-arid")

    graph = get_graph()
    result = graph.invoke(
        {
            "session_id": "test_output_structure_spec",
            "turn_count": 1,
            "message": "Analyze this site: Region: semi-arid, Land use: monoculture wheat, SOC: 0.35%, Rainfall: low",
            "incoming_data": None,
            "site_input": site,
        }
    )

    reply = result["reply"]
    assert "## Site assessment" in reply
    assert "Current conditions:" in reply
    assert "Main environmental pressures:" in reply
    assert "Unknowns:" in reply
    assert "## Recommendation 1" in reply
    assert "### What to do" in reply
    assert "### Why it fits this site" in reply
    assert "### Ecological mechanism" in reply
    assert "### Impacted metrics" in reply
    assert "### Time horizon" in reply
    assert "### Confidence" in reply
    assert "### Evidence" in reply
    assert "### Limitation" in reply

