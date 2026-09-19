from app.parsing import rule_based_parse, parse_message, is_greeting
from app.models import SiteInput, Recommendation


def test_rule_based_parse_extracts_soc_and_rainfall():
    msg = "Soil organic carbon is 0.3%, rainfall is low, monoculture wheat, semi-arid region"
    data = rule_based_parse(msg)
    assert data.soil_organic_carbon_pct == 0.3
    assert data.rainfall == "low"
    assert data.region == "semi-arid"
    assert "wheat" in (data.land_use or "").lower()


def test_missing_required_fields():
    site = SiteInput(soil_organic_carbon_pct=0.5)
    missing = site.missing_required()
    assert "rainfall" in missing
    assert "land_use" in missing
    assert "region" in missing
    assert "soil_organic_carbon_pct" not in missing


def test_recommendation_schema_requires_multiple_metrics():
    rec = Recommendation(
        action="Introduce legume cover crops",
        reasoning="Fixes nitrogen, raises SOC 15-25% over 2-3 years.",
        metrics_impacted=["soil_organic_carbon", "microbial_diversity"],
        time_horizon="medium",
        confidence="high",
        source="FAO (2021)",
    )
    assert len(rec.metrics_impacted) >= 2


def test_numeric_rainfall_in_mm_is_bucketed_when_field_pending():
    pending = SiteInput(soil_organic_carbon_pct=0.3)
    result = parse_message("rainfall 300 mm", pending)
    assert result.site_input.rainfall == "low"
    assert result.understood


def test_rainfall_accepts_cm_and_inches():
    pending = SiteInput(soil_organic_carbon_pct=0.3)
    cm_result = parse_message("60 cm of rain a year", pending)
    assert cm_result.site_input.rainfall == "medium"  # 600mm

    inch_result = parse_message("40 inches annually", pending)
    assert inch_result.site_input.rainfall == "high"  # ~1016mm


def test_rainfall_accepts_descriptive_words():
    pending = SiteInput(soil_organic_carbon_pct=0.3)
    result = parse_message("it's pretty scarce here", pending)
    assert result.site_input.rainfall == "low"


def test_typo_in_land_use_is_corrected_when_field_pending():
    pending = SiteInput(soil_organic_carbon_pct=0.3, rainfall="low")
    result = parse_message("used fro monoclture", pending)
    assert result.site_input.land_use == "monoculture"
    assert result.understood


def test_typo_in_region_is_corrected_when_field_pending():
    pending = SiteInput(soil_organic_carbon_pct=0.3, rainfall="low", land_use="monoculture")
    result = parse_message("temprate", pending)
    assert result.site_input.region == "temperate"


def test_greeting_detection():
    assert is_greeting("hello")
    assert is_greeting("Hi!")
    assert is_greeting("good morning")
    assert not is_greeting("hi, rainfall is low")
    assert not is_greeting("monoculture wheat")


def test_unrecognized_reply_marked_not_understood():
    # rainfall/SOC require identifiable content; land_use/region are free
    # text and accept almost anything rather than looping forever.
    pending = SiteInput(soil_organic_carbon_pct=0.3)
    result = parse_message("xyz123 blorp", pending)
    assert result.understood is False


def test_greeting_never_marked_as_misunderstood():
    pending = SiteInput()
    result = parse_message("hello", pending)
    assert result.understood is True
