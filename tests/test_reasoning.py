from app.models import SiteInput
from app.reasoning import generate_recommendations, is_evidence_compatible


def test_diverse_recommendations_and_no_contradictions():
    input_a = SiteInput(
        soil_organic_carbon_pct=0.3,
        rainfall="high",
        land_use="cropland",
        region="tropical",
        pollution_level="high",
    )
    input_b = SiteInput(
        soil_organic_carbon_pct=0.4,
        rainfall="low",
        land_use="monoculture wheat",
        region="semi-arid",
        pollution_level="low",
    )
    input_c = SiteInput(
        soil_organic_carbon_pct=2.0,
        rainfall="medium",
        land_use="grassland",
        region="temperate",
        deforestation_trend="increasing",
    )
    input_d = SiteInput(
        soil_organic_carbon_pct=0.8,
        rainfall="medium",
        land_use="agroforestry",
        region="temperate",
    )

    recs_a = generate_recommendations(input_a)
    recs_b = generate_recommendations(input_b)
    recs_c = generate_recommendations(input_c)
    recs_d = generate_recommendations(input_d)

    # 1. Assert each input produces recommendations
    assert len(recs_a) > 0
    assert len(recs_b) > 0
    assert len(recs_c) > 0
    assert len(recs_d) > 0

    # 2. Assert all recommendations specify at least 2 impacted metrics
    for recs in [recs_a, recs_b, recs_c, recs_d]:
        for r in recs:
            assert len(r.metrics_impacted) >= 2

    # 3. Assert (a): recommendation sets are not identical across inputs
    actions_a = {r.action for r in recs_a}
    actions_b = {r.action for r in recs_b}
    actions_c = {r.action for r in recs_c}
    actions_d = {r.action for r in recs_d}

    assert actions_a != actions_b
    assert actions_b != actions_c
    assert actions_a != actions_c
    assert actions_c != actions_d

    # 4. Assert high priority signals get reflected:
    # High pollution input A must have pollution-related recommendation
    pollution_recs = [r for r in recs_a if "pollution_level" in r.metrics_impacted or "pesticide" in r.reasoning.lower()]
    assert len(pollution_recs) >= 1

    # Increasing deforestation input C must have deforestation-related recommendation
    deforestation_recs = [r for r in recs_c if "deforestation_trend" in r.metrics_impacted or "deforestation" in r.reasoning.lower()]
    assert len(deforestation_recs) >= 1

    # 5. Assert (b): No recommendation contradicts values explicitly given in that input
    for inp, recs in [(input_a, recs_a), (input_b, recs_b), (input_c, recs_c), (input_d, recs_d)]:
        for r in recs:
            reasoning_lower = r.reasoning.lower()

            # High rainfall should never cite low-rainfall techniques
            if inp.rainfall == "high":
                assert "low-rainfall" not in reasoning_lower and "low rainfall" not in reasoning_lower
                assert "semi-arid" not in reasoning_lower

            # Low rainfall should never cite wetland techniques
            if inp.rainfall == "low":
                assert "wetland" not in reasoning_lower

            # Pure cropland should not cite rotational grazing on rangelands
            if inp.land_use == "cropland":
                assert "rotational grazing" not in reasoning_lower and "rangelands" not in reasoning_lower

            # Grassland should not cite tillage or arable crop rotations
            if inp.land_use == "grassland":
                assert "tillage" not in reasoning_lower and "crop rotation" not in reasoning_lower

            # Non-urban sites should not cite urban corridors
            if "urban" not in (inp.land_use or "").lower():
                assert "urban and peri-urban" not in reasoning_lower
