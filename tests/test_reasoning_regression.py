"""
Regression tests verifying:
- Test 1: Coastal orchard, SOC 3.2%, high rainfall -> no contradiction, signals no urgent metric flagged
- Test 2: Alpine pasture, SOC 0.2%, low rainfall, pollution=high, deforestation=increasing -> covers >=3 of {SOC, rainfall, pollution, deforestation}
- Test 3: Subtropical polyculture, SOC 1.5%, medium rainfall, pollution=low -> no deforestation or rangeland
- Test 4: Semi-arid monoculture wheat, SOC 0.3%, low rainfall -> >=1 rec with soil_organic_carbon, >=1 with land_use or habitat_diversity
- Test 5: deforestation_trend never in infer_primary_metrics unless deforestation_trend == "increasing"
- Test 6: 5 different inputs -> no two produce identical action sets
- Bug 1: Turn 1 greeting/intent statement does not show retry message; Turn 2+ gibberish does
- Bug 3: Completed assessment with filler message returns post-assessment status without repeating reasoning
"""

from app.models import SiteInput
from app.relationships import infer_primary_metrics
from app.reasoning import generate_recommendations
from app.agent import get_graph


def test_regression_1_coastal_orchard():
    inp = SiteInput(
        soil_organic_carbon_pct=3.2,
        rainfall="high",
        land_use="orchard",
        region="coastal",
    )
    # 1. infer_primary_metrics should be empty
    primaries = infer_primary_metrics(inp)
    assert primaries == [], f"Expected empty primaries for healthy orchard, got {primaries}"

    # 2. Recommendations should not cite grazing/rangeland/grassland or deforestation
    recs = generate_recommendations(inp)
    assert len(recs) > 0
    forbidden = ["deforestation", "rangeland", "rangelands", "grassland", "rotational grazing"]
    for r in recs:
        text = (r.action + " " + r.reasoning).lower()
        for term in forbidden:
            assert term not in text, f"Forbidden term '{term}' found in recommendation: {text}"

    # 3. Agent reply signals no urgent metric flagged
    graph = get_graph()
    result = graph.invoke(
        {
            "session_id": "test_orchard_healthy",
            "turn_count": 2,
            "message": None,
            "incoming_data": None,
            "site_input": inp,
        }
    )
    reply_lower = result["reply"].lower()
    assert (
        "no urgent" in reply_lower
        or "nothing in your inputs indicates an urgent" in reply_lower
        or "general resilience" in reply_lower
    ), f"Expected no-urgent signal in reply: {result['reply']}"


def test_regression_2_alpine_pasture_multi_metric():
    inp = SiteInput(
        soil_organic_carbon_pct=0.2,
        rainfall="low",
        land_use="pasture",
        region="alpine",
        pollution_level="high",
        deforestation_trend="increasing",
    )
    recs = generate_recommendations(inp)
    assert len(recs) > 0

    all_impacted = {m for r in recs for m in r.metrics_impacted}
    text_pool = " ".join(f"{r.action} {r.reasoning}" for r in recs).lower()

    # Track how many of the 4 urgent domains are addressed
    domains_covered = set()
    if "soil_organic_carbon" in all_impacted or any(w in text_pool for w in ["soil organic carbon", "soil carbon", "soc", "compost", "grazing"]):
        domains_covered.add("soil_organic_carbon")
    if "rainfall" in all_impacted or "water_availability" in all_impacted or any(w in text_pool for w in ["rainfall", "rainwater", "water", "moisture", "drought"]):
        domains_covered.add("rainfall")
    if "pollution_level" in all_impacted or any(w in text_pool for w in ["pollution", "pesticide", "buffer strip", "chemical"]):
        domains_covered.add("pollution_level")
    if "deforestation_trend" in all_impacted or any(w in text_pool for w in ["deforestation", "reforestation", "forest", "canopy"]):
        domains_covered.add("deforestation_trend")

    assert len(domains_covered) >= 3, f"Expected >= 3 domains covered, got {domains_covered}"


def test_regression_3_subtropical_polyculture():
    inp = SiteInput(
        soil_organic_carbon_pct=1.5,
        rainfall="medium",
        land_use="polyculture cropping",
        region="subtropical",
        pollution_level="low",
    )
    recs = generate_recommendations(inp)
    assert len(recs) > 0

    forbidden = ["deforestation", "rangeland", "rangelands", "rotational grazing"]
    for r in recs:
        text = (r.action + " " + r.reasoning).lower()
        for term in forbidden:
            assert term not in text, f"Forbidden term '{term}' found in polyculture rec: {text}"


def test_regression_4_semi_arid_monoculture_wheat():
    inp = SiteInput(
        soil_organic_carbon_pct=0.3,
        rainfall="low",
        land_use="monoculture wheat",
        region="semi-arid",
    )
    recs = generate_recommendations(inp)
    assert len(recs) > 0

    has_soc = any(
        "soil_organic_carbon" in r.metrics_impacted or "soil" in r.reasoning.lower() or "carbon" in r.reasoning.lower()
        for r in recs
    )
    has_land_or_habitat = any(
        any(m in r.metrics_impacted for m in ["land_use", "habitat_diversity", "habitat_fragmentation", "species_richness"])
        or any(w in r.reasoning.lower() for w in ["monoculture", "rotation", "diversif", "crop"])
        for r in recs
    )

    assert has_soc, "Expected at least one recommendation addressing soil organic carbon"
    assert has_land_or_habitat, "Expected at least one recommendation addressing land use or habitat diversity"


def test_regression_5_deforestation_never_inferred_unless_increasing():
    # Never set
    inp1 = SiteInput(soil_organic_carbon_pct=1.0, rainfall="low", land_use="cropland", region="temperate")
    assert "deforestation_trend" not in infer_primary_metrics(inp1)

    # Set to stable
    inp2 = SiteInput(soil_organic_carbon_pct=1.0, rainfall="low", land_use="cropland", region="temperate", deforestation_trend="stable")
    assert "deforestation_trend" not in infer_primary_metrics(inp2)

    # Set to decreasing
    inp3 = SiteInput(soil_organic_carbon_pct=1.0, rainfall="low", land_use="cropland", region="temperate", deforestation_trend="decreasing")
    assert "deforestation_trend" not in infer_primary_metrics(inp3)

    # Set to increasing -> must be present
    inp4 = SiteInput(soil_organic_carbon_pct=1.0, rainfall="low", land_use="cropland", region="temperate", deforestation_trend="increasing")
    assert "deforestation_trend" in infer_primary_metrics(inp4)


def test_regression_6_variation_across_5_inputs():
    inputs = [
        SiteInput(soil_organic_carbon_pct=0.3, rainfall="low", land_use="monoculture wheat", region="semi-arid"),
        SiteInput(soil_organic_carbon_pct=2.5, rainfall="high", land_use="agroforestry", region="tropical"),
        SiteInput(soil_organic_carbon_pct=1.8, rainfall="medium", land_use="grassland", region="temperate"),
        SiteInput(soil_organic_carbon_pct=0.5, rainfall="high", land_use="cropland", region="temperate", pollution_level="high"),
        SiteInput(soil_organic_carbon_pct=3.0, rainfall="low", land_use="rangeland", region="semi-arid"),
    ]

    action_sets = [frozenset(r.action for r in generate_recommendations(inp)) for inp in inputs]

    # Verify each produced recommendations
    for s in action_sets:
        assert len(s) > 0

    # Verify no two action sets are identical
    for i in range(len(action_sets)):
        for j in range(i + 1, len(action_sets)):
            assert action_sets[i] != action_sets[j], f"Inputs {i} and {j} produced identical actions: {action_sets[i]}"


def test_regression_bug_1_first_turn_greeting_and_intent():
    graph = get_graph()

    # 1. Turn 1 greeting with typo: "helllo" -> must be warm intro + first question, NEVER "I couldn't quite catch that"
    res1 = graph.invoke(
        {
            "session_id": "test_turn1_greeting",
            "turn_count": 1,
            "message": "helllo",
            "incoming_data": None,
            "site_input": SiteInput(),
        }
    )
    assert "I couldn't quite catch" not in res1["reply"]
    assert "Darukaa.Earth biodiversity assistant" in res1["reply"]

    # 2. Turn 1 intent statement: "Biodiversity is declining on my land" -> calm warm intro + first question
    res2 = graph.invoke(
        {
            "session_id": "test_turn1_intent",
            "turn_count": 1,
            "message": "Biodiversity is declining on my land",
            "incoming_data": None,
            "site_input": SiteInput(),
        }
    )
    assert "I couldn't quite catch" not in res2["reply"]
    assert "Darukaa.Earth biodiversity assistant" in res2["reply"]

    # 3. Turn 2 gibberish: "blorp asdf" -> must show retry explanation
    res3 = graph.invoke(
        {
            "session_id": "test_turn2_gibberish",
            "turn_count": 2,
            "message": "blorp asdf",
            "incoming_data": None,
            "site_input": SiteInput(),
        }
    )
    assert "I couldn't quite catch" in res3["reply"]


def test_regression_bug_3_completed_assessment_no_filler():
    graph = get_graph()
    completed_input = SiteInput(
        soil_organic_carbon_pct=0.4,
        rainfall="low",
        land_use="monoculture wheat",
        region="semi-arid",
        assessment_completed=True,
    )

    # Post-assessment filler message
    res = graph.invoke(
        {
            "session_id": "test_completed_filler",
            "turn_count": 5,
            "message": "thanks, that was helpful",
            "incoming_data": None,
            "site_input": completed_input,
        }
    )
    assert res["status"] == "complete"
    assert "different scenario" in res["reply"].lower()
    assert res["recommendations"] == []
