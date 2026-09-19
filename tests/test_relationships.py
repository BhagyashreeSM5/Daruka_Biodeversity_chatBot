from app.relationships import infer_primary_metrics, linked_metrics, RELATIONSHIPS
from app.models import SiteInput


def test_relationships_have_multiple_links():
    for metric, links in RELATIONSHIPS.items():
        assert len(links) >= 1
        assert metric not in links  # no self-loops


def test_infer_primary_metrics_low_soc_is_urgent():
    site = SiteInput(soil_organic_carbon_pct=0.3, rainfall="low", land_use="monoculture wheat", region="semi-arid")
    primaries = infer_primary_metrics(site)
    assert "soil_organic_carbon" in primaries
    assert "land_use" in primaries
    assert "rainfall" in primaries


def test_linked_metrics_soil_organic_carbon():
    links = linked_metrics("soil_organic_carbon")
    assert "microbial_diversity" in links
    assert "water_availability" in links
