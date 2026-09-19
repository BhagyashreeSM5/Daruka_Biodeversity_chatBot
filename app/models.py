from typing import Any, Literal, Optional
import re
from pydantic import BaseModel, Field


SITE_FIELDS = [
    "soil_organic_carbon_pct",
    "soil_ph",
    "soil_moisture",
    "rainfall",
    "temperature",
    "land_use",
    "region",
    "latitude",
    "longitude",
    "pollution_level",
    "deforestation_trend",
    "human_impact",
    "biodiversity_status",
    "species_richness",
    "habitat_diversity",
]


class SiteInput(BaseModel):
    """Structured environmental input for a piece of land. All fields optional
    at the schema level — completeness is enforced by the conversation graph,
    not by validation, so partial multi-turn input is possible."""

    soil_organic_carbon_pct: Optional[float] = Field(
        None, description="Soil organic carbon, percent"
    )
    soil_ph: Optional[float] = Field(None, description="Soil pH, optional")
    soil_moisture: Optional[str] = Field(
        None, description="Soil moisture, optional (e.g. 'low', 'medium', 'high')"
    )
    rainfall: Optional[Literal["low", "medium", "high"]] = None
    temperature: Optional[str] = Field(
        None, description="e.g. 'warm', 'temperate', 'cold', 'hot'"
    )
    land_use: Optional[str] = Field(
        None, description="e.g. 'monoculture wheat', 'agroforestry', 'grassland'"
    )
    region: Optional[str] = Field(None, description="e.g. 'semi-arid', 'tropical'")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    pollution_level: Optional[Literal["low", "medium", "high"]] = None
    deforestation_trend: Optional[Literal["stable", "increasing", "decreasing"]] = None

    # --- New fields for full-spec compliance ---
    human_impact: Optional[str] = Field(
        None,
        description="Human pressure description (e.g. 'overgrazing', 'moderate pesticide use', 'road construction')",
    )
    biodiversity_status: Optional[str] = Field(
        None,
        description="Biodiversity observation (e.g. 'high species richness', 'low pollinator diversity')",
    )
    species_richness: Optional[str] = Field(
        None, description="e.g. 'high', 'moderate', 'low'"
    )
    habitat_diversity: Optional[str] = Field(
        None, description="e.g. 'high', 'fragmented', 'poor'"
    )

    # Tracks origin of each field value: KNOWN / INFERRED / UNKNOWN
    field_origin: dict[str, str] = Field(default_factory=dict)

    # Strictly preserves explicit user-provided values to prevent corruption
    user_values: dict[str, Any] = Field(default_factory=dict)

    intro_shown: bool = False
    assessment_completed: bool = False

    def model_post_init(self, __context) -> None:
        for f in SITE_FIELDS:
            if f not in self.field_origin:
                val = getattr(self, f, None)
                if val is not None:
                    self.field_origin[f] = "KNOWN"
                    if f not in self.user_values:
                        self.user_values[f] = val
                else:
                    self.field_origin[f] = "UNKNOWN"

    def missing_required(self) -> list[str]:
        required = ["soil_organic_carbon_pct", "rainfall", "land_use", "region"]
        return [f for f in required if getattr(self, f) is None]

    def set_known(self, field: str, value: Any = None) -> None:
        """Mark a field as directly supplied by the user and preserve its value."""
        self.field_origin[field] = "KNOWN"
        if value is not None:
            self.user_values[field] = value
            setattr(self, field, value)
        else:
            v = getattr(self, field, None)
            if v is not None:
                self.user_values[field] = v

    def set_inferred(self, field: str, value: Any = None) -> None:
        """Mark a field as scientifically derived from known data."""
        self.field_origin[field] = "INFERRED"
        if value is not None:
            setattr(self, field, value)

    def get_origin(self, field: str) -> str:
        """Return the origin of a field value — KNOWN, INFERRED, or UNKNOWN."""
        return self.field_origin.get(field, "UNKNOWN")

    def validate_user_values(self) -> None:
        """Enforce Section 5: User-provided values must NEVER be corrupted or overwritten.
        For every KNOWN field: final value == latest explicit user-provided value."""
        for f, expected in self.user_values.items():
            if expected is not None:
                setattr(self, f, expected)
                self.field_origin[f] = "KNOWN"

    def known_fields_summary(self) -> dict[str, str]:
        """Return a dict of field -> origin for all non-None site fields."""
        result = {}
        for f in SITE_FIELDS:
            val = getattr(self, f, None)
            if val is not None:
                result[f] = self.get_origin(f)
            else:
                result[f] = "UNKNOWN"
        return result

    @classmethod
    def from_dict_or_nested(cls, raw: dict) -> "SiteInput":
        """Normalize flat or nested environmental JSON payloads into SiteInput."""
        normalized: dict[str, Any] = {}

        # Handle nested soil object
        soil = raw.get("soil")
        if isinstance(soil, dict):
            soc = soil.get("organic_carbon") or soil.get("soil_organic_carbon_pct") or soil.get("soc")
            if soc is not None:
                try:
                    soc_val = float(re.sub(r"[^\d.]", "", str(soc)))
                    normalized["soil_organic_carbon_pct"] = soc_val
                except (ValueError, TypeError):
                    pass
            if "ph" in soil or "soil_ph" in soil:
                try:
                    normalized["soil_ph"] = float(soil.get("ph") or soil.get("soil_ph"))
                except (ValueError, TypeError):
                    pass
            if "moisture" in soil or "soil_moisture" in soil:
                normalized["soil_moisture"] = str(soil.get("moisture") or soil.get("soil_moisture")).lower()

        # Handle nested climate object
        climate = raw.get("climate")
        if isinstance(climate, dict):
            rf = climate.get("rainfall")
            if rf is not None:
                rf_str = str(rf).lower()
                if rf_str in ("low", "medium", "high"):
                    normalized["rainfall"] = rf_str
                else:
                    # check if mm / numbers
                    mm_match = re.search(r"(\d+(?:\.\d+)?)", rf_str)
                    if mm_match:
                        val = float(mm_match.group(1))
                        if "in" in rf_str:
                            val *= 25.4
                        elif "cm" in rf_str:
                            val *= 10.0
                        if val < 500:
                            normalized["rainfall"] = "low"
                        elif val < 1000:
                            normalized["rainfall"] = "medium"
                        else:
                            normalized["rainfall"] = "high"
                    elif any(w in rf_str for w in ["arid", "dry", "sparse", "minimal", "low"]):
                        normalized["rainfall"] = "low"
                    elif any(w in rf_str for w in ["heavy", "abundant", "wet", "high"]):
                        normalized["rainfall"] = "high"
                    else:
                        normalized["rainfall"] = "medium"

            if "temperature" in climate:
                normalized["temperature"] = str(climate["temperature"]).lower()

        # Handle nested biodiversity object
        biodiversity = raw.get("biodiversity")
        if isinstance(biodiversity, dict):
            sr = biodiversity.get("species_richness")
            hd = biodiversity.get("habitat_diversity")
            ps = biodiversity.get("pollinator_status")
            if sr:
                normalized["species_richness"] = str(sr).lower()
            if hd:
                normalized["habitat_diversity"] = str(hd).lower()
            parts = [f"{k}: {v}" for k, v in biodiversity.items() if v]
            if parts:
                normalized["biodiversity_status"] = ", ".join(parts)
        elif isinstance(biodiversity, str):
            normalized["biodiversity_status"] = biodiversity

        # Direct root keys
        for k, v in raw.items():
            if k in ("soil", "climate", "biodiversity"):
                continue
            if k in SITE_FIELDS and v is not None:
                if k == "soil_organic_carbon_pct":
                    try:
                        normalized[k] = float(re.sub(r"[^\d.]", "", str(v)))
                    except (ValueError, TypeError):
                        pass
                elif k == "soil_ph":
                    try:
                        normalized[k] = float(v)
                    except (ValueError, TypeError):
                        pass
                elif k == "rainfall":
                    v_str = str(v).lower()
                    if v_str in ("low", "medium", "high"):
                        normalized[k] = v_str
                else:
                    normalized[k] = v

        site = cls(**normalized)
        for k in normalized.keys():
            site.set_known(k)
        return site


class EvidenceChunk(BaseModel):
    id: str
    text: str
    source: str
    metric_tags: list[str]
    year: Optional[int] = None


class Recommendation(BaseModel):
    action: str
    reasoning: str
    metrics_impacted: list[str]
    time_horizon: Literal["short", "medium", "long"]
    confidence: Literal["low", "medium", "high"]
    source: str
    # --- New structured fields (spec §10, §15, §17) ---
    what_to_do: str = ""
    why_it_fits: str = ""
    ecological_mechanism: str = ""
    limitation: str = ""
    evidence_title: str = ""
    evidence_excerpt: str = ""

    def model_post_init(self, __context) -> None:
        if not self.ecological_mechanism and self.reasoning:
            self.ecological_mechanism = self.reasoning
        elif not self.reasoning and self.ecological_mechanism:
            self.reasoning = self.ecological_mechanism
        if not self.evidence_title and self.source:
            self.evidence_title = self.source
        if not self.evidence_excerpt and self.reasoning:
            self.evidence_excerpt = self.reasoning


class ChatTurn(BaseModel):
    session_id: str
    message: Optional[str] = None
    data: Optional[Any] = None  # accepts SiteInput or dict (nested JSON)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    status: Literal["awaiting_input", "complete"]
    missing_fields: list[str] = []
    recommendations: list[Recommendation] = []
