from typing import Literal, Optional
from pydantic import BaseModel, Field


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
    land_use: Optional[str] = Field(
        None, description="e.g. 'monoculture wheat', 'agroforestry', 'grassland'"
    )
    region: Optional[str] = Field(None, description="e.g. 'semi-arid', 'tropical'")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    pollution_level: Optional[Literal["low", "medium", "high"]] = None
    deforestation_trend: Optional[Literal["stable", "increasing", "decreasing"]] = None
    intro_shown: bool = False
    assessment_completed: bool = False

    def missing_required(self) -> list[str]:
        required = ["soil_organic_carbon_pct", "rainfall", "land_use", "region"]
        return [f for f in required if getattr(self, f) is None]



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


class ChatTurn(BaseModel):
    session_id: str
    message: Optional[str] = None
    data: Optional[SiteInput] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    status: Literal["awaiting_input", "complete"]
    missing_fields: list[str] = []
    recommendations: list[Recommendation] = []
