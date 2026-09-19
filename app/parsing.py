import re
import difflib
from dataclasses import dataclass

from app.models import SiteInput
from app.llm import get_llm

FIELD_LABELS = {
    "soil_organic_carbon_pct": "Soil Organic Carbon",
    "rainfall": "Rainfall",
    "land_use": "Land Use",
    "region": "Region",
    "soil_ph": "Soil pH",
    "soil_moisture": "Soil Moisture",
    "pollution_level": "Pollution Level",
    "deforestation_trend": "Deforestation Trend",
}

FIELD_QUESTIONS = {
    "soil_organic_carbon_pct": "What's the soil organic carbon percentage (e.g. 0.3%)?",
    "rainfall": "What's the rainfall pattern — low, medium, or high (a number in mm/cm/inches works too)?",
    "land_use": "What's the current land use (e.g. monoculture wheat, agroforestry, grassland)?",
    "region": "What type of region is this (e.g. semi-arid, tropical, temperate)?",
}

FIELD_HELP_EXAMPLES = {
    "soil_organic_carbon_pct": "e.g. '0.3%' or 'SOC is around 1.2'",
    "rainfall": "e.g. 'low', 'moderate', '650mm', or '20 inches a year'",
    "land_use": "e.g. 'monoculture wheat', 'agroforestry', 'grassland', 'pasture'",
    "region": "e.g. 'semi-arid', 'tropical', 'temperate', 'coastal'",
}

GREETING = (
    "Hi, I'm the Darukaa.Earth biodiversity assistant. To give you evidence-backed "
    "recommendations I need a few details about your site. "
)

CANONICAL_GREETINGS = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "namaste",
    "greetings",
    "yo",
    "howdy",
]

_GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|good\s*(morning|afternoon|evening)|namaste|greetings|yo|howdy)\b[\s!.,]*$",
    re.I,
)


def is_greeting(message: str) -> bool:
    """True only for a bare greeting with no other content — a message like
    'hi, rainfall is low' should still go through full field extraction."""
    raw = message.strip()
    if not raw:
        return False
    if bool(_GREETING_RE.match(raw)):
        return True
    cleaned = re.sub(r"[!.,?]+$", "", raw.lower()).strip()
    words = cleaned.split()
    if len(words) == 1:
        matches = difflib.get_close_matches(cleaned, CANONICAL_GREETINGS, n=1, cutoff=0.75)
        if matches:
            return True
    elif len(words) <= 3:
        matches = difflib.get_close_matches(cleaned, [g for g in CANONICAL_GREETINGS if " " in g], n=1, cutoff=0.8)
        if matches:
            return True
    return False


# ---------------------------------------------------------------------------
# Field extraction patterns
# ---------------------------------------------------------------------------

_SOC_RE = re.compile(r"(?:soil organic carbon|soc)[^\d]{0,15}(\d+(?:\.\d+)?)\s*%?", re.I)
_SOC_RE_BARE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:soil organic carbon|soc)?", re.I)

_RAINFALL_WORD_RE = re.compile(r"\b(low|medium|high)\b.{0,15}rainfall|rainfall.{0,15}\b(low|medium|high)\b", re.I)
# Descriptive rainfall vocabulary -> bucket. Mirrors common agronomy usage;
# not a cited standard, just sensible defaults.
_RAINFALL_DESCRIPTORS = {
    "scarce": "low", "scanty": "low", "arid": "low", "dry": "low", "sparse": "low",
    "moderate": "medium", "average": "medium", "normal": "medium",
    "heavy": "high", "abundant": "high", "wet": "high", "plentiful": "high",
}
_RAINFALL_DESCRIPTOR_RE = re.compile(
    r"\b(" + "|".join(_RAINFALL_DESCRIPTORS.keys()) + r")\b", re.I
)
# Numeric + unit, optionally with "/year" or "/yr" or "annual"
_RAINFALL_NUM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mm|cm|in(?:ch(?:es)?)?)\b(?:\s*/\s*(?:yr|year))?", re.I
)
# Illustrative annual-rainfall thresholds in millimetres. Engineering defaults
# chosen for demo purposes, not a cited scientific standard — flagged here so
# they're easy to find and tune, and honest if asked about their provenance.
_RAINFALL_MM_LOW_MAX = 450
_RAINFALL_MM_MEDIUM_MAX = 1000

_REGION_VOCAB = [
    "semi-arid", "arid", "tropical", "temperate", "subtropical", "coastal", "alpine",
]
_LAND_USE_VOCAB = [
    "monoculture", "agroforestry", "polyculture", "intercropping",
    "grassland", "cropland", "pasture", "orchard", "rangeland",
]
_CROPS = [
    "wheat", "rice", "maize", "corn", "millet", "soy", "cotton",
    "barley", "sugarcane", "vegetables", "vegetable",
]
_CROPS_RE = r"(?:" + "|".join(_CROPS) + r")"
_LAND_USE_PATTERNS = [
    re.compile(rf"\b(monoculture\s+{_CROPS_RE}|polyculture\s+{_CROPS_RE}|intercropping\s+{_CROPS_RE})\b", re.I),
    re.compile(r"\b(monoculture|agroforestry|polyculture|intercropping|grassland|cropland|pasture|orchard|rangeland)\b", re.I),
    re.compile(rf"\b({_CROPS_RE})\b", re.I),
]

_STRAY_WORDS = {"fro", "for", "a", "an", "is", "was", "are", "our", "the", "of", "it", "its"}
_FILLER_PREFIXES = re.compile(
    r"^(it'?s|its|we (use|have)|used for|used|for|fro|land use is|region is|currently)\s+",
    re.I,
)


def _mm_from_match(match: "re.Match") -> float:
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("cm"):
        return value * 10
    if unit.startswith("in"):
        return value * 25.4
    return value  # mm


def _bucket_rainfall_mm(mm: float) -> str:
    if mm < _RAINFALL_MM_LOW_MAX:
        return "low"
    if mm < _RAINFALL_MM_MEDIUM_MAX:
        return "medium"
    return "high"


def _extract_rainfall(message: str, require_keyword: bool) -> str | None:
    """require_keyword=True is used for the general (non-pending) scan so a
    stray number elsewhere in a sentence isn't misread as rainfall; when this
    field is the one pending, the caller passes require_keyword=False so a
    bare '300 mm' or 'moderate' resolves on its own."""
    word_match = _RAINFALL_WORD_RE.search(message)
    if word_match:
        val = (word_match.group(1) or word_match.group(2) or "").lower()
        if val in ("low", "medium", "high"):
            return val

    if require_keyword and not re.search(r"\brain(fall)?\b", message, re.I):
        return None

    num_match = _RAINFALL_NUM_RE.search(message)
    if num_match:
        return _bucket_rainfall_mm(_mm_from_match(num_match))

    desc_match = _RAINFALL_DESCRIPTOR_RE.search(message)
    if desc_match:
        return _RAINFALL_DESCRIPTORS[desc_match.group(1).lower()]

    if not require_keyword:
        for word in ("low", "medium", "high"):
            if re.search(rf"\b{word}\b", message, re.I):
                return word

    return None


def _fuzzy_match_vocab(token: str, vocab: list[str], cutoff: float = 0.75) -> str | None:
    close = difflib.get_close_matches(token.lower(), vocab, n=1, cutoff=cutoff)
    return close[0] if close else None


def _clean_free_text(text: str) -> str:
    cleaned = _FILLER_PREFIXES.sub("", text).strip(" .,")
    cleaned = " ".join(w for w in cleaned.split() if w.lower() not in _STRAY_WORDS)
    return cleaned


def rule_based_parse(message: str) -> SiteInput:
    """General-purpose scan across a free-form message. Used first; anything
    it misses for a single pending field is retried by _parse_expected_field."""
    data = SiteInput()

    soc_match = _SOC_RE.search(message) or _SOC_RE_BARE.search(message)
    if soc_match:
        try:
            data.soil_organic_carbon_pct = float(soc_match.group(1))
        except (ValueError, IndexError):
            pass

    rainfall = _extract_rainfall(message, require_keyword=True)
    if rainfall:
        data.rainfall = rainfall

    lowered = message.lower()
    for region_word in _REGION_VOCAB:
        if re.search(rf"\b{re.escape(region_word)}\b", lowered):
            data.region = region_word
            break
    else:
        # tolerate spacing/typo variants like "semi arid", "semiarid", "temprate"
        for token in re.findall(r"[a-z\-]+", lowered):
            match = _fuzzy_match_vocab(token.replace("-", ""), [v.replace("-", "") for v in _REGION_VOCAB], cutoff=0.8)
            if match:
                data.region = next(v for v in _REGION_VOCAB if v.replace("-", "") == match)
                break

    for pattern in _LAND_USE_PATTERNS:
        match = pattern.search(message)
        if match:
            data.land_use = match.group(1).lower()
            break

    if "high pollution" in lowered or "polluted" in lowered:
        data.pollution_level = "high"
    elif "low pollution" in lowered:
        data.pollution_level = "low"

    if "deforestation" in lowered and ("increasing" in lowered or "rising" in lowered):
        data.deforestation_trend = "increasing"

    return data


def _parse_expected_field(message: str, field: str) -> object | None:
    """Context-aware fallback: interpret the raw reply as an answer to the
    ONE field currently pending, tolerating typos/units/bare values that the
    general scanner above requires a keyword for."""
    text = message.strip()

    if field == "soil_organic_carbon_pct":
        m = re.search(r"(?<![a-zA-Z0-9])(\d+(?:\.\d+)?)\s*%?(?![a-zA-Z0-9])", text)
        return float(m.group(1)) if m else None

    if field == "rainfall":
        return _extract_rainfall(text, require_keyword=False)

    if field == "region":
        cleaned = _clean_free_text(text)
        if not cleaned:
            return None
        for token in re.findall(r"[a-z\-]+", cleaned.lower()):
            match = _fuzzy_match_vocab(
                token.replace("-", ""), [v.replace("-", "") for v in _REGION_VOCAB], cutoff=0.75
            )
            if match:
                return next(v for v in _REGION_VOCAB if v.replace("-", "") == match)
        return cleaned.lower()  # accept free text rather than looping forever

    if field == "land_use":
        cleaned = _clean_free_text(text)
        if not cleaned:
            return None
        for token in cleaned.lower().split():
            close = _fuzzy_match_vocab(token, _LAND_USE_VOCAB, cutoff=0.75)
            if close:
                return cleaned.lower().replace(token, close)
        return cleaned.lower()

    return None


# ---------------------------------------------------------------------------
# Public parse entry point
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    site_input: SiteInput
    newly_filled: list[str]      # fields this turn actually added
    understood: bool             # False => could not fill the pending field at all


SITE_FIELDS = [
    "soil_organic_carbon_pct",
    "soil_ph",
    "soil_moisture",
    "rainfall",
    "land_use",
    "region",
    "latitude",
    "longitude",
    "pollution_level",
    "deforestation_trend",
]


def parse_message(message: str, existing: SiteInput) -> ParseResult:
    llm = get_llm()
    parsed = _llm_parse(message, llm) if llm is not None else None
    if parsed is None:
        parsed = rule_based_parse(message)

    # If the user provided a full set of required fields, treat as a fresh scenario
    if not parsed.missing_required():
        merged = parsed.model_copy()
        newly_filled = [f for f in SITE_FIELDS if getattr(parsed, f) is not None]
        return ParseResult(site_input=merged, newly_filled=newly_filled, understood=True)

    merged = existing.model_copy()
    newly_filled: list[str] = []
    for field in SITE_FIELDS:
        value = getattr(parsed, field, None)
        if value is not None:
            setattr(merged, field, value)
            newly_filled.append(field)

    pending_before = existing.missing_required()
    understood = True

    if pending_before and not newly_filled:
        target_field = pending_before[0]
        value = _parse_expected_field(message, target_field)
        if value is not None:
            setattr(merged, target_field, value)
            newly_filled.append(target_field)
        elif not is_greeting(message):
            understood = False

    return ParseResult(site_input=merged, newly_filled=newly_filled, understood=understood)


def _llm_parse(message: str, llm) -> SiteInput | None:
    try:
        prompt = (
            "Extract environmental site fields from the user message as strict JSON "
            "with keys soil_organic_carbon_pct (float or null), rainfall "
            "('low'|'medium'|'high' or null), land_use (string or null), "
            "region (string or null), pollution_level ('low'|'medium'|'high' or null), "
            "deforestation_trend ('stable'|'increasing'|'decreasing' or null). "
            "Convert any numeric rainfall (mm/cm/inches) to low/medium/high using "
            "<450mm=low, 450-1000mm=medium, >1000mm=high. Correct obvious typos in "
            "land_use/region to the closest standard term. Only include fields "
            "explicitly stated or clearly implied. Respond with JSON only.\n\n"
            f"Message: {message}"
        )
        result = llm.invoke(prompt)
        import json

        content = result.content if hasattr(result, "content") else str(result)
        content = content.strip().strip("`").replace("json\n", "").strip()
        payload = json.loads(content)
        valid_fields = SiteInput.model_fields.keys()
        return SiteInput(**{k: v for k, v in payload.items() if k in valid_fields})
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Question / acknowledgment phrasing
# ---------------------------------------------------------------------------

def humanize_known_fields(site_input: SiteInput, fields: list[str]) -> str:
    parts = []
    for f in fields:
        value = getattr(site_input, f, None)
        if value is None:
            continue
        label = FIELD_LABELS.get(f, f.replace("_", " ").title())
        if f == "soil_organic_carbon_pct":
            parts.append(f"{label}: {value}%")
        else:
            parts.append(f"{label}: {str(value).title()}")
    return ", ".join(parts)


def next_clarifying_question(site_input: SiteInput, is_first_turn: bool = False) -> str:
    missing = site_input.missing_required()
    if not missing:
        return ""
    field = missing[0]
    question = FIELD_QUESTIONS.get(field, f"Could you provide {field.replace('_', ' ')}?")

    # Context-aware nudge: reference what's already known so it doesn't read
    # like an identical form no matter the conversation so far.
    if field == "rainfall" and site_input.region:
        question = f"Given it's a {site_input.region} region, {question[0].lower()}{question[1:]}"
    elif field == "land_use" and site_input.rainfall:
        question = f"With {site_input.rainfall} rainfall in mind, {question[0].lower()}{question[1:]}"

    return f"{GREETING}{question}" if is_first_turn else question


def clarification_retry_message(field: str) -> str:
    label = FIELD_LABELS.get(field, field.replace("_", " "))
    example = FIELD_HELP_EXAMPLES.get(field, "")
    return f"I couldn't quite catch the {label} from that reply. Could you specify it more directly ({example})?"
