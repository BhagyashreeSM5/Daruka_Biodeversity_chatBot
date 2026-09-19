import re
import difflib
from dataclasses import dataclass

from app.models import SiteInput, SITE_FIELDS
from app.llm import get_llm

FIELD_LABELS = {
    "soil_organic_carbon_pct": "Soil Organic Carbon",
    "rainfall": "Rainfall",
    "temperature": "Temperature",
    "land_use": "Land Use",
    "region": "Region",
    "soil_ph": "Soil pH",
    "soil_moisture": "Soil Moisture",
    "pollution_level": "Pollution Level",
    "deforestation_trend": "Deforestation Trend",
    "human_impact": "Human Impact",
    "biodiversity_status": "Biodiversity Status",
    "species_richness": "Species Richness",
    "habitat_diversity": "Habitat Diversity",
}

# Correction intent patterns (spec §7)
_CORRECTION_RE = re.compile(
    r"\b(actually|correction|no[,.]?\s*(it'?s|the)|not\s+\w+[,.]?\s*(?:it'?s|but)|wrong|correct\s+that|i\s+meant)\b",
    re.I,
)

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


FILLER_WORDS = {
    "thanks",
    "thank you",
    "ok",
    "okay",
    "got it",
    "cool",
    "great",
    "perfect",
    "sounds good",
    "nice",
    "awesome",
    "thx",
    "thnk u",
    "k",
    "kk",
    "sure",
    "alright",
    "understood",
    "gotcha",
    "yep",
    "yeah",
    "yes",
    "no",
    "nah",
    "nope",
}


def is_conversational_filler(message: str) -> bool:
    raw = message.strip()
    if not raw:
        return True
    cleaned = re.sub(r"[!.,?]+$", "", raw.lower()).strip()
    if cleaned in FILLER_WORDS:
        return True
    cleaned_words = [w.strip("!.,?").lower() for w in raw.split()]
    if cleaned_words and cleaned_words[0] in FILLER_WORDS:
        if len(cleaned_words) <= 6:
            return True
    return False


QUESTION_INDICATORS = [
    "how",
    "what",
    "why",
    "can",
    "could",
    "should",
    "would",
    "tell",
    "explain",
    "recommend",
    "suggest",
    "ways",
    "methods",
    "practices",
    "improve",
    "increase",
    "reduce",
    "support",
    "help",
    "give",
    "provide",
    "analyze",
    "assessment",
]


def is_user_question(message: str) -> bool:
    raw = message.strip()
    if not raw:
        return False
    if is_conversational_filler(raw) or is_greeting(raw):
        return False
    if "?" in raw:
        return True
    lowered = raw.lower()
    if any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in QUESTION_INDICATORS):
        return True
    words = lowered.split()
    if len(words) >= 5:
        return True
    return False


# ---------------------------------------------------------------------------
# Field extraction patterns
# ---------------------------------------------------------------------------

_SOC_RE = re.compile(r"(?:soil organic carbon|soc)[^\d]{0,15}(\d+(?:\.\d+)?)\s*%?", re.I)
_SOC_RE_BARE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:soil organic carbon|soc)?", re.I)

_RAINFALL_WORD_RE = re.compile(r"\b(low|medium|high)\b.{0,15}rainfall|rainfall.{0,15}\b(low|medium|high)\b", re.I)

_PH_RE = re.compile(r"\b(?:soil\s*)?ph\b[^\d]{0,10}(\d+(?:\.\d+)?)", re.I)
_MOISTURE_RE = re.compile(r"\b(?:soil\s*)?moisture\b[^\w]{0,10}(low|medium|high|moderate)", re.I)

_POLLUTION_HIGH_RE = re.compile(r"\b(high\s+pollution|polluted|heavy\s+pollution|chemical\s+runoff)\b", re.I)
_POLLUTION_LOW_RE = re.compile(r"\b(low\s+pollution|unpolluted|clean\s+water)\b", re.I)
_POLLUTION_MED_RE = re.compile(r"\b(medium\s+pollution|moderate\s+pollution)\b", re.I)

_DEFORESTATION_INC_RE = re.compile(r"\b(edge-deforestation|deforestation|forest\s+loss|tree\s+clearing|deforested)\b", re.I)
_DEFORESTATION_DEC_RE = re.compile(r"\b(reforestation|afforestation|decreasing\s+deforestation|reduced\s+deforestation)\b", re.I)

# Human impact patterns (spec §5) — separate from pollution
_HUMAN_IMPACT_RE = re.compile(
    r"\b("
    r"overgrazing|over-grazing|over\s+grazing|"
    r"(?:moderate|heavy|light|low|high)?\s*pesticide\s*(?:use|application|exposure|drift)?|"
    r"agricultural\s+runoff|nutrient\s+runoff|chemical\s+(?:farming|runoff)|"
    r"road\s+construction|road\s+building|"
    r"mining|quarrying|"
    r"logging|illegal\s+logging|"
    r"urbanization|urban\s+sprawl|"
    r"(?:habitat|land)\s+(?:clearing|conversion)|"
    r"drainage|wetland\s+drainage|"
    r"fire|burning|slash.and.burn"
    r")\b",
    re.I,
)

# Biodiversity status patterns (spec §4)
_BIODIVERSITY_HIGH_RE = re.compile(
    r"\b("
    r"high\s+(?:species\s+)?(?:richness|diversity|biodiversity)|"
    r"species[- ]rich|biodiverse|"
    r"continuous\s+habitat|intact\s+habitat|"
    r"healthy\s+ecosystem|pristine"
    r")\b",
    re.I,
)
_BIODIVERSITY_LOW_RE = re.compile(
    r"\b("
    r"low\s+(?:species\s+)?(?:richness|diversity|biodiversity)|"
    r"low\s+pollinator\s+(?:diversity|populations?)|"
    r"declining\s+(?:native\s+)?(?:grasses|species|populations?|biodiversity)|"
    r"degraded\s+(?:habitat|ecosystem)|"
    r"fragmented\s+(?:habitat|corridors?|wildlife)"
    r")\b",
    re.I,
)

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
_RAINFALL_NUM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mm|cm|in(?:ch(?:es)?)?)\b(?:\s*/\s*(?:yr|year))?", re.I
)
_RAINFALL_MM_LOW_MAX = 450
_RAINFALL_MM_MEDIUM_MAX = 1000

_REGION_VOCAB = [
    "semi-arid", "arid", "tropical", "temperate", "subtropical", "coastal", "alpine",
    "mediterranean", "boreal", "montane", "savanna",
]
_LAND_USE_VOCAB = [
    "monoculture", "agroforestry", "polyculture", "intercropping",
    "grassland", "cropland", "pasture", "orchard", "rangeland",
    "forest", "mixed native forest", "woodland", "vineyard",
    "plantation", "silvopasture", "forest edge",
]
_CROPS = [
    "wheat", "rice", "maize", "corn", "millet", "soy", "cotton",
    "barley", "sugarcane", "vegetables", "vegetable",
]
_CROPS_RE = r"(?:" + "|".join(_CROPS) + r")"
_LAND_USE_PATTERNS = [
    re.compile(rf"\b(mixed\s+native\s+forest)\b", re.I),
    re.compile(rf"\b(forest\s+edge)\b", re.I),
    re.compile(rf"\b(monoculture\s+{_CROPS_RE}|polyculture\s+{_CROPS_RE}|intercropping\s+{_CROPS_RE})\b", re.I),
    re.compile(r"\b(monoculture|agroforestry|polyculture|intercropping|grassland|cropland|pasture|orchard|rangeland|vineyard|plantation|silvopasture|woodland)\b", re.I),
    re.compile(rf"\b({_CROPS_RE})\b", re.I),
]

_STRAY_WORDS = {"fro", "for", "a", "an", "is", "was", "are", "our", "the", "of", "it", "its"}
_FILLER_PREFIXES = re.compile(
    r"^(it'?s|its|we (use|have)|used for|used|for|fro|land use is|region is|currently)\s+",
    re.I,
)

# --- New-site detection phrases (spec §2A) ---
_NEW_SITE_PHRASES = re.compile(
    r"\b(new\s+site|another\s+site|different\s+(?:region|site|location|area)|"
    r"consider\s+this\s+(?:scenario|site|case)|new\s+case|separate\s+site|"
    r"next\s+site|second\s+site|alternative\s+site|"
    r"analyze\s+this|evaluate\s+this|assess\s+this)\b",
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
    origin: dict[str, str] = {}

    soc_match = _SOC_RE.search(message) or _SOC_RE_BARE.search(message)
    if soc_match:
        try:
            data.soil_organic_carbon_pct = float(soc_match.group(1))
            origin["soil_organic_carbon_pct"] = "KNOWN"
        except (ValueError, IndexError):
            pass

    rainfall = _extract_rainfall(message, require_keyword=True)
    if rainfall:
        data.rainfall = rainfall
        origin["rainfall"] = "KNOWN"

    lowered = message.lower()

    # Region extraction
    for region_word in _REGION_VOCAB:
        if re.search(rf"\b{re.escape(region_word)}\b", lowered):
            data.region = region_word
            origin["region"] = "KNOWN"
            break
    else:
        for token in re.findall(r"[a-z\-]+", lowered):
            match = _fuzzy_match_vocab(token.replace("-", ""), [v.replace("-", "") for v in _REGION_VOCAB], cutoff=0.8)
            if match:
                data.region = next(v for v in _REGION_VOCAB if v.replace("-", "") == match)
                origin["region"] = "KNOWN"
                break

    # Land use — try multi-word patterns first (e.g. "mixed native forest", "forest edge")
    for pattern in _LAND_USE_PATTERNS:
        match = pattern.search(message)
        if match:
            data.land_use = match.group(1).lower()
            origin["land_use"] = "KNOWN"
            break

    ph_match = _PH_RE.search(message)
    if ph_match:
        try:
            data.soil_ph = float(ph_match.group(1))
            origin["soil_ph"] = "KNOWN"
        except (ValueError, IndexError):
            pass

    moisture_match = _MOISTURE_RE.search(message)
    if moisture_match:
        m_val = moisture_match.group(1).lower()
        data.soil_moisture = "medium" if m_val == "moderate" else m_val
        origin["soil_moisture"] = "KNOWN"

    # Pollution — only from explicit pollution keywords, NOT from human_impact
    if _POLLUTION_HIGH_RE.search(lowered):
        data.pollution_level = "high"
        origin["pollution_level"] = "KNOWN"
    elif _POLLUTION_MED_RE.search(lowered):
        data.pollution_level = "medium"
        origin["pollution_level"] = "KNOWN"
    elif _POLLUTION_LOW_RE.search(lowered):
        data.pollution_level = "low"
        origin["pollution_level"] = "KNOWN"

    # Deforestation
    if _DEFORESTATION_DEC_RE.search(lowered):
        data.deforestation_trend = "decreasing"
        origin["deforestation_trend"] = "KNOWN"
    elif _DEFORESTATION_INC_RE.search(lowered):
        if not any(w in lowered for w in ["stable", "none", "no deforestation", "zero deforestation"]):
            data.deforestation_trend = "increasing"
            origin["deforestation_trend"] = "KNOWN"

    # Human impact (spec §5) — extracted as free-text, NOT auto-mapped to pollution
    impact_match = _HUMAN_IMPACT_RE.search(message)
    if impact_match:
        data.human_impact = impact_match.group(0).strip().lower()
        origin["human_impact"] = "KNOWN"

    # Biodiversity status
    bio_high = _BIODIVERSITY_HIGH_RE.search(message)
    bio_low = _BIODIVERSITY_LOW_RE.search(message)
    if bio_high and bio_low:
        # Both detected — capture the full description
        data.biodiversity_status = f"{bio_high.group(0).strip().lower()}, {bio_low.group(0).strip().lower()}"
        origin["biodiversity_status"] = "KNOWN"
    elif bio_low:
        data.biodiversity_status = bio_low.group(0).strip().lower()
        origin["biodiversity_status"] = "KNOWN"
    elif bio_high:
        data.biodiversity_status = bio_high.group(0).strip().lower()
        origin["biodiversity_status"] = "KNOWN"

    for f in SITE_FIELDS:
        if f in origin:
            data.field_origin[f] = origin[f]
        elif f not in data.field_origin:
            data.field_origin[f] = "UNKNOWN"
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
# New-site detection (spec §2A, §3)
# ---------------------------------------------------------------------------

def _is_new_site_message(parsed: SiteInput, existing: SiteInput, message: str) -> bool:
    """Determine whether the user's message represents a brand-new site scenario
    rather than an incremental update to the existing site.

    Uses multiple signals — not just field count — per spec §2A."""

    # Signal 1: Explicit new-site phrases
    if _NEW_SITE_PHRASES.search(message):
        return True

    # Count how many *core* site fields the new message fills
    core_fields = ["soil_organic_carbon_pct", "rainfall", "land_use", "region"]
    new_core_count = sum(1 for f in core_fields if getattr(parsed, f, None) is not None)

    # Count ALL site fields filled in this message
    all_new_fields = [f for f in SITE_FIELDS if getattr(parsed, f, None) is not None]
    new_total = len(all_new_fields)

    # Signal 2: If user supplies all 4 required fields, it's a new site
    if new_core_count >= 4:
        return True

    # Signal 3: 3+ core fields AND at least one differs from existing
    if new_core_count >= 3:
        changed = 0
        for f in core_fields:
            new_val = getattr(parsed, f, None)
            old_val = getattr(existing, f, None)
            if new_val is not None and old_val is not None and new_val != old_val:
                changed += 1
        if changed >= 2:
            return True

    # Signal 4: Incompatible ecosystem — region changed AND land_use changed
    new_region = getattr(parsed, "region", None)
    new_land_use = getattr(parsed, "land_use", None)
    old_region = getattr(existing, "region", None)
    old_land_use = getattr(existing, "land_use", None)
    if (new_region and old_region and new_region != old_region and
            new_land_use and old_land_use and new_land_use != old_land_use):
        return True

    # Signal 5: High total field count (5+) with substantial changes
    if new_total >= 5:
        return True

    # Signal 6: Assessment was already completed and user provides 3+ new fields
    if existing.assessment_completed and new_total >= 3:
        return True

    return False


# ---------------------------------------------------------------------------
# Public parse entry point
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    site_input: SiteInput
    newly_filled: list[str]      # fields this turn actually added
    understood: bool             # False => could not fill the pending field at all
    is_correction: bool = False  # True when user is correcting a previously stated value


def parse_message(message: str, existing: SiteInput) -> ParseResult:
    llm = get_llm()
    parsed = _llm_parse(message, llm) if llm is not None else None
    if parsed is None:
        parsed = rule_based_parse(message)

    # --- Correction detection (spec §7) ---
    is_correction = bool(_CORRECTION_RE.search(message))

    # --- New-site detection (spec §1, §3) ---
    if not is_correction and _is_new_site_message(parsed, existing, message):
        # Fresh site: start from parsed data, don't merge with old state
        fresh = parsed.model_copy()
        fresh.assessment_completed = False
        fresh.intro_shown = existing.intro_shown  # preserve UI state only
        fresh.user_values = {}  # clean slate
        newly_filled = [f for f in SITE_FIELDS if getattr(parsed, f) is not None]
        # Ensure all origins are KNOWN for explicitly provided fields, UNKNOWN for rest
        for f in SITE_FIELDS:
            if getattr(fresh, f, None) is not None:
                fresh.set_known(f)
            else:
                fresh.field_origin[f] = "UNKNOWN"
        return ParseResult(site_input=fresh, newly_filled=newly_filled, understood=True)

    # If the user provided all required fields (even without new-site signal), treat as complete
    if not parsed.missing_required():
        merged = parsed.model_copy()
        merged.user_values = {}  # track fresh values
        newly_filled = [f for f in SITE_FIELDS if getattr(parsed, f) is not None]
        for f in newly_filled:
            merged.set_known(f)
        return ParseResult(site_input=merged, newly_filled=newly_filled, understood=True, is_correction=is_correction)

    # --- Incremental merge onto existing site ---
    merged = existing.model_copy()
    newly_filled: list[str] = []
    for field in SITE_FIELDS:
        value = getattr(parsed, field, None)
        if value is not None:
            setattr(merged, field, value)
            merged.set_known(field)
            newly_filled.append(field)

    pending_before = existing.missing_required()
    understood = True

    if pending_before and not newly_filled:
        target_field = pending_before[0]
        value = _parse_expected_field(message, target_field)
        if value is not None:
            setattr(merged, target_field, value)
            merged.set_known(target_field)
            newly_filled.append(target_field)
        elif not is_greeting(message):
            understood = False

    # Final user-value integrity check (spec §5)
    merged.validate_user_values()

    return ParseResult(site_input=merged, newly_filled=newly_filled, understood=understood, is_correction=is_correction)


def _llm_parse(message: str, llm) -> SiteInput | None:
    try:
        prompt = (
            "Extract environmental site fields from the user message as strict JSON "
            "with keys soil_organic_carbon_pct (float or null), rainfall "
            "('low'|'medium'|'high' or null), land_use (string or null), "
            "region (string or null), pollution_level ('low'|'medium'|'high' or null), "
            "deforestation_trend ('stable'|'increasing'|'decreasing' or null), "
            "soil_ph (float or null), soil_moisture ('low'|'medium'|'high' or null), "
            "human_impact (string or null), biodiversity_status (string or null). "
            "Convert any numeric rainfall (mm/cm/inches) to low/medium/high using "
            "<450mm=low, 450-1000mm=medium, >1000mm=high. Correct obvious typos in "
            "land_use/region to the closest standard term. Only include fields "
            "explicitly stated or clearly implied. Do NOT invent values for fields "
            "the user did not mention. Respond with JSON only.\n\n"
            f"Message: {message}"
        )
        result = llm.invoke(prompt)
        import json

        content = result.content if hasattr(result, "content") else str(result)
        content = content.strip().strip("`").replace("json\n", "").strip()
        payload = json.loads(content)
        valid_fields = SiteInput.model_fields.keys()
        site = SiteInput(**{k: v for k, v in payload.items() if k in valid_fields})
        # Mark all LLM-extracted fields as KNOWN
        for k, v in payload.items():
            if k in valid_fields and v is not None:
                site.set_known(k)
        return site
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
    """Adaptive clarification (spec §8): ask for the information most useful
    for the CURRENT reasoning problem, not always the same fixed order."""
    from app.relationships import infer_primary_metrics
    missing = site_input.missing_required()
    if not missing:
        return ""

    # --- Adaptive field priority based on current context ---
    # Determine what pressures exist from what we know so far
    known_pressures = infer_primary_metrics(site_input)
    hi = (site_input.human_impact or "").lower()
    bio = (site_input.biodiversity_status or "").lower()

    # Re-prioritize missing fields based on likely usefulness
    priority_order = list(missing)
    if any(p in known_pressures for p in ["soil_organic_carbon", "grazing_pressure"]):
        # Soil degradation context: soil/climate info most useful
        preferred = ["soil_organic_carbon_pct", "rainfall", "region", "land_use"]
        priority_order = [f for f in preferred if f in missing] + [f for f in missing if f not in preferred]
    elif any(p in known_pressures for p in ["pesticide_exposure"]):
        # Pollinator context: land use and region most useful
        preferred = ["land_use", "region", "rainfall", "soil_organic_carbon_pct"]
        priority_order = [f for f in preferred if f in missing] + [f for f in missing if f not in preferred]
    elif any(p in known_pressures for p in ["habitat_fragmentation", "habitat_loss", "deforestation_trend"]):
        # Habitat context: land use and region critical
        preferred = ["land_use", "region", "soil_organic_carbon_pct", "rainfall"]
        priority_order = [f for f in preferred if f in missing] + [f for f in missing if f not in preferred]
    elif "pollinator" in hi or "pollinator" in bio:
        preferred = ["land_use", "region", "rainfall", "soil_organic_carbon_pct"]
        priority_order = [f for f in preferred if f in missing] + [f for f in missing if f not in preferred]

    field = priority_order[0]
    question = FIELD_QUESTIONS.get(field, f"Could you provide {field.replace('_', ' ')}?")

    # Context-aware framing: reference known data to make the question feel intelligent
    if field == "rainfall" and site_input.region:
        question = f"Given it's a {site_input.region} region, {question[0].lower()}{question[1:]}"
    elif field == "rainfall" and site_input.land_use:
        question = f"For {site_input.land_use} systems, {question[0].lower()}{question[1:]}"
    elif field == "land_use" and site_input.rainfall:
        question = f"With {site_input.rainfall} rainfall in mind, {question[0].lower()}{question[1:]}"
    elif field == "land_use" and site_input.region:
        question = f"In this {site_input.region} environment, {question[0].lower()}{question[1:]}"
    elif field == "region" and site_input.land_use:
        question = f"For this {site_input.land_use} site, {question[0].lower()}{question[1:]}"
    elif field == "soil_organic_carbon_pct" and site_input.land_use:
        question = f"For your {site_input.land_use} system, {question[0].lower()}{question[1:]}"

    return f"{GREETING}{question}" if is_first_turn else question


def clarification_retry_message(field: str) -> str:
    label = FIELD_LABELS.get(field, field.replace("_", " "))
    example = FIELD_HELP_EXAMPLES.get(field, "")
    return f"I couldn't quite catch the {label} from that reply. Could you specify it more directly ({example})?"
