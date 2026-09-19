"""
Optional LLM access via LangChain + Gemini.

Every function here has a non-LLM fallback so the system is fully
functional (parsing, clarifying questions, phrasing) with zero API key,
per the constraint that this must not be a generic LLM-only solution.
Set GOOGLE_API_KEY in .env to enable natural-language polish on top.
"""

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def get_llm():
    if not settings.google_api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.google_api_key,
            temperature=0.2,
        )
    except Exception:
        return None


def llm_available() -> bool:
    return get_llm() is not None
