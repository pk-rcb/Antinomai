"""
config.py — API key loading + Groq model health check with auto-fallback.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Always load from backend/.env regardless of working directory
_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE)

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    raise RuntimeError(
        "Missing API keys. Create backend/.env with:\n"
        "  GROQ_API_KEY=gsk_...\n"
        "  TAVILY_API_KEY=tvly-..."
    )

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# -------------------------------------------------------------------
# Model fallback chain — ordered by preference.
# The health check at startup picks the first one that responds OK.
# -------------------------------------------------------------------
_MODEL_CHAIN_PRIMARY = [
    "openai/gpt-oss-120b",       # OpenAI OSS flagship on Groq (current)
    "openai/gpt-oss-20b",        # Smaller/faster OpenAI OSS on Groq
    "qwen/qwen3.6-27b",          # Qwen 3.6 27B — good general fallback
    "llama-3.1-8b-instant",      # Llama 3.1 8B instant (check availability)
]

_MODEL_CHAIN_VISION = [
    "qwen/qwen3.6-27b",          # Preferred vision model
    "openai/gpt-oss-120b",       # Fallback multimodal
    "openai/gpt-oss-20b",        # Smaller vision-capable fallback
    "llama-3.1-8b-instant",      # Last-resort text-only fallback
]


def _probe_model(model_name: str) -> bool:
    """Send a tiny test prompt. Returns True if the model is reachable."""
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        llm = ChatGroq(model=model_name, temperature=0.0, max_retries=1)
        llm.invoke([HumanMessage(content="ping")])
        return True
    except Exception as e:
        print(f"[Config] Model '{model_name}' unavailable: {e}")
        return False


def resolve_models() -> tuple[str, str]:
    """
    Returns (primary_model, vision_model) — first working ones in the chain.
    Call once at startup.
    """
    primary = None
    for m in _MODEL_CHAIN_PRIMARY:
        print(f"[Config] Checking primary model: {m}")
        if _probe_model(m):
            primary = m
            print(f"[Config] ✅ Primary model selected: {m}")
            break

    vision = None
    for m in _MODEL_CHAIN_VISION:
        print(f"[Config] Checking vision model: {m}")
        if _probe_model(m):
            vision = m
            print(f"[Config] ✅ Vision model selected: {m}")
            break

    if not primary:
        raise RuntimeError("All Groq primary models are unavailable. Check your GROQ_API_KEY.")
    if not vision:
        print("[Config] ⚠️  No vision model available — vision analysis will be disabled.")
        vision = primary  # graceful degrade: use text model, it will refuse image input

    return primary, vision

