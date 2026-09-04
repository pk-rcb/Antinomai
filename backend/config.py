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
    Returns (primary_model, vision_model).
    Hardcoded to avoid sequential API probing that causes a 5-10s startup delay.
    """
    primary = "openai/gpt-oss-120b"
    vision = "qwen/qwen3.6-27b"
    
    print(f"[Config] ✅ Primary model selected: {primary}")
    print(f"[Config] ✅ Vision model selected: {vision}")
    
    return primary, vision

