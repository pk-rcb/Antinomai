"""
state.py — Shared application state & model registry.
Holds the resolved primary/vision model names after startup health check.
"""
from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

# ── Runtime model registry ────────────────────────────────────────────────────
_PRIMARY_MODEL: str = "llama-3.3-70b-versatile"
_VISION_MODEL:  str = "meta-llama/llama-4-scout-17b-16e-instruct"


def set_models(primary: str, vision: str) -> None:
    global _PRIMARY_MODEL, _VISION_MODEL
    _PRIMARY_MODEL = primary
    _VISION_MODEL  = vision


def get_primary_model() -> str:
    return _PRIMARY_MODEL


def get_vision_model() -> str:
    return _VISION_MODEL


# ── LangGraph state schemas ────────────────────────────────────────────────────
class ApplicationState(TypedDict):
    messages:              List[BaseMessage]
    next_destination:      str
    user_input_type:       str
    portfolio_report:      str
    enable_sentiment_check: bool
    session_id:            str


class IntentRoute(BaseModel):
    destination: str = Field(
        description="Must be exactly one of: 'debate', 'vision', 'trivia', 'fundamental', 'portfolio', 'research'"
    )


class Asset(BaseModel):
    ticker: str   = Field(description="Official ticker symbol. Append .NS for Indian stocks.")
    shares: float = Field(description="Exact number of shares owned.")


class PortfolioExtraction(BaseModel):
    assets: List[Asset] = Field(description="All extracted assets and share counts.")

