"""
main.py — FastAPI backend for Antinomai.
Endpoints:
  POST /api/chat    — run the LangGraph agent, return SSE stream
  POST /api/upload  — receive chart image for vision analysis
  GET  /api/health  — model status + API key check
  DELETE /api/clear — clear session
"""
import base64
import hashlib
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Startup: resolve models ────────────────────────────────────────────────────
from backend.config import resolve_models
from backend.state import set_models, get_primary_model, get_vision_model

_HEALTH: dict = {"primary_model": None, "vision_model": None, "status": "starting"}
_SESSION_IMAGES: dict = {}   # session_id -> base64 image data


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Checking Groq model availability...")
    try:
        primary, vision = resolve_models()
        set_models(primary, vision)
        _HEALTH["primary_model"] = primary
        _HEALTH["vision_model"]  = vision
        _HEALTH["status"]        = "ok"
        print(f"[Startup] ✅ Ready — primary={primary}, vision={vision}")
    except RuntimeError as e:
        _HEALTH["status"] = f"error: {e}"
        print(f"[Startup] ❌ {e}")
    yield
    print("[Shutdown] Bye.")


app = FastAPI(title="Antinomai API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://*.vercel.app", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:                str
    session_id:             Optional[str] = None
    enable_sentiment_check: bool = False


class HealthResponse(BaseModel):
    status:        str
    primary_model: Optional[str]
    vision_model:  Optional[str]
    groq_key_set:  bool
    tavily_key_set: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status         = _HEALTH["status"],
        primary_model  = _HEALTH["primary_model"],
        vision_model   = _HEALTH["vision_model"],
        groq_key_set   = bool(os.environ.get("GROQ_API_KEY")),
        tavily_key_set = bool(os.environ.get("TAVILY_API_KEY")),
    )


@app.post("/api/upload")
async def upload_chart(session_id: str, file: UploadFile = File(...)):
    """Store a chart image (base64) keyed by session_id for the next chat turn."""
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Only JPG/PNG images are accepted.")
    data = await file.read()
    _SESSION_IMAGES[session_id] = {
        "b64":  base64.b64encode(data).decode(),
        "mime": file.content_type,
    }
    return {"ok": True, "filename": file.filename}


@app.delete("/api/clear")
async def clear_session(session_id: str):
    _SESSION_IMAGES.pop(session_id, None)
    return {"ok": True}


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio using Groq Whisper Large V3."""
    import groq
    try:
        client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
        audio_bytes = await file.read()
        transcription = client.audio.transcriptions.create(
            file=(file.filename or "audio.webm", audio_bytes),
            model="whisper-large-v3",
            response_format="json",
        )
        return {"text": transcription.text}
    except Exception as e:
        print(f"[Transcribe] Error: {e}")
        raise HTTPException(500, str(e))


# ── Research Vault endpoints ───────────────────────────────────────────────────
class VaultIngestTextRequest(BaseModel):
    text:        str
    source_name: str
    ticker:      Optional[str] = None
    doc_type:    str = "other"


@app.post("/api/vault/ingest")
async def vault_ingest(
    ticker:   Optional[str] = None,
    doc_type: str = "other",
    file:     Optional[UploadFile] = File(None),
):
    """
    Ingest a PDF, TXT, or MD file into the Research Vault.
    Returns doc_id and chunk count.
    """
    from backend.vault import ingest_file
    if file is None:
        raise HTTPException(400, "No file provided.")
    allowed = ("application/pdf", "text/plain", "text/markdown")
    if file.content_type not in allowed and not (file.filename or "").endswith((".pdf", ".txt", ".md")):
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use PDF, TXT, or MD.")
    try:
        data   = await file.read()
        doc_id = ingest_file(data=data, filename=file.filename or "document", ticker=ticker or None, doc_type=doc_type)
        return {"ok": True, "doc_id": doc_id, "filename": file.filename}
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        print(f"[Vault] Ingest error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/vault/ingest-text")
async def vault_ingest_text(req: VaultIngestTextRequest):
    """Ingest plain text directly (no file upload) into the Research Vault."""
    from backend.vault import ingest_text
    try:
        doc_id = ingest_text(
            text=req.text,
            source_name=req.source_name,
            ticker=req.ticker,
            doc_type=req.doc_type,
        )
        return {"ok": True, "doc_id": doc_id}
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        print(f"[Vault] Ingest-text error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/vault/docs")
async def vault_list_docs():
    """Return all documents in the Research Vault (one entry per document, not per chunk)."""
    from backend.vault import list_documents
    try:
        docs = list_documents()
        return {
            "docs": [
                {
                    "doc_id":      d.doc_id,
                    "source":      d.source,
                    "ticker":      d.ticker,
                    "doc_type":    d.doc_type,
                    "date_added":  d.date_added,
                    "chunk_count": d.chunk_count,
                }
                for d in docs
            ],
            "total_docs": len(docs),
        }
    except Exception as e:
        print(f"[Vault] List error: {e}")
        raise HTTPException(500, str(e))


@app.delete("/api/vault/docs/{doc_id}")
async def vault_delete_doc(doc_id: str):
    """Delete all chunks belonging to a document from the Research Vault."""
    from backend.vault import delete_document
    try:
        deleted = delete_document(doc_id)
        if not deleted:
            raise HTTPException(404, f"Document '{doc_id}' not found in vault.")
        return {"ok": True, "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Vault] Delete error: {e}")
        raise HTTPException(500, str(e))


class VaultSearchRequest(BaseModel):
    query:         str
    ticker_filter: Optional[str] = None
    n_results:     int = 5


@app.post("/api/vault/search")
async def vault_search(req: VaultSearchRequest):
    """Semantic search over the vault — useful for UI preview and debugging."""
    from backend.vault import retrieve
    try:
        chunks = retrieve(query=req.query, ticker_filter=req.ticker_filter, n_results=req.n_results)
        return {
            "results": [
                {
                    "content":    c.content,
                    "source":     c.source,
                    "ticker":     c.ticker,
                    "doc_type":   c.doc_type,
                    "date_added": c.date_added,
                    "doc_id":     c.doc_id,
                }
                for c in chunks
            ],
            "count": len(chunks),
        }
    except Exception as e:
        print(f"[Vault] Search error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Run the LangGraph agent and stream the response via SSE.
    Each SSE event is a JSON line:
      {"type": "token",   "content": "..."}
      {"type": "done",    "route": "trivia", "portfolio": null}
      {"type": "error",   "content": "..."}
    """
    if _HEALTH["status"] != "ok":
        raise HTTPException(503, f"Service not ready: {_HEALTH['status']}")

    from backend.agents import get_app
    from langchain_core.messages import HumanMessage

    session_id = req.session_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:10]
    img_data   = _SESSION_IMAGES.pop(session_id, None)

    # Build LangGraph message
    if img_data:
        lc_message = HumanMessage(content=[
            {"type": "text",      "text": req.message},
            {"type": "image_url", "image_url": {"url": f"data:{img_data['mime']};base64,{img_data['b64']}"}},
        ])
    else:
        lc_message = HumanMessage(content=req.message)

    initial_state = {
        "messages":               [lc_message],
        "next_destination":       "",
        "user_input_type":        "",
        "portfolio_report":       "",
        "enable_sentiment_check": req.enable_sentiment_check,
    }

    thread_id = f"ant_{session_id}_{uuid.uuid4().hex[:8]}"
    config    = {"configurable": {"thread_id": thread_id}}

    async def event_stream():
        import json
        try:
            _app   = get_app()
            result = _app.invoke(initial_state, config=config)

            route          = result.get("next_destination", "trivia")
            final_content  = result["messages"][-1].content
            portfolio_data = None

            if route == "portfolio" and "portfolio_rows" in result:
                portfolio_data = {
                    "rows":  result.get("portfolio_rows", []),
                    "total": result.get("portfolio_total", 0),
                    "beta":  result.get("portfolio_beta"),
                }

            # Stream the final content token-by-token (simulate streaming)
            # For true streaming, use LangGraph's astream_events in a future iteration
            chunk_size = 4
            for i in range(0, len(final_content), chunk_size):
                chunk = final_content[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                # tiny yield to keep SSE alive — no real sleep needed for token sim

            yield f"data: {json.dumps({'type': 'done', 'route': route, 'portfolio': portfolio_data})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            import json
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

