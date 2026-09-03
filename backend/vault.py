"""
vault.py — Research Vault: ChromaDB-backed RAG engine for Antinomai.

Supports local ChromaDB (dev) and Qdrant Cloud (prod) via VECTOR_STORE env var.
  VECTOR_STORE=chromadb  (default) → persists to ./chroma_db/
  VECTOR_STORE=qdrant              → uses QDRANT_URL + QDRANT_API_KEY env vars

Public API
----------
ingest_file(data, filename, ticker, doc_type) -> str   (doc_id)
ingest_text(text, source_name, ticker, doc_type) -> str
retrieve(query, ticker_filter, n_results)       -> list[VaultChunk]
list_documents()                                -> list[DocMeta]
delete_document(doc_id)                         -> bool
vault_doc_count()                               -> int
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Constants ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "finrag_vault"
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 80
# EMBED_MODEL: all-MiniLM-L6-v2 via chromadb DefaultEmbeddingFunction (ONNX, no PyTorch)

VALID_DOC_TYPES = {
    "earnings_call",
    "annual_report",
    "research_note",
    "sec_filing",
    "other",
}


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class VaultChunk:
    content:    str
    source:     str
    ticker:     Optional[str]
    doc_type:   str
    date_added: str
    doc_id:     str


@dataclass
class DocMeta:
    doc_id:      str
    source:      str
    ticker:      Optional[str]
    doc_type:    str
    date_added:  str
    chunk_count: int = 0


# ── Splitter (shared) ──────────────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── ChromaDB backend ───────────────────────────────────────────────────────────
_chroma_collection = None


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        import os

        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            import requests as _requests
            from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

            class GeminiRESTEmbeddingFunction(EmbeddingFunction):
                """Call Google Gemini embedding API directly via REST."""
                _URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"

                def __init__(self, api_key: str):
                    self._api_key = api_key

                def _embed_one(self, text: str) -> list[float]:
                    resp = _requests.post(
                        self._URL,
                        params={"key": self._api_key},
                        json={"model": "models/text-embedding-004",
                              "content": {"parts": [{"text": text}]}},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    return resp.json()["embedding"]["values"]

                def __call__(self, input: Documents) -> Embeddings:
                    return [self._embed_one(t) for t in input]

            embed_fn = GeminiRESTEmbeddingFunction(api_key=gemini_key)
        else:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            embed_fn = DefaultEmbeddingFunction()

        db_path  = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        client   = chromadb.PersistentClient(path=db_path)
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[Vault] ChromaDB collection ready — {_chroma_collection.count()} chunks")
    return _chroma_collection


# ── Qdrant backend (prod) ─────────────────────────────────────────────────────
_qdrant_client     = None
_qdrant_collection = None


def _get_qdrant_collection():
    """Returns (qdrant_client, collection_name) for prod use."""
    global _qdrant_client, _qdrant_collection
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        # Gemini text-embedding-004 = 768 dims; local ONNX all-MiniLM = 384 dims
        vec_size = 768 if os.environ.get("GEMINI_API_KEY") else 384

        url     = os.environ["QDRANT_URL"]
        api_key = os.environ.get("QDRANT_API_KEY")
        _qdrant_client = QdrantClient(url=url, api_key=api_key)

        existing = [c.name for c in _qdrant_client.get_collections().collections]
        if COLLECTION_NAME in existing:
            info = _qdrant_client.get_collection(COLLECTION_NAME)
            current_size = info.config.params.vectors.size
            if current_size != vec_size:
                print(f"[Vault] Recreating Qdrant collection (dim {current_size} → {vec_size})")
                _qdrant_client.delete_collection(COLLECTION_NAME)
                _qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
                )
        else:
            _qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
            )
        _qdrant_collection = COLLECTION_NAME
        print(f"[Vault] Qdrant collection ready — {COLLECTION_NAME} (dim={vec_size})")
    return _qdrant_client, _qdrant_collection


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed using Google Gemini REST API (if key provided) or fallback to local ONNX."""
    import os
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if gemini_key:
        import requests
        url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
        embeddings = []
        for text in texts:
            resp = requests.post(
                url,
                params={"key": gemini_key},
                json={"model": "models/text-embedding-004",
                      "content": {"parts": [{"text": text}]}},
                timeout=30,
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"]["values"])
        return embeddings
    else:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        embed_fn = DefaultEmbeddingFunction()
        return [list(e) for e in embed_fn(texts)]


# ── Backend selector ──────────────────────────────────────────────────────────
def _use_qdrant() -> bool:
    return os.environ.get("VECTOR_STORE", "chromadb").lower() == "qdrant"


# ── Core: ingest ──────────────────────────────────────────────────────────────
def _ingest_chunks(
    chunks:     list[str],
    doc_id:     str,
    source:     str,
    ticker:     Optional[str],
    doc_type:   str,
    date_added: str,
):
    """Low-level: embed and store chunks into the active vector store."""
    if not chunks:
        return

    ids       = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source":     source,
            "ticker":     ticker or "",
            "doc_type":   doc_type,
            "date_added": date_added,
            "doc_id":     doc_id,
        }
        for _ in chunks
    ]

    if _use_qdrant():
        from qdrant_client.models import PointStruct
        client, col = _get_qdrant_collection()
        embeddings  = _embed_texts(chunks)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                vector=emb,
                payload={**meta, "content": chunk},
            )
            for chunk_id, emb, meta, chunk in zip(ids, embeddings, metadatas, chunks)
        ]
        client.upsert(collection_name=col, points=points)
    else:
        col = _get_chroma_collection()
        col.add(documents=chunks, ids=ids, metadatas=metadatas)

    print(f"[Vault] Ingested {len(chunks)} chunks for doc_id={doc_id} ({source})")


def ingest_text(
    text:        str,
    source_name: str,
    ticker:      Optional[str] = None,
    doc_type:    str = "other",
) -> str:
    """Chunk, embed, and store plain text. Returns the doc_id."""
    if doc_type not in VALID_DOC_TYPES:
        doc_type = "other"

    doc_id     = uuid.uuid4().hex
    date_added = date.today().isoformat()
    chunks     = _splitter.split_text(text)

    if not chunks:
        raise ValueError("Document produced zero chunks after splitting.")

    _ingest_chunks(chunks, doc_id, source_name, ticker, doc_type, date_added)
    return doc_id


def ingest_file(
    data:     bytes,
    filename: str,
    ticker:   Optional[str] = None,
    doc_type: str = "other",
) -> str:
    """
    Ingest a PDF or TXT/MD file from raw bytes.
    Returns the doc_id.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        text = _extract_pdf_text(data)
    elif ext in ("txt", "md"):
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Supported: .pdf, .txt, .md")

    if not text.strip():
        raise ValueError("File produced no extractable text.")

    return ingest_text(text, source_name=filename, ticker=ticker, doc_type=doc_type)


def _extract_pdf_text(data: bytes) -> str:
    """Extract all text from a PDF."""
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(data))
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


# ── Core: retrieve ────────────────────────────────────────────────────────────
def retrieve(
    query:         str,
    ticker_filter: Optional[str] = None,
    n_results:     int = 5,
) -> list[VaultChunk]:
    """
    Semantic search over the vault.
    Optionally filters by ticker metadata.
    Returns up to n_results VaultChunk objects.
    """
    if _use_qdrant():
        return _retrieve_qdrant(query, ticker_filter, n_results)
    return _retrieve_chroma(query, ticker_filter, n_results)


def _retrieve_chroma(query: str, ticker_filter: Optional[str], n_results: int) -> list[VaultChunk]:
    col   = _get_chroma_collection()
    total = col.count()
    if total == 0:
        return []

    safe_n  = min(n_results, total)
    where   = {"ticker": ticker_filter} if ticker_filter else None
    results = col.query(
        query_texts=[query],
        n_results=safe_n,
        where=where,
        include=["documents", "metadatas"],
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(VaultChunk(
            content=doc,
            source=meta.get("source", "unknown"),
            ticker=meta.get("ticker") or None,
            doc_type=meta.get("doc_type", "other"),
            date_added=meta.get("date_added", ""),
            doc_id=meta.get("doc_id", ""),
        ))
    return chunks


def _retrieve_qdrant(query: str, ticker_filter: Optional[str], n_results: int) -> list[VaultChunk]:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client, col = _get_qdrant_collection()
    embeddings  = _embed_texts([query])
    qfilter     = None
    if ticker_filter:
        qfilter = Filter(
            must=[FieldCondition(key="ticker", match=MatchValue(value=ticker_filter))]
        )

    hits   = client.search(
        collection_name=col,
        query_vector=embeddings[0],
        limit=n_results,
        query_filter=qfilter,
        with_payload=True,
    )
    chunks = []
    for hit in hits:
        p = hit.payload or {}
        chunks.append(VaultChunk(
            content=p.get("content", ""),
            source=p.get("source", "unknown"),
            ticker=p.get("ticker") or None,
            doc_type=p.get("doc_type", "other"),
            date_added=p.get("date_added", ""),
            doc_id=p.get("doc_id", ""),
        ))
    return chunks


# ── Core: list documents ──────────────────────────────────────────────────────
def list_documents() -> list[DocMeta]:
    """Return one DocMeta per unique doc_id (collapses chunks into parent docs)."""
    if _use_qdrant():
        return _list_documents_qdrant()
    return _list_documents_chroma()


def _list_documents_chroma() -> list[DocMeta]:
    col   = _get_chroma_collection()
    total = col.count()
    if total == 0:
        return []

    all_meta = col.get(include=["metadatas"])["metadatas"]
    seen: dict[str, DocMeta] = {}
    for meta in all_meta:
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in seen:
            seen[doc_id] = DocMeta(
                doc_id=doc_id,
                source=meta.get("source", ""),
                ticker=meta.get("ticker") or None,
                doc_type=meta.get("doc_type", "other"),
                date_added=meta.get("date_added", ""),
                chunk_count=0,
            )
        seen[doc_id].chunk_count += 1

    return sorted(seen.values(), key=lambda d: d.date_added, reverse=True)


def _list_documents_qdrant() -> list[DocMeta]:
    client, col = _get_qdrant_collection()
    results, _  = client.scroll(collection_name=col, limit=10_000, with_payload=True)
    seen: dict[str, DocMeta] = {}
    for point in results:
        p      = point.payload or {}
        doc_id = p.get("doc_id", "unknown")
        if doc_id not in seen:
            seen[doc_id] = DocMeta(
                doc_id=doc_id,
                source=p.get("source", ""),
                ticker=p.get("ticker") or None,
                doc_type=p.get("doc_type", "other"),
                date_added=p.get("date_added", ""),
                chunk_count=0,
            )
        seen[doc_id].chunk_count += 1
    return sorted(seen.values(), key=lambda d: d.date_added, reverse=True)


# ── Core: delete ──────────────────────────────────────────────────────────────
def delete_document(doc_id: str) -> bool:
    """Delete all chunks belonging to a document. Returns True if any deleted."""
    if _use_qdrant():
        return _delete_document_qdrant(doc_id)
    return _delete_document_chroma(doc_id)


def _delete_document_chroma(doc_id: str) -> bool:
    col     = _get_chroma_collection()
    results = col.get(where={"doc_id": doc_id}, include=[])
    ids     = results.get("ids", [])
    if not ids:
        return False
    col.delete(ids=ids)
    print(f"[Vault] Deleted {len(ids)} chunks for doc_id={doc_id}")
    return True


def _delete_document_qdrant(doc_id: str) -> bool:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client, col = _get_qdrant_collection()
    client.delete(
        collection_name=col,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )
    return True


# ── Utility ────────────────────────────────────────────────────────────────────
def vault_doc_count() -> int:
    """Fast count of unique documents in the vault (not chunks)."""
    try:
        return len(list_documents())
    except Exception:
        return 0
