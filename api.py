"""
api.py
------
FastAPI backend — endpoints:
  POST /upload       → ingest a document (adds to existing, no duplicates)
  POST /query        → ask a question, get an answer
  GET  /documents    → list all ingested documents
  DELETE /documents  → clear all documents from the database

Run with: uvicorn api:app --reload
"""

import os
import shutil
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from ingest import ingest, CHROMA_DIR, BM25_PATH
from retrieval import HybridRetriever
from chain import RAGChain

load_dotenv()

app = FastAPI(title="Production RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals ───────────────────────────────────────────────────────────────────
_retriever   = None
_chain       = None
DOCS_REGISTRY = "./ingested_docs.json"  # tracks which docs have been ingested


def get_chain():
    global _retriever, _chain
    if _chain is None:
        _retriever = HybridRetriever()
        _chain     = RAGChain(_retriever)
    return _chain


def reset_chain():
    """Force chain to reload on next query."""
    global _retriever, _chain
    _retriever = None
    _chain     = None


def load_doc_registry() -> list:
    if os.path.exists(DOCS_REGISTRY):
        with open(DOCS_REGISTRY) as f:
            return json.load(f)
    return []


def save_doc_registry(docs: list):
    with open(DOCS_REGISTRY, "w") as f:
        json.dump(docs, f)


# ── Models ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question:     str
    answer:       str
    guardrail:    dict
    num_sources:  int
    source_texts: list


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "RAG API is running", "docs": "/docs"}


@app.get("/documents")
def list_documents():
    """Return list of all ingested documents."""
    docs = load_doc_registry()
    return {"documents": docs, "count": len(docs)}


@app.delete("/documents")
def clear_documents():
    """
    Wipe the entire vector DB and BM25 index.
    Use this to start fresh with new documents.
    """
    # Delete ChromaDB folder
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # Delete BM25 index
    if os.path.exists(BM25_PATH):
        os.remove(BM25_PATH)

    # Clear registry
    save_doc_registry([])

    # Reset chain so it doesn't hold stale data
    reset_chain()

    return {"status": "success", "message": "All documents cleared."}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF or TXT.
    - Skips if the same filename was already ingested
    - Adds to existing documents (does not wipe previous ones)
    """
    allowed = {".pdf", ".txt", ".md"}
    ext     = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}"
        )

    # Check if already ingested
    docs = load_doc_registry()
    if file.filename in docs:
        return {
            "status":   "skipped",
            "filename": file.filename,
            "message":  "Document already ingested. Clear DB first if you want to re-ingest."
        }

    # Save to temp file, ingest, delete
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ingest(tmp_path)
        docs.append(file.filename)
        save_doc_registry(docs)
        reset_chain()
    finally:
        os.unlink(tmp_path)

    return {
        "status":   "success",
        "filename": file.filename,
        "message":  f"Ingested. Total documents: {len(docs)}"
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Ask a question against all ingested documents."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    docs = load_doc_registry()
    if not docs:
        raise HTTPException(
            status_code=404,
            detail="No documents ingested yet. Upload a document first."
        )

    try:
        chain  = get_chain()
        result = chain.run(request.question)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question     = result["question"],
        answer       = result["answer"],
        guardrail    = result["guardrail"],
        num_sources  = result["num_sources"],
        source_texts = [doc.page_content for doc in result["sources"]],
    )


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)