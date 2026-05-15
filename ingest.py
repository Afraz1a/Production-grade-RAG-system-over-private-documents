"""
ingest.py
---------
Loads documents → splits into parent/child chunks → embeds → stores in ChromaDB.
Also builds a BM25 index for hybrid search later.

Usage:
    python ingest.py --file path/to/your.pdf
    python ingest.py --file path/to/your.txt
"""

import os
import pickle
import argparse
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from gpu_config import DEVICE, get_optimal_batch_size

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR   = "./chroma_db"       # where ChromaDB persists data
BM25_PATH    = "./bm25_index.pkl"  # where BM25 index is saved
COLLECTION   = "rag_docs"

# Parent chunks → sent to LLM for context (bigger = more context)
PARENT_SIZE  = 2000
PARENT_OVERLAP = 200

# Child chunks → used for retrieval (smaller = more precise matches)
CHILD_SIZE   = 400
CHILD_OVERLAP = 50

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_document(file_path: str):
    """Load a PDF or TXT file into LangChain documents."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
    elif path.suffix.lower() in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use PDF or TXT.")

    docs = loader.load()
    print(f"[ingest] Loaded {len(docs)} page(s) from '{path.name}'")
    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────
def create_parent_child_chunks(docs):
    """
    Parent-child chunking strategy:
      - Parent chunks (large) carry full context → stored separately
      - Child chunks (small) are what we actually retrieve → stored in ChromaDB
      - Each child has a metadata field pointing to its parent's content

    Why this matters:
      Naive RAG retrieves tiny chunks that lack context.
      With parent-child, we retrieve precisely but answer with full context.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_SIZE,
        chunk_overlap=PARENT_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_SIZE,
        chunk_overlap=CHILD_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    parent_chunks = parent_splitter.split_documents(docs)
    print(f"[ingest] Created {len(parent_chunks)} parent chunk(s)")

    child_chunks = []
    for p_idx, parent in enumerate(parent_chunks):
        children = child_splitter.split_documents([parent])
        for child in children:
            # Tag each child with its parent's content and index
            child.metadata["parent_content"] = parent.page_content
            child.metadata["parent_id"]      = str(p_idx)
        child_chunks.extend(children)

    print(f"[ingest] Created {len(child_chunks)} child chunk(s) "
          f"(avg {len(child_chunks)//max(len(parent_chunks),1)} per parent)")
    return child_chunks


# ── Embedding + Vector Store ──────────────────────────────────────────────────
def store_in_chroma(child_chunks):
    """
    Embed child chunks and store in ChromaDB.

    Uses a local HuggingFace model (all-mpnet-base-v2) running on your RTX 4050.
    - Faster than OpenAI for bulk ingestion (no network round-trips)
    - Free (no API cost)
    - all-mpnet-base-v2 is one of the best open-source embedding models

    Note: retrieval.py must use the same embedding model — both use this function.
    """
    batch_size = get_optimal_batch_size()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[ingest] Embedding {len(child_chunks)} chunks on {DEVICE.upper()} "
          f"(batch size: {batch_size})...")

    vectorstore = Chroma.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    print(f"[ingest] ✓ Stored {len(child_chunks)} chunks in ChromaDB at '{CHROMA_DIR}'")
    return vectorstore


# ── BM25 Index ────────────────────────────────────────────────────────────────
def build_bm25_index(child_chunks):
    """
    Build a BM25 keyword index from the same child chunks.
    This is the 'sparse' half of hybrid search.

    Saved to disk so retrieval.py can load it without re-ingesting.
    """
    corpus = [chunk.page_content for chunk in child_chunks]
    tokenized = [text.lower().split() for text in corpus]
    bm25 = BM25Okapi(tokenized)

    # Save index + raw texts together so we can map scores → documents
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus, "chunks": child_chunks}, f)

    print(f"[ingest] BM25 index saved to '{BM25_PATH}'")
    return bm25


# ── Main ──────────────────────────────────────────────────────────────────────
def ingest(file_path: str):
    """Full ingestion pipeline for one file."""
    print(f"\n{'='*50}")
    print(f"Ingesting: {file_path}")
    print(f"{'='*50}\n")

    docs         = load_document(file_path)
    child_chunks = create_parent_child_chunks(docs)
    vectorstore  = store_in_chroma(child_chunks)
    bm25         = build_bm25_index(child_chunks)

    print(f"\n✓ Ingestion complete. Ready to query.\n")
    return vectorstore, bm25


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a document into the RAG system")
    parser.add_argument("--file", required=True, help="Path to PDF or TXT file")
    args = parser.parse_args()
    ingest(args.file)