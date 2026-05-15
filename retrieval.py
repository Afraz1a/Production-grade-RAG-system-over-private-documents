"""
retrieval.py
------------
Hybrid search = BM25 (keyword) + ChromaDB (semantic) fused with RRF.
Then re-ranks the top candidates with a cross-encoder for precision.

No API key needed here — cross-encoder runs locally.
"""

import os
import pickle
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from gpu_config import DEVICE, get_optimal_batch_size

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR  = "./chroma_db"
BM25_PATH   = "./bm25_index.pkl"
COLLECTION  = "rag_docs"

DENSE_TOP_K   = 20   # candidates from semantic search
SPARSE_TOP_K  = 20   # candidates from BM25
RERANK_TOP_K  = 5    # final chunks after re-ranking (sent to LLM)

RRF_K = 60           # RRF constant — higher = smoother rank fusion


# ── Load saved indexes ────────────────────────────────────────────────────────
def load_vectorstore():
    """Load ChromaDB with the same local GPU embedding model used during ingestion."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectorstore


def load_bm25():
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError(
            f"BM25 index not found at '{BM25_PATH}'. Run ingest.py first."
        )
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["corpus"], data["chunks"]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────
def reciprocal_rank_fusion(dense_docs, sparse_docs, k=RRF_K):
    """
    Merge two ranked lists into one without needing raw scores.

    RRF formula: score(doc) = Σ 1 / (k + rank)
    A doc appearing high in both lists gets a strong combined score.
    A doc that only appears in one list still gets partial credit.
    """
    scores = {}    # doc_id → fused score
    id_to_doc = {} # doc_id → actual document

    for rank, doc in enumerate(dense_docs):
        doc_id = doc.page_content[:80]   # use start of content as a stable key
        scores[doc_id]    = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        id_to_doc[doc_id] = doc

    for rank, doc in enumerate(sparse_docs):
        doc_id = doc.page_content[:80]
        scores[doc_id]    = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        id_to_doc[doc_id] = doc

    # Sort by fused score, highest first
    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [id_to_doc[doc_id] for doc_id in ranked_ids]


# ── BM25 search ──────────────────────────────────────────────────────────────
def bm25_search(query: str, bm25, chunks, top_k=SPARSE_TOP_K):
    """Keyword-based retrieval using BM25."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its score and sort
    scored = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [doc for doc, score in scored[:top_k]]


# ── Cross-encoder re-ranking ──────────────────────────────────────────────────
class Reranker:
    """
    Cross-encoder re-ranking on GPU (RTX 4050).
    Unlike bi-encoders (which embed query and doc separately),
    cross-encoders read query + chunk TOGETHER → much more accurate.
    GPU batching makes this fast even with 20+ candidates.
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[retrieval] Loading cross-encoder on {DEVICE.upper()}...")
        self.model      = CrossEncoder(model_name, device=DEVICE)
        self.batch_size = get_optimal_batch_size()
        print(f"[retrieval] Re-ranker ready — batch size: {self.batch_size}")

    def rerank(self, query: str, docs, top_k=RERANK_TOP_K):
        if not docs:
            return []

        pairs  = [(query, doc.page_content) for doc in docs]

        # batch_size pushes multiple pairs through GPU in parallel
        # without it, each pair runs sequentially → very slow on CPU
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


# ── Main retrieval pipeline ───────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self):
        print("[retrieval] Loading indexes...")
        self.vectorstore        = load_vectorstore()
        self.bm25, self.corpus, self.chunks = load_bm25()
        self.reranker           = Reranker()
        print("[retrieval] All indexes loaded. Ready.\n")

    def retrieve(self, query: str, verbose=False):
        """
        Full pipeline:
        1. Dense search  → top 20 semantic matches
        2. BM25 search   → top 20 keyword matches
        3. RRF fusion    → merged ranked list
        4. Re-ranking    → top 5 precise results
        5. Parent fetch  → swap child chunks for their parent content
        """

        # 1. Dense (semantic) retrieval
        dense_docs = self.vectorstore.similarity_search(query, k=DENSE_TOP_K)

        # 2. Sparse (keyword) retrieval
        sparse_docs = bm25_search(query, self.bm25, self.chunks)

        # 3. RRF fusion
        fused = reciprocal_rank_fusion(dense_docs, sparse_docs)

        if verbose:
            print(f"[retrieval] Dense: {len(dense_docs)} | "
                  f"Sparse: {len(sparse_docs)} | Fused: {len(fused)}")

        # 4. Cross-encoder re-ranking on fused candidates
        reranked = self.reranker.rerank(query, fused, top_k=RERANK_TOP_K)

        # 5. Swap child chunks → parent chunks (more context for the LLM)
        final_docs = self._fetch_parent_context(reranked)

        if verbose:
            print(f"[retrieval] Final docs after rerank + parent fetch: {len(final_docs)}")

        return final_docs

    def _fetch_parent_context(self, docs):
        """
        Each child chunk stores its parent's full text in metadata.
        We swap child content → parent content so the LLM gets more context.
        """
        seen = set()
        result = []
        for doc in docs:
            parent_text = doc.metadata.get("parent_content", doc.page_content)
            if parent_text not in seen:
                seen.add(parent_text)
                doc.page_content = parent_text  # replace child with parent content
                result.append(doc)
        return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = HybridRetriever()
    query = input("Enter a test query: ")
    results = retriever.retrieve(query, verbose=True)

    print(f"\nTop {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content[:300])
        print()