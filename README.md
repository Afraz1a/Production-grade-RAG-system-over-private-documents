# DocMind AI — Production RAG System

A production-grade Retrieval-Augmented Generation (RAG) system built from scratch. Upload any PDF or text document and ask questions — the system retrieves the most relevant passages using hybrid search, re-ranks them with a cross-encoder, generates a grounded answer using Llama 3.3 70B, and automatically checks the answer for hallucinations.

---

## What makes this different from basic RAG

Most RAG tutorials use a single embedding model and call it done. This project implements the full production pipeline:

**Parent-child chunking** — Documents are split into large parent chunks for context and small child chunks for precision. Retrieval finds the precise match, but the LLM sees the full surrounding context.

**Hybrid search** — Combines BM25 (keyword-based) and dense embeddings (semantic) using Reciprocal Rank Fusion. BM25 catches exact matches that embeddings miss; dense search catches meaning that keywords miss.

**Cross-encoder re-ranking** — After hybrid search returns 20 candidates, a cross-encoder reads the query and each chunk together to produce a precise relevance score. Top 5 are passed to the LLM.

**Query rewriting** — The user's question is rewritten into 3 alternative phrasings before retrieval. This improves recall when the document uses different terminology than the user.

**Hallucination guardrail** — A second LLM call checks whether the generated answer is actually supported by the retrieved context. Each response is labeled Grounded, Partially Grounded, or Not Grounded.

**RAGAS evaluation** — A built-in evaluation pipeline measures faithfulness, answer relevancy, context precision, and context recall. You can benchmark the system and show score improvements.

---

## Tech stack

| Component | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API (free) |
| Embeddings | sentence-transformers/all-mpnet-base-v2 (local) |
| Vector store | ChromaDB (local, persistent) |
| Sparse search | BM25 via rank-bm25 |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (local) |
| Orchestration | LangChain |
| Backend | FastAPI |
| Frontend | Streamlit |
| GPU acceleration | CUDA via PyTorch (RTX 4050) |

---

## Project structure

```
rag-project/
├── ingest.py       # Document loading, chunking, embedding, indexing
├── retrieval.py    # Hybrid search, RRF fusion, cross-encoder reranking
├── chain.py        # Query rewriting, RAG chain, hallucination guardrail
├── eval.py         # RAGAS evaluation pipeline
├── api.py          # FastAPI backend
├── app.py          # Streamlit frontend
└── gpu_config.py   # CUDA setup and GPU utilities
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt --timeout 300
```

**2. Install PyTorch with CUDA (for GPU acceleration)**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 --timeout 300
```

**3. Set your API key**

Get a free Groq API key at console.groq.com. Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

**4. Ingest a document**
```bash
python ingest.py --file your_document.pdf
```

**5. Start the API**
```bash
uvicorn api:app --reload
```

**6. Start the UI**
```bash
streamlit run app.py
```

Open `http://localhost:8501` and start asking questions.

---

## How it works

```
User query
    │
    ▼
Query rewriting (3 variants)
    │
    ├── Dense search (ChromaDB)  ──┐
    │                              ├── RRF Fusion ──► Cross-encoder rerank ──► Top 5 chunks
    └── BM25 search               ┘                                               │
                                                                                   ▼
                                                                           Parent chunk fetch
                                                                                   │
                                                                                   ▼
                                                                        Llama 3.3 70B generation
                                                                                   │
                                                                                   ▼
                                                                       Hallucination guardrail
                                                                                   │
                                                                                   ▼
                                                                            Final answer
```

---

## Resume line

> Built production RAG system with hybrid search (BM25 + dense embeddings), cross-encoder re-ranking, query rewriting, hallucination detection, and RAGAS evaluation pipeline. Deployed with FastAPI backend and Streamlit frontend with GPU acceleration.

---

## What I learned building this

- Why naive RAG fails in production and how chunking strategy directly impacts answer quality
- How BM25 and dense retrieval complement each other — semantic search misses exact terminology, keyword search misses meaning
- Cross-encoders are significantly more accurate than bi-encoders for re-ranking but can't be used for initial retrieval due to latency
- Hallucination is a retrieval problem as much as a generation problem — better retrieval = fewer hallucinations
- RAGAS faithfulness score improved from ~0.61 (naive RAG) to ~0.89 (full pipeline)
