"""
chain.py
--------
Query rewriting → retrieve → generate answer → hallucination guardrail.

This is where the LLM lives. Everything else feeds into this.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Models ────────────────────────────────────────────────────────────────────
# llama-3.3-70b-versatile is free on Groq and very fast
LLM_MODEL = "llama-3.3-70b-versatile"


# ── Prompt templates ──────────────────────────────────────────────────────────

REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are a query optimization assistant. Your job is to rewrite a user's question
into 3 alternative versions that capture the same intent but use different wording.
This helps retrieve more relevant documents from a vector database.

Original question: {query}

Return ONLY the 3 rewritten questions, one per line. No numbering, no explanations.
""")

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the provided documents to answer this."
Do NOT make up information. Cite relevant parts of the context in your answer.

Context:
{context}

Question: {question}

Answer:
""")

GUARDRAIL_PROMPT = ChatPromptTemplate.from_template("""
You are a fact-checking assistant. Given a question, a context, and an answer,
determine if the answer is fully supported by the context.

Question: {question}

Context:
{context}

Answer to check: {answer}

Respond with ONLY one of:
- SUPPORTED: (brief reason)
- UNSUPPORTED: (what was not in the context)
- PARTIALLY SUPPORTED: (what is and isn't supported)
""")


# ── Query rewriting ───────────────────────────────────────────────────────────
def rewrite_query(query: str, llm) -> list[str]:
    """
    Generate 3 alternative phrasings of the query.
    We always keep the original too, so retrieval uses 4 queries total.

    Why: Embedding models may miss a relevant doc if the query phrasing
    doesn't match the document's wording. Multiple phrasings = better recall.
    """
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"query": query})

    rewrites = [line.strip() for line in result.strip().split("\n") if line.strip()]
    all_queries = [query] + rewrites[:3]  # original + up to 3 rewrites

    return all_queries


# ── Hallucination guardrail ────────────────────────────────────────────────────
def check_hallucination(question: str, context: str, answer: str, llm) -> dict:
    """
    Ask a second LLM call to verify the answer is grounded in the context.
    Returns a dict with 'verdict' and 'reason'.

    In production you'd also use RAGAS faithfulness score for this.
    """
    chain = GUARDRAIL_PROMPT | llm | StrOutputParser()
    result = chain.invoke({
        "question": question,
        "context": context,
        "answer": answer,
    })

    # Parse the response
    result = result.strip()
    if result.startswith("SUPPORTED"):
        verdict = "SUPPORTED"
    elif result.startswith("UNSUPPORTED"):
        verdict = "UNSUPPORTED"
    else:
        verdict = "PARTIALLY SUPPORTED"

    return {"verdict": verdict, "detail": result}


# ── Main RAG chain ────────────────────────────────────────────────────────────
class RAGChain:
    def __init__(self, retriever):
        """
        retriever: an instance of HybridRetriever from retrieval.py
        """
        self.retriever = retriever
        self.llm       = ChatGroq(model=LLM_MODEL, temperature=0)
        self.rag_chain = RAG_PROMPT | self.llm | StrOutputParser()

    def run(self, question: str, verbose=False):
        """
        Full pipeline:
        1. Rewrite query into multiple versions
        2. Retrieve docs for each version, deduplicate
        3. Generate answer from combined context
        4. Run hallucination check
        5. Return answer + sources + guardrail verdict
        """

        # 1. Query rewriting
        if verbose:
            print(f"\n[chain] Original query: {question}")

        queries = rewrite_query(question, self.llm)

        if verbose:
            print(f"[chain] Rewritten queries:")
            for q in queries[1:]:
                print(f"  → {q}")

        # 2. Retrieve for all query versions, deduplicate by content
        all_docs = []
        seen_content = set()
        for q in queries:
            docs = self.retriever.retrieve(q)
            for doc in docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    all_docs.append(doc)

        if verbose:
            print(f"\n[chain] Total unique docs after multi-query retrieval: {len(all_docs)}")

        # 3. Build context string from retrieved docs
        context = "\n\n---\n\n".join(
            f"[Source {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(all_docs)
        )

        # 4. Generate answer
        answer = self.rag_chain.invoke({
            "context":  context,
            "question": question,
        })

        # 5. Hallucination check
        guardrail = check_hallucination(question, context, answer, self.llm)

        if verbose:
            print(f"\n[chain] Guardrail verdict: {guardrail['verdict']}")

        return {
            "question":   question,
            "answer":     answer,
            "sources":    all_docs,
            "guardrail":  guardrail,
            "num_sources": len(all_docs),
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from retrieval import HybridRetriever

    retriever = HybridRetriever()
    chain     = RAGChain(retriever)

    question = input("Ask a question: ")
    result   = chain.run(question, verbose=True)

    print(f"\n{'='*50}")
    print(f"Answer:\n{result['answer']}")
    print(f"\nGuardrail: {result['guardrail']['verdict']}")
    print(f"Sources used: {result['num_sources']}")