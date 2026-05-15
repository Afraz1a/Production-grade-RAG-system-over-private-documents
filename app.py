"""
app.py — DocMind AI
Clean professional UI, no avatar conflicts, no overlapping elements.
"""

import streamlit as st
import requests
import time
import json
import os
from datetime import datetime

API_URL    = "http://localhost:8000"
STATE_FILE = "./chat_state.json"

st.set_page_config(
    page_title="DocMind AI",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Base */
.stApp { background-color: #0a0e17 !important; }

/* Hide all streamlit default chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Lock sidebar permanently open - no collapse button */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarOpenButton"]     { display: none !important; }

/* Sidebar always visible */
[data-testid="stSidebar"] {
    transform: none !important;
    display: flex !important;
    visibility: visible !important;
    background-color: #0d1117 !important;
    border-right: 1px solid #161d2b !important;
    width: 270px !important;
    min-width: 270px !important;
    max-width: 270px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
    overflow-x: hidden !important;
}
[data-testid="stChatMessageAvatarUser"] { display: none !important; width: 0 !important; }
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; width: 0 !important; }
[data-testid="stChatMessageAvatar"] { display: none !important; width: 0 !important; }

/* Chat message content takes full width */
[data-testid="stChatMessageContent"] {
    width: 100% !important;
    max-width: 100% !important;
}
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 2px 0 !important;
    gap: 0 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #161d2b !important;
    width: 270px !important;
    min-width: 270px !important;
    max-width: 270px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
    overflow-x: hidden !important;
}

/* Buttons */
.stButton > button {
    width: 100% !important;
    background-color: #161d2b !important;
    color: #8899aa !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 7px 12px !important;
    text-align: left !important;
    transition: all 0.15s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background-color: #1e2d3d !important;
    color: #c8d8e8 !important;
    border-color: #2d4a6a !important;
}
[data-testid="stBaseButton-primary"] {
    background-color: #1a3a6e !important;
    color: #93c5fd !important;
    border-color: #1e4a8a !important;
}
[data-testid="stBaseButton-primary"]:hover {
    background-color: #1e4a8a !important;
    color: #bfdbfe !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1px dashed #1e2d3d !important;
    border-radius: 8px !important;
    padding: 4px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 8px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span {
    font-size: 0.75rem !important;
    color: #4a5568 !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    width: 100% !important;
    background-color: #161d2b !important;
    color: #8899aa !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 7px 12px !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background-color: #0d1117 !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 10px !important;
    color: #c8d8e8 !important;
    font-size: 0.88rem !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background-color: #1a3a6e !important;
    border-radius: 8px !important;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #0d1117 !important;
    border: 1px solid #161d2b !important;
    border-radius: 8px !important;
    margin-top: 8px !important;
}
details summary {
    font-size: 0.75rem !important;
    color: #4a5568 !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #0d1117 !important;
    border: 1px solid #161d2b !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}
[data-testid="stMetricLabel"] { font-size: 0.65rem !important; color: #4a5568 !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { font-size: 1.1rem !important; color: #c8d8e8 !important; font-weight: 600 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0e17; }
::-webkit-scrollbar-thumb { background: #1e2d3d; border-radius: 4px; }

/* Divider */
hr { border-color: #161d2b !important; margin: 0 !important; }

/* Spinner */
[data-testid="stSpinner"] p { color: #4a5568 !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Persistent storage ────────────────────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {"messages": [], "query_history": [], "total_queries": 0, "total_time": 0.0}

def save_state():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "messages":      st.session_state["messages"],
                "query_history": st.session_state["query_history"],
                "total_queries": st.session_state["total_queries"],
                "total_time":    st.session_state["total_time"],
            }, f, ensure_ascii=False, indent=2)
    except: pass

def clear_state():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

if "state_loaded" not in st.session_state:
    saved = load_state()
    # Sanitize loaded messages — ensure meta fields are always safe dicts
    clean_messages = []
    for m in saved.get("messages", []):
        if m.get("role") == "assistant":
            meta = m.get("meta") or {}
            m["meta"] = {
                "response_time": float(meta.get("response_time", 0)),
                "num_sources":   int(meta.get("num_sources", 0)),
                "guardrail":     meta.get("guardrail") or {"verdict": ""},
                "sources":       meta.get("sources") or [],
            }
        clean_messages.append(m)
    saved["messages"] = clean_messages
    st.session_state.update({**saved, "ingested_docs": [], "state_loaded": True})


# ── Helpers ───────────────────────────────────────────────────────────────────
def verdict_html(v):
    if v == "SUPPORTED":
        return '<span style="display:inline-block;padding:2px 10px;border-radius:100px;font-size:0.68rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;background:#052e16;color:#34d399;border:1px solid #065f46;">Grounded</span>'
    elif v == "UNSUPPORTED":
        return '<span style="display:inline-block;padding:2px 10px;border-radius:100px;font-size:0.68rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;background:#1c0505;color:#f87171;border:1px solid #7f1d1d;">Not Grounded</span>'
    else:
        return '<span style="display:inline-block;padding:2px 10px;border-radius:100px;font-size:0.68rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;background:#1c1005;color:#fbbf24;border:1px solid #78350f;">Partially Grounded</span>'

def export_chat():
    lines = [f"DocMind AI  Export  {datetime.now().strftime('%Y-%m-%d %H:%M')}", "="*60]
    for m in st.session_state["messages"]:
        role = "User" if m["role"] == "user" else "Assistant"
        lines += [f"\n[{role}]", m["content"]]
        if m.get("meta"):
            lines.append(f"Grounding: {m['meta']['guardrail']['verdict']}")
    return "\n".join(lines)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Logo block
    st.markdown("""
    <div style="padding:24px 20px 18px; border-bottom:1px solid #161d2b;">
        <div style="font-size:1rem;font-weight:700;color:#f0f4f8;letter-spacing:-0.02em;">DocMind AI</div>
        <div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;margin-top:3px;font-weight:500;">Document Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    total_q = st.session_state["total_queries"]
    avg_t   = (st.session_state["total_time"] / total_q) if total_q > 0 else 0.0

    st.markdown(f"""
    <div style="padding:18px 20px;border-bottom:1px solid #161d2b;">
        <div style="font-size:0.62rem;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">Statistics</div>
        <div style="display:flex;gap:8px;">
            <div style="flex:1;background:#0d1117;border:1px solid #161d2b;border-radius:8px;padding:10px 12px;">
                <div style="font-size:1.3rem;font-weight:700;color:#f0f4f8;line-height:1;">{total_q}</div>
                <div style="font-size:0.62rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.08em;margin-top:3px;">Queries</div>
            </div>
            <div style="flex:1;background:#0d1117;border:1px solid #161d2b;border-radius:8px;padding:10px 12px;">
                <div style="font-size:1.3rem;font-weight:700;color:#f0f4f8;line-height:1;">{avg_t:.1f}s</div>
                <div style="font-size:0.62rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.08em;margin-top:3px;">Avg Time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Documents
    st.markdown("""
    <div style="padding:18px 20px 10px;border-bottom:1px solid #161d2b;">
        <div style="font-size:0.62rem;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">Documents</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div style='padding:0 12px;'>", unsafe_allow_html=True)

        try:
            r = requests.get(f"{API_URL}/documents", timeout=3)
            if r.status_code == 200:
                st.session_state["ingested_docs"] = r.json().get("documents", [])
        except: pass

        uploaded = st.file_uploader("Upload Document", type=["pdf", "txt", "md"], label_visibility="collapsed")

        if uploaded:
            if st.button("Ingest Document", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        r = requests.post(f"{API_URL}/upload",
                            files={"file": (uploaded.name, uploaded.getvalue())}, timeout=300)
                        if r.status_code == 200:
                            d = r.json()
                            if d["status"] == "skipped":
                                st.warning("Already ingested.")
                            else:
                                st.success("Ingested.")
                        else:
                            st.error(r.json().get("detail", "Failed"))
                    except Exception as e:
                        st.error(str(e))
                st.rerun()

        if st.session_state["ingested_docs"]:
            for doc in st.session_state["ingested_docs"]:
                name = doc[:28] + "..." if len(doc) > 28 else doc
                st.markdown(f"""
                <div style="background:#0d1117;border:1px solid #161d2b;border-radius:6px;padding:7px 10px;margin:4px 0;font-size:0.74rem;color:#64748b;display:flex;align-items:center;gap:6px;">
                    <span style="color:#2563eb;font-size:0.7rem;">&#9632;</span> {name}
                </div>
                """, unsafe_allow_html=True)

            if st.button("Clear All Documents"):
                try:
                    r = requests.delete(f"{API_URL}/documents", timeout=10)
                    if r.status_code == 200:
                        st.session_state["ingested_docs"] = []
                        st.rerun()
                except Exception as e:
                    st.error(str(e))

        st.markdown("</div>", unsafe_allow_html=True)

    # Recent queries
    st.markdown("""
    <div style="padding:18px 20px 10px;border-bottom:1px solid #161d2b;">
        <div style="font-size:0.62rem;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">Recent Queries</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div style='padding:4px 12px 12px;'>", unsafe_allow_html=True)
        history = st.session_state.get("query_history", [])
        if history:
            for q in reversed(history[-6:]):
                label = q[:30] + "..." if len(q) > 30 else q
                st.markdown(f"""
                <div style="padding:6px 10px;border-radius:6px;background:#0d1117;border:1px solid #161d2b;
                    font-size:0.73rem;color:#4a5568;margin:3px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                    {label}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<p style="font-size:0.73rem;color:#1e2d3d;padding:0 8px;">No queries yet</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Actions
    st.markdown("""
    <div style="padding:18px 20px 10px;border-bottom:1px solid #161d2b;">
        <div style="font-size:0.62rem;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">Actions</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div style='padding:8px 12px;'>", unsafe_allow_html=True)
        if st.button("Clear Conversation"):
            st.session_state.update({"messages": [], "query_history": [], "total_queries": 0, "total_time": 0.0})
            clear_state()
            st.rerun()
        st.download_button(
            "Export Conversation",
            data=export_chat(),
            file_name=f"docmind_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Pipeline info
    st.markdown("""
    <div style="padding:18px 20px;">
        <div style="font-size:0.62rem;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">Pipeline</div>
        <div style="font-size:0.72rem;color:#2d3f57;line-height:2.1;">
            Query Rewriting<br>
            Hybrid Search &nbsp;&middot;&nbsp; BM25 + Dense<br>
            Cross-Encoder Reranking<br>
            Llama 3.3 70B Generation<br>
            Hallucination Detection
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="background:#0d1117;border-bottom:1px solid #161d2b;padding:16px 36px;">
    <div style="font-size:0.9rem;font-weight:600;color:#f0f4f8;letter-spacing:-0.01em;">Conversation</div>
    <div style="font-size:0.7rem;color:#4a5568;margin-top:2px;">Ask questions about your ingested documents</div>
</div>
""", unsafe_allow_html=True)

# Chat area wrapper
st.markdown('<div style="max-width:820px;margin:0 auto;padding:28px 36px 120px;">', unsafe_allow_html=True)

# Empty state
if not st.session_state["messages"]:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;">
        <div style="width:48px;height:48px;background:#0d1117;border:1px solid #161d2b;border-radius:12px;margin:0 auto 20px;display:flex;align-items:center;justify-content:center;">
            <span style="font-size:1.2rem;color:#1e2d3d;">D</span>
        </div>
        <h3 style="color:#2d3f57;font-size:1rem;font-weight:600;margin-bottom:8px;">No conversation yet</h3>
        <p style="color:#1e2d3d;font-size:0.82rem;">Upload a document from the sidebar, then ask your first question.</p>
        <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-top:24px;">
            <span style="background:#0d1117;border:1px solid #161d2b;border-radius:100px;padding:5px 14px;font-size:0.74rem;color:#2d3f57;">Medical Reports</span>
            <span style="background:#0d1117;border:1px solid #161d2b;border-radius:100px;padding:5px 14px;font-size:0.74rem;color:#2d3f57;">Legal Contracts</span>
            <span style="background:#0d1117;border:1px solid #161d2b;border-radius:100px;padding:5px 14px;font-size:0.74rem;color:#2d3f57;">Research Papers</span>
            <span style="background:#0d1117;border:1px solid #161d2b;border-radius:100px;padding:5px 14px;font-size:0.74rem;color:#2d3f57;">Policy Documents</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render messages
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style="display:flex;justify-content:flex-end;margin:10px 0;">
            <div style="background:#111827;border:1px solid #1e2d3d;border-radius:14px 14px 4px 14px;
                padding:12px 16px;max-width:78%;font-size:0.86rem;color:#c8d8e8;line-height:1.6;">
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        meta = msg.get("meta", {})
        src_count = meta.get("num_sources", 0)
        resp_time = meta.get("response_time", 0)
        verdict   = meta.get("guardrail", {}).get("verdict", "")
        sources   = meta.get("sources", [])

        st.markdown(f"""
        <div style="margin:10px 0;">
            <div style="background:#0d1117;border:1px solid #161d2b;border-radius:4px 14px 14px 14px;
                padding:14px 18px;max-width:92%;font-size:0.86rem;color:#94a3b8;line-height:1.75;">
                {msg["content"]}
            </div>
        """, unsafe_allow_html=True)

        if meta:
            st.markdown(f"""
            <div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;align-items:center;">
                <span style="background:#0d1117;border:1px solid #161d2b;border-radius:6px;padding:3px 9px;font-size:0.68rem;color:#4a5568;">
                    {resp_time:.1f}s
                </span>
                <span style="background:#0d1117;border:1px solid #161d2b;border-radius:6px;padding:3px 9px;font-size:0.68rem;color:#4a5568;">
                    {src_count} sources
                </span>
                <span style="background:#0d1117;border:1px solid #161d2b;border-radius:6px;padding:3px 9px;font-size:0.68rem;color:#4a5568;">
                    Llama 3.3 70B
                </span>
                {verdict_html(verdict)}
            </div>
            """, unsafe_allow_html=True)

        if sources:
            with st.expander(f"View {len(sources)} retrieved sources"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"""
                    <div style="background:#080b12;border:1px solid #161d2b;border-left:3px solid #1e40af;
                        border-radius:0 6px 6px 0;padding:10px 14px;margin:6px 0;
                        font-size:0.78rem;color:#4a5568;line-height:1.6;">
                        <div style="font-size:0.62rem;font-weight:700;color:#2563eb;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:5px;">Source {i}</div>
                        {src[:500]}{"..." if len(src) > 500 else ""}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about your document..."):

    st.session_state["messages"].append({"role": "user", "content": question})
    st.session_state["query_history"].append(question)

    with st.spinner("Searching and generating response..."):
        start = time.time()
        try:
            resp    = requests.post(f"{API_URL}/query", json={"question": question}, timeout=300)
            elapsed = time.time() - start

            if resp.status_code == 200:
                data      = resp.json()
                answer    = data["answer"]
                guardrail = data["guardrail"]
                sources   = data.get("source_texts", [])
                n_src     = data["num_sources"]

                st.session_state["total_queries"] += 1
                st.session_state["total_time"]    += elapsed
                st.session_state["messages"].append({
                    "role": "assistant", "content": answer,
                    "meta": {
                        "response_time": elapsed,
                        "num_sources":   n_src,
                        "guardrail":     guardrail,
                        "sources":       sources,
                    }
                })
                save_state()
                st.rerun()

            else:
                st.error(resp.json().get("detail", "An error occurred."))

        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Run: uvicorn api:app --reload")
        except Exception as e:
            st.error(f"Error: {e}")