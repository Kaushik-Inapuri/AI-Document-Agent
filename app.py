"""
app.py
------
DocMind AI — Intelligent Document Analysis Agent
Futuristic Blue Tech UI
"""

import os
import time
import streamlit as st

from modules.loader      import load_document
from modules.chunking    import split_text
from modules.embeddings  import create_embeddings
from modules.vectorstore import create_vector_db
from modules.qa_system   import create_qa_chain, ask_question
from modules.summarizer  import summarize_document
from utils.helpers       import (
    format_file_size, estimate_reading_time,
    count_words, ensure_dir,
)

st.set_page_config(
    page_title="DocMind AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Exo+2:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background: #020b18;
    color: #c8e6ff;
}
.main { background: #020b18; }
.block-container { padding: 2rem 2.5rem 5rem; max-width: 1200px; }

/* ── Animated background grid ── */
.main::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(0,120,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,120,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Typography ── */
h1 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 900 !important;
    letter-spacing: 2px !important;
    background: linear-gradient(90deg, #00d4ff, #0078ff, #00d4ff);
    background-size: 200% auto;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: shine 3s linear infinite;
}
h2, h3 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    color: #00d4ff !important;
}
@keyframes shine {
    to { background-position: 200% center; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020d1f 0%, #041428 100%) !important;
    border-right: 1px solid rgba(0,120,255,0.3) !important;
    box-shadow: 4px 0 20px rgba(0,120,255,0.1);
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    color: #00d4ff !important;
    border: 1px solid #00d4ff !important;
    border-radius: 4px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    padding: 10px 24px !important;
    text-transform: uppercase !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.3s !important;
}
.stButton > button::before {
    content: '' !important;
    position: absolute !important;
    top: 0; left: -100% !important;
    width: 100%; height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.15), transparent) !important;
    transition: left 0.4s !important;
}
.stButton > button:hover::before { left: 100% !important; }
.stButton > button:hover {
    background: rgba(0,212,255,0.1) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.4), inset 0 0 20px rgba(0,212,255,0.05) !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(0,20,50,0.8) !important;
    border: 1px solid rgba(0,120,255,0.4) !important;
    color: #c8e6ff !important;
    border-radius: 4px !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 14px !important;
    box-shadow: inset 0 0 10px rgba(0,120,255,0.05) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.25), inset 0 0 10px rgba(0,212,255,0.05) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(0,120,255,0.3);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    font-weight: 600;
    color: rgba(0,180,255,0.5);
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 4px 4px 0 0;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    background: rgba(0,120,255,0.1) !important;
    border-bottom: 2px solid #00d4ff !important;
    text-shadow: 0 0 10px rgba(0,212,255,0.8) !important;
}

/* ── Cards ── */
.cyber-card {
    background: linear-gradient(135deg, rgba(0,20,50,0.9), rgba(0,10,30,0.95));
    border: 1px solid rgba(0,120,255,0.3);
    border-radius: 6px;
    padding: 20px 24px;
    margin: 10px 0;
    position: relative;
    box-shadow: 0 4px 20px rgba(0,80,255,0.1), inset 0 1px 0 rgba(0,212,255,0.1);
}
.cyber-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00d4ff, #0056ff);
    border-radius: 6px 0 0 6px;
}

/* ── Metric boxes ── */
.metric-box {
    background: linear-gradient(135deg, rgba(0,20,50,0.9), rgba(0,10,30,0.95));
    border: 1px solid rgba(0,120,255,0.3);
    border-radius: 6px;
    padding: 18px 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,80,255,0.1);
}
.metric-box::after {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(0,120,255,0.05) 0%, transparent 60%);
    animation: pulse-bg 3s ease-in-out infinite;
}
@keyframes pulse-bg {
    0%, 100% { opacity: 0.5; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1.1); }
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 24px;
    font-weight: 700;
    color: #00d4ff;
    display: block;
    text-shadow: 0 0 10px rgba(0,212,255,0.6);
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(0,180,255,0.6);
    margin-top: 4px;
    display: block;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Chat bubbles ── */
.bubble-user {
    background: linear-gradient(135deg, #003d99, #0056cc);
    color: #fff;
    border-radius: 12px 12px 2px 12px;
    padding: 14px 20px;
    max-width: 70%;
    font-size: 14px;
    line-height: 1.7;
    border: 1px solid rgba(0,150,255,0.4);
    box-shadow: 0 4px 20px rgba(0,80,255,0.3), 0 0 30px rgba(0,80,255,0.1);
    position: relative;
}
.bubble-bot {
    background: linear-gradient(135deg, rgba(0,20,50,0.95), rgba(0,15,40,0.98));
    border: 1px solid rgba(0,120,255,0.3);
    color: #c8e6ff;
    border-radius: 12px 12px 12px 2px;
    padding: 14px 20px;
    max-width: 80%;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0 4px 20px rgba(0,50,150,0.2);
    position: relative;
}
.bubble-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    color: rgba(0,180,255,0.6);
}
.source-pill {
    display: inline-block;
    background: rgba(0,80,200,0.15);
    border: 1px solid rgba(0,150,255,0.3);
    color: #00d4ff;
    border-radius: 3px;
    padding: 2px 10px;
    font-size: 10px;
    font-family: 'Share Tech Mono', monospace;
    margin: 5px 3px 0;
    letter-spacing: 0.5px;
}

/* ── Step rows ── */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(0,80,200,0.15);
}
.step-num {
    width: 26px; height: 26px;
    border: 1px solid #00d4ff;
    border-radius: 3px;
    color: #00d4ff;
    font-size: 11px;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-family: 'Orbitron', monospace;
    box-shadow: 0 0 8px rgba(0,212,255,0.3);
}
.step-num-done {
    width: 26px; height: 26px;
    background: rgba(0,212,255,0.15);
    border: 1px solid #00d4ff;
    border-radius: 3px;
    color: #00d4ff;
    font-size: 11px;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-family: 'Orbitron', monospace;
    box-shadow: 0 0 15px rgba(0,212,255,0.5);
    text-shadow: 0 0 8px rgba(0,212,255,0.8);
}
.step-title { font-size: 12px; font-weight: 600; color: #c8e6ff; font-family: 'Exo 2', sans-serif; }
.step-desc { font-size: 11px; color: rgba(100,180,255,0.6); font-family: 'Share Tech Mono', monospace; margin-top: 2px; }
.divider { border: none; border-top: 1px solid rgba(0,80,200,0.2); margin: 16px 0; }

/* ── Scanning animation for processing ── */
@keyframes scan {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}

/* ── Glitch text effect ── */
@keyframes glitch {
    0%, 100% { text-shadow: 0 0 10px rgba(0,212,255,0.8); }
    25% { text-shadow: -2px 0 rgba(255,0,100,0.5), 2px 0 rgba(0,255,200,0.5); }
    75% { text-shadow: 2px 0 rgba(255,0,100,0.5), -2px 0 rgba(0,255,200,0.5); }
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(0,20,50,0.5) !important;
    border: 1px dashed rgba(0,120,255,0.4) !important;
    border-radius: 6px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #020b18; }
::-webkit-scrollbar-thumb { background: rgba(0,120,255,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4ff; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(0,20,50,0.6) !important;
    border: 1px solid rgba(0,80,200,0.3) !important;
    color: #00d4ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── API Key ────────────────────────────────────────────────────────────────────
api_key = st.secrets["GROQ_API_KEY"]

# ── Session state ──────────────────────────────────────────────────────────────
_defaults = {
    "vectorstore":  None,
    "qa_chain":     None,
    "chat_history": [],
    "doc_meta":     None,
    "summary":      None,
    "raw_docs":     None,
    "all_chunks":   [],
    "processing":   False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px;">
        <div style="font-family:'Orbitron',monospace;font-size:18px;font-weight:900;
                    color:#00d4ff;text-shadow:0 0 15px rgba(0,212,255,0.7);
                    letter-spacing:3px;">
            ⚡ DOCMIND
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                    color:rgba(0,180,255,0.5);letter-spacing:2px;margin-top:4px;">
            AI DOCUMENT ANALYSIS SYSTEM v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                color:rgba(0,180,255,0.6);text-transform:uppercase;
                letter-spacing:1px;margin-bottom:8px;">
        📂 Upload Document
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "upload", type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        if st.button("⚡ INITIALIZE DOCUMENT", use_container_width=True):
            with st.spinner("🔄 Extracting text..."):
                ensure_dir("data/uploaded_docs")
                save_path = f"data/uploaded_docs/{uploaded_file.name}"
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                raw_docs = load_document(save_path)
                full_text = " ".join(d.page_content for d in raw_docs)

            with st.spinner("🔄 Chunking..."):
                chunks = split_text(raw_docs)

            with st.spinner("🔄 Generating embeddings..."):
                embeddings = create_embeddings()

            with st.spinner("🔄 Building vector index..."):
                vs = create_vector_db(chunks, embeddings)
                qa = create_qa_chain(vs, api_key)

            st.session_state.vectorstore  = vs
            st.session_state.qa_chain     = qa
            st.session_state.raw_docs     = raw_docs
            st.session_state.all_chunks   = chunks
            st.session_state.chat_history = []
            st.session_state.summary      = None
            st.session_state.doc_meta     = {
                "name":      uploaded_file.name,
                "size":      format_file_size(uploaded_file.size),
                "words":     f"{count_words(full_text):,}",
                "read_time": estimate_reading_time(full_text),
                "chunks":    len(chunks),
                "pages":     len(raw_docs),
            }
            st.success("✅ SYSTEM READY")

    if st.session_state.doc_meta:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        m = st.session_state.doc_meta
        ext = m["name"].rsplit(".", 1)[-1].upper()
        icon = {"PDF": "📄", "DOCX": "📝", "TXT": "📃"}.get(ext, "📄")
        st.markdown(f"""
        <div class="cyber-card" style="padding:12px 16px;">
            <div style="font-family:'Orbitron',monospace;font-size:11px;
                        color:#00d4ff;margin-bottom:8px;letter-spacing:1px;">
                {icon} LOADED FILE
            </div>
            <div style="font-size:12px;color:#c8e6ff;margin-bottom:6px;
                        font-family:'Exo 2',sans-serif;word-break:break-all;">
                {m['name']}
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                        color:rgba(0,180,255,0.6);line-height:1.9;">
                SIZE: {m['size']}<br>
                WORDS: {m['words']}<br>
                PAGES: {m['pages']}<br>
                CHUNKS: {m['chunks']}<br>
                READ TIME: {m['read_time']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🗑 CLEAR DOCUMENT", use_container_width=True):
            for k, v in _defaults.items():
                st.session_state[k] = v
            st.rerun()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                color:rgba(0,180,255,0.5);text-transform:uppercase;
                letter-spacing:1px;margin-bottom:8px;">
        RAG Pipeline Status
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("Upload Document",   "PDF · DOCX · TXT"),
        ("Text Extraction",   "LangChain loaders"),
        ("Text Chunking",     "600-char chunks"),
        ("Embeddings",        "MiniLM-L6-v2"),
        ("Vector Index",      "FAISS similarity"),
        ("User Question",     "Natural language"),
        ("Similarity Search", "Top-3 chunks"),
        ("LLM Answer",        "Groq LLaMA 3.3"),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        done = st.session_state.qa_chain is not None and i <= 5
        num_style = "step-num-done" if done else "step-num"
        icon = "✓" if done else str(i)
        st.markdown(f"""
        <div class="step-row">
            <div class="{num_style}">{icon}</div>
            <div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:8px;">
    <h1 style="font-size:42px;margin:0;">⚡ DOCMIND AI</h1>
    <div style="font-family:'Share Tech Mono',monospace;font-size:12px;
                color:rgba(0,180,255,0.5);letter-spacing:2px;margin-top:4px;">
        INTELLIGENT DOCUMENT ANALYSIS SYSTEM // POWERED BY RAG + GROQ
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid rgba(0,120,255,0.3);margin:12px 0 24px;'>", unsafe_allow_html=True)

# ── Metrics ────────────────────────────────────────────────────────────────────
if st.session_state.doc_meta:
    m = st.session_state.doc_meta
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, m["words"],     "TOTAL WORDS"),
        (c2, m["chunks"],    "CHUNKS"),
        (c3, len(st.session_state.chat_history), "MESSAGES"),
        (c4, "ONLINE",       "SYSTEM"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <span class="metric-value">{val}</span>
                <span class="metric-label">{label}</span>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_qa, tab_summary, tab_insights, tab_chunks = st.tabs([
    "⚡  Q&A INTERFACE",
    "📋  SUMMARY",
    "🔍  KEY INSIGHTS",
    "🗂  CHUNK EXPLORER",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Q&A
# ═══════════════════════════════════════════════════════════════════════════════
with tab_qa:
    if not st.session_state.qa_chain:
        st.markdown("""
        <div class="cyber-card" style="text-align:center;padding:60px 40px;">
            <div style="font-size:56px;margin-bottom:20px;">⚡</div>
            <div style="font-family:'Orbitron',monospace;font-size:18px;font-weight:700;
                        color:#00d4ff;margin-bottom:12px;letter-spacing:2px;">
                SYSTEM STANDBY
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:13px;
                        color:rgba(0,180,255,0.5);">
                Upload a document and click INITIALIZE to activate the AI engine
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                        color:rgba(0,180,255,0.5);margin-bottom:12px;letter-spacing:1px;">
                ▶ SELECT A QUERY OR TYPE YOUR OWN:
            </div>""", unsafe_allow_html=True)
            suggestions = [
                "What is the main topic?",
                "Summarize the document.",
                "What are the key points?",
                "What methodology is used?",
                "What are the conclusions?",
                "What challenges are mentioned?",
            ]
            cols = st.columns(3)
            for i, s in enumerate(suggestions):
                if cols[i % 3].button(s, key=f"sug_{i}", use_container_width=True):
                    st.session_state._pending_q = s
                    st.rerun()

        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="display:flex;justify-content:flex-end;margin:12px 0;">
                        <div>
                            <div class="bubble-label" style="text-align:right;">
                                [ USER INPUT ]
                            </div>
                            <div class="bubble-user">{msg['content']}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    src_pills = ""
                    for i, src in enumerate(msg.get("sources", [])[:3]):
                        page = src.metadata.get("page", "")
                        label = f"SRC-{i+1}" + (f" P.{page}" if page else "")
                        src_pills += f'<span class="source-pill">📌 {label}</span>'
                    st.markdown(f"""
                    <div style="display:flex;justify-content:flex-start;margin:12px 0;">
                        <div>
                            <div class="bubble-label">[ DOCMIND AI RESPONSE ]</div>
                            <div class="bubble-bot">{msg['content']}
                                {('<div style="margin-top:8px;">' + src_pills + '</div>') if src_pills else ''}
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        col_q, col_btn = st.columns([5, 1])
        with col_q:
            user_q = st.text_input(
                "q", key="q_input",
                placeholder="// Enter query...",
                label_visibility="collapsed",
            )
        with col_btn:
            send = st.button("SEND", use_container_width=True)

        if hasattr(st.session_state, "_pending_q"):
            user_q = st.session_state._pending_q
            del st.session_state._pending_q
            send = True

        if send and user_q.strip():
            q = user_q.strip()
            with st.spinner("🔄 Processing query..."):
                result = ask_question(st.session_state.qa_chain, q)
            st.session_state.chat_history.append({"role": "user", "content": q, "sources": []})
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑 CLEAR SESSION"):
                st.session_state.chat_history = []
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    if not st.session_state.raw_docs:
        st.info("Initialize a document first.")
    else:
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                    color:rgba(0,180,255,0.5);margin-bottom:16px;letter-spacing:1px;">
            ▶ GENERATE FULL DOCUMENT ANALYSIS REPORT
        </div>""", unsafe_allow_html=True)

        if st.button("⚡ GENERATE SUMMARY REPORT", use_container_width=False):
            with st.spinner("🔄 Analyzing document... This may take 60-90 seconds for large files."):
                summary = summarize_document(st.session_state.raw_docs, api_key)
                st.session_state.summary = summary

        if st.session_state.summary:
            st.markdown(f"""
            <div class="cyber-card" style="padding:28px 32px;">
                <div style="font-family:'Orbitron',monospace;font-size:12px;
                            color:#00d4ff;margin-bottom:16px;letter-spacing:2px;">
                    ── ANALYSIS REPORT ──
                </div>
                <div style="font-family:'Exo 2',sans-serif;font-size:14px;
                            line-height:1.8;color:#c8e6ff;">
                    {st.session_state.summary.replace(chr(10), '<br>')}
                </div>
            </div>""", unsafe_allow_html=True)
            st.download_button(
                "⬇ EXPORT REPORT (.txt)",
                data=st.session_state.summary,
                file_name=f"report_{st.session_state.doc_meta['name']}.txt",
                mime="text/plain",
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — KEY INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    if not st.session_state.qa_chain:
        st.info("Initialize a document first.")
    else:
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                    color:rgba(0,180,255,0.5);margin-bottom:16px;letter-spacing:1px;">
            ▶ SELECT INSIGHT MODULES TO EXTRACT:
        </div>""", unsafe_allow_html=True)

        insight_questions = {
            "🎯 MAIN TOPIC":      "What is the main topic or purpose of this document?",
            "🔑 KEY POINTS":      "List the most important key points discussed in this document.",
            "📊 DATA & FINDINGS": "What data, statistics, or findings are presented?",
            "⚠ CHALLENGES":      "What challenges, limitations, or problems are discussed?",
            "✅ CONCLUSIONS":     "What are the main conclusions or recommendations?",
            "🔬 METHODOLOGY":     "What methodology, approach, or techniques are described?",
        }
        col1, col2 = st.columns(2)
        selected = {}
        for i, (label, _) in enumerate(insight_questions.items()):
            col = col1 if i % 2 == 0 else col2
            selected[label] = col.checkbox(label, value=True, key=f"chk_{i}")

        if st.button("⚡ EXTRACT INSIGHTS", use_container_width=False):
            chosen = {k: v for k, v in insight_questions.items() if selected.get(k)}
            if not chosen:
                st.warning("Select at least one module.")
            else:
                for label, question in chosen.items():
                    with st.spinner(f"🔄 Extracting {label}..."):
                        result = ask_question(st.session_state.qa_chain, question)
                    st.markdown(f"""
                    <div class="cyber-card">
                        <div style="font-family:'Orbitron',monospace;font-weight:700;
                                    font-size:12px;margin-bottom:10px;color:#00d4ff;
                                    letter-spacing:1px;">{label}</div>
                        <div style="font-size:14px;line-height:1.7;color:#c8e6ff;
                                    font-family:'Exo 2',sans-serif;">
                            {result['answer']}
                        </div>
                    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHUNKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chunks:
    if not st.session_state.all_chunks:
        st.info("Initialize a document first.")
    else:
        chunks = st.session_state.all_chunks
        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                    color:rgba(0,180,255,0.5);margin-bottom:16px;letter-spacing:1px;">
            ▶ VECTOR INDEX: <span style="color:#00d4ff;">{len(chunks)} CHUNKS</span> LOADED
        </div>""", unsafe_allow_html=True)

        search = st.text_input("🔎 Filter by keyword", placeholder="// search chunks...")
        filtered = [c for c in chunks if not search or search.lower() in c.page_content.lower()]
        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                    color:rgba(0,180,255,0.4);margin-bottom:8px;">
            SHOWING {min(15, len(filtered))} OF {len(filtered)} RESULTS
        </div>""", unsafe_allow_html=True)

        for i, chunk in enumerate(filtered[:15]):
            page = chunk.metadata.get("page", "—")
            src  = chunk.metadata.get("source", "document")
            with st.expander(f"CHUNK-{i+1:03d}  //  {len(chunk.page_content)} chars  //  {src}  //  page {page}"):
                st.code(chunk.page_content, language=None)
