"""
app.py
------
DocMind AI — Intelligent Document Analysis Agent
Burgundy Minimal Elegant UI
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
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=Jost:wght@300;400;500;600&family=Courier+Prime:wght@400;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    background-color: #1a0a0a;
    color: #e8d5c4;
}
.main {
    background:
        radial-gradient(ellipse at top left, rgba(120,20,40,0.15) 0%, transparent 50%),
        radial-gradient(ellipse at bottom right, rgba(80,10,20,0.2) 0%, transparent 50%),
        #1a0a0a;
}
.block-container { padding: 2.5rem 3rem 5rem; max-width: 1150px; }

/* ── Typography ── */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 700 !important;
    font-style: italic !important;
    letter-spacing: 1px !important;
    color: #c8a882 !important;
}
h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600 !important;
    color: #c8a882 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #120606 0%, #1a0808 50%, #160606 100%) !important;
    border-right: 1px solid rgba(160,60,60,0.2) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.8rem 1.4rem; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    color: #c8a882 !important;
    border: 1px solid rgba(160,80,60,0.5) !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    padding: 10px 24px !important;
    text-transform: uppercase !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: rgba(120,30,30,0.3) !important;
    border-color: #c8a882 !important;
    color: #e8d5c4 !important;
    box-shadow: 0 2px 20px rgba(120,30,30,0.2) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(30,8,8,0.8) !important;
    border: 1px solid rgba(140,60,50,0.35) !important;
    color: #e8d5c4 !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 14px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(180,100,70,0.6) !important;
    box-shadow: 0 0 0 2px rgba(120,30,30,0.15) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(140,60,50,0.25);
    gap: 0px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Jost', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: rgba(180,120,100,0.5);
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 12px 24px;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #c8a882 !important;
    background: transparent !important;
    border-bottom: 1px solid #c8a882 !important;
}

/* ── Cards ── */
.burg-card {
    background: rgba(25,8,8,0.7);
    border: 1px solid rgba(140,60,50,0.2);
    border-radius: 3px;
    padding: 24px 28px;
    margin: 12px 0;
    position: relative;
}
.burg-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 24px; right: 24px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(160,80,60,0.3), transparent);
}

/* ── Metric boxes ── */
.metric-box {
    background: rgba(25,8,8,0.6);
    border: 1px solid rgba(140,60,50,0.2);
    border-top: 2px solid rgba(160,80,60,0.5);
    border-radius: 2px;
    padding: 20px 12px;
    text-align: center;
}
.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    font-weight: 600;
    color: #c8a882;
    display: block;
    line-height: 1;
}
.metric-label {
    font-family: 'Jost', sans-serif;
    font-size: 9px;
    color: rgba(180,120,100,0.5);
    margin-top: 6px;
    display: block;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ── Chat bubbles ── */
.bubble-user {
    background: linear-gradient(135deg, rgba(100,20,20,0.6), rgba(80,15,15,0.7));
    color: #e8d5c4;
    border-radius: 12px 12px 2px 12px;
    padding: 14px 20px;
    max-width: 68%;
    font-size: 14px;
    line-height: 1.75;
    border: 1px solid rgba(160,80,60,0.25);
    font-family: 'Jost', sans-serif;
}
.bubble-bot {
    background: rgba(22,7,7,0.8);
    border: 1px solid rgba(140,60,50,0.2);
    color: #e8d5c4;
    border-radius: 12px 12px 12px 2px;
    padding: 14px 20px;
    max-width: 80%;
    font-size: 14px;
    line-height: 1.75;
    font-family: 'Jost', sans-serif;
}
.bubble-label {
    font-family: 'Jost', sans-serif;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
    color: rgba(180,120,100,0.45);
}
.source-pill {
    display: inline-block;
    background: rgba(100,25,25,0.3);
    border: 1px solid rgba(160,80,60,0.25);
    color: #c8a882;
    border-radius: 2px;
    padding: 2px 10px;
    font-size: 10px;
    font-family: 'Courier Prime', monospace;
    margin: 5px 3px 0;
    letter-spacing: 0.5px;
}

/* ── Step rows ── */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(100,30,30,0.15);
}
.step-num {
    width: 24px; height: 24px;
    border: 1px solid rgba(160,80,60,0.4);
    border-radius: 50%;
    color: rgba(180,120,100,0.5);
    font-size: 10px;
    font-weight: 600;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-family: 'Courier Prime', monospace;
}
.step-num-done {
    width: 24px; height: 24px;
    background: rgba(120,30,30,0.4);
    border: 1px solid rgba(180,100,70,0.6);
    border-radius: 50%;
    color: #c8a882;
    font-size: 10px;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-family: 'Courier Prime', monospace;
}
.step-title {
    font-size: 12px;
    font-weight: 500;
    color: #e8d5c4;
    font-family: 'Jost', sans-serif;
    letter-spacing: 0.5px;
}
.step-desc {
    font-size: 10px;
    color: rgba(180,120,100,0.4);
    font-family: 'Courier Prime', monospace;
    margin-top: 2px;
}
.divider {
    border: none;
    border-top: 1px solid rgba(120,40,40,0.2);
    margin: 18px 0;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(25,8,8,0.5) !important;
    border: 1px dashed rgba(140,60,50,0.3) !important;
    border-radius: 3px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #1a0a0a; }
::-webkit-scrollbar-thumb {
    background: rgba(140,60,50,0.4);
    border-radius: 2px;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(25,8,8,0.6) !important;
    border: 1px solid rgba(120,40,40,0.2) !important;
    color: #c8a882 !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 11px !important;
    border-radius: 2px !important;
}

/* ── Ornament line ── */
.ornament {
    text-align: center;
    color: rgba(160,80,60,0.4);
    font-size: 16px;
    letter-spacing: 8px;
    margin: 8px 0;
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
    <div style="padding:8px 0 20px;">
        <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                    font-weight:700;font-style:italic;color:#c8a882;
                    letter-spacing:1px;">
            DocMind AI
        </div>
        <div style="font-family:'Jost',sans-serif;font-size:10px;
                    color:rgba(180,120,100,0.4);letter-spacing:3px;
                    text-transform:uppercase;margin-top:4px;">
            Document Intelligence
        </div>
        <div class="ornament" style="margin-top:12px;">— ✦ —</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Jost',sans-serif;font-size:10px;
                color:rgba(180,120,100,0.45);text-transform:uppercase;
                letter-spacing:2px;margin-bottom:10px;">
        Upload Document
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "upload", type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        if st.button("Process Document", use_container_width=True):
            with st.spinner("Extracting text..."):
                ensure_dir("data/uploaded_docs")
                save_path = f"data/uploaded_docs/{uploaded_file.name}"
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                raw_docs = load_document(save_path)
                full_text = " ".join(d.page_content for d in raw_docs)

            with st.spinner("Chunking..."):
                chunks = split_text(raw_docs)

            with st.spinner("Generating embeddings..."):
                embeddings = create_embeddings()

            with st.spinner("Building index..."):
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
            st.success("Document ready.")

    if st.session_state.doc_meta:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        m = st.session_state.doc_meta
        ext = m["name"].rsplit(".", 1)[-1].upper()
        icon = {"PDF": "📄", "DOCX": "📝", "TXT": "📃"}.get(ext, "📄")
        st.markdown(f"""
        <div class="burg-card" style="padding:14px 18px;">
            <div style="font-family:'Jost',sans-serif;font-size:9px;
                        color:rgba(180,120,100,0.45);text-transform:uppercase;
                        letter-spacing:2px;margin-bottom:8px;">
                {icon} Active Document
            </div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:14px;
                        color:#c8a882;margin-bottom:10px;font-style:italic;">
                {m['name']}
            </div>
            <div style="font-family:'Courier Prime',monospace;font-size:10px;
                        color:rgba(180,120,100,0.45);line-height:2;">
                Size &nbsp;&nbsp;· {m['size']}<br>
                Words · {m['words']}<br>
                Pages · {m['pages']}<br>
                Chunks· {m['chunks']}<br>
                Read &nbsp;&nbsp;· {m['read_time']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Remove Document", use_container_width=True):
            for k, v in _defaults.items():
                st.session_state[k] = v
            st.rerun()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Jost',sans-serif;font-size:9px;
                color:rgba(180,120,100,0.4);text-transform:uppercase;
                letter-spacing:2px;margin-bottom:10px;">
        Pipeline
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
<div style="margin-bottom:6px;">
    <h1 style="font-size:48px;margin:0;line-height:1.1;">DocMind AI</h1>
    <div style="font-family:'Jost',sans-serif;font-size:12px;
                color:rgba(180,120,100,0.45);letter-spacing:3px;
                text-transform:uppercase;margin-top:6px;">
        Intelligent Document Analysis
    </div>
</div>
<div class="ornament" style="text-align:left;margin:14px 0 20px;">
    ─── ✦ ──────────────────────────────────────
</div>
""", unsafe_allow_html=True)

# ── Metrics ────────────────────────────────────────────────────────────────────
if st.session_state.doc_meta:
    m = st.session_state.doc_meta
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, m["words"],     "Words"),
        (c2, m["chunks"],    "Chunks"),
        (c3, len(st.session_state.chat_history), "Messages"),
        (c4, "Ready",        "Status"),
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
    "Q & A",
    "Summary",
    "Key Insights",
    "Chunks",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Q&A
# ═══════════════════════════════════════════════════════════════════════════════
with tab_qa:
    if not st.session_state.qa_chain:
        st.markdown("""
        <div class="burg-card" style="text-align:center;padding:70px 40px;">
            <div style="font-family:'Cormorant Garamond',serif;font-size:48px;
                        color:rgba(160,80,60,0.3);margin-bottom:20px;">📜</div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                        font-style:italic;color:#c8a882;margin-bottom:10px;">
                No Document Loaded
            </div>
            <div style="font-family:'Jost',sans-serif;font-size:13px;
                        color:rgba(180,120,100,0.4);letter-spacing:1px;">
                Upload a document from the sidebar to begin
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="font-family:'Jost',sans-serif;font-size:10px;
                        color:rgba(180,120,100,0.4);margin-bottom:14px;
                        text-transform:uppercase;letter-spacing:2px;">
                Suggested Questions
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
                    <div style="display:flex;justify-content:flex-end;margin:14px 0;">
                        <div>
                            <div class="bubble-label" style="text-align:right;">You</div>
                            <div class="bubble-user">{msg['content']}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    src_pills = ""
                    for i, src in enumerate(msg.get("sources", [])[:3]):
                        page = src.metadata.get("page", "")
                        label = f"Source {i+1}" + (f" · p.{page}" if page else "")
                        src_pills += f'<span class="source-pill">· {label}</span>'
                    st.markdown(f"""
                    <div style="display:flex;justify-content:flex-start;margin:14px 0;">
                        <div>
                            <div class="bubble-label">DocMind</div>
                            <div class="bubble-bot">{msg['content']}
                                {('<div style="margin-top:10px;">' + src_pills + '</div>') if src_pills else ''}
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        col_q, col_btn = st.columns([5, 1])
        with col_q:
            user_q = st.text_input(
                "q", key="q_input",
                placeholder="Ask anything about your document...",
                label_visibility="collapsed",
            )
        with col_btn:
            send = st.button("Ask", use_container_width=True)

        if hasattr(st.session_state, "_pending_q"):
            user_q = st.session_state._pending_q
            del st.session_state._pending_q
            send = True

        if send and user_q.strip():
            q = user_q.strip()
            with st.spinner("Thinking..."):
                result = ask_question(st.session_state.qa_chain, q)
            st.session_state.chat_history.append({"role": "user", "content": q, "sources": []})
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear Conversation"):
                st.session_state.chat_history = []
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    if not st.session_state.raw_docs:
        st.info("Upload and process a document first.")
    else:
        st.markdown("""
        <div style="font-family:'Jost',sans-serif;font-size:10px;
                    color:rgba(180,120,100,0.4);margin-bottom:20px;
                    text-transform:uppercase;letter-spacing:2px;">
            Generate a structured summary of your document
        </div>""", unsafe_allow_html=True)

        if st.button("Generate Summary", use_container_width=False):
            with st.spinner("Analysing document..."):
                summary = summarize_document(st.session_state.raw_docs, api_key)
                st.session_state.summary = summary

        if st.session_state.summary:
            st.markdown(f"""
            <div class="burg-card" style="padding:32px 36px;">
                <div style="font-family:'Cormorant Garamond',serif;font-size:13px;
                            color:rgba(180,100,70,0.5);letter-spacing:3px;
                            text-transform:uppercase;margin-bottom:20px;">
                    ── Document Summary ──
                </div>
                <div style="font-family:'Jost',sans-serif;font-size:14px;
                            line-height:1.9;color:#e8d5c4;">
                    {st.session_state.summary.replace(chr(10), '<br>')}
                </div>
            </div>""", unsafe_allow_html=True)
            st.download_button(
                "Download Summary",
                data=st.session_state.summary,
                file_name=f"summary_{st.session_state.doc_meta['name']}.txt",
                mime="text/plain",
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — KEY INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    if not st.session_state.qa_chain:
        st.info("Upload and process a document first.")
    else:
        st.markdown("""
        <div style="font-family:'Jost',sans-serif;font-size:10px;
                    color:rgba(180,120,100,0.4);margin-bottom:20px;
                    text-transform:uppercase;letter-spacing:2px;">
            Select insight categories to extract
        </div>""", unsafe_allow_html=True)

        insight_questions = {
            "Main Topic":      "What is the main topic or purpose of this document?",
            "Key Points":      "List the most important key points discussed in this document.",
            "Data & Findings": "What data, statistics, or findings are presented?",
            "Challenges":      "What challenges, limitations, or problems are discussed?",
            "Conclusions":     "What are the main conclusions or recommendations?",
            "Methodology":     "What methodology, approach, or techniques are described?",
        }
        col1, col2 = st.columns(2)
        selected = {}
        for i, (label, _) in enumerate(insight_questions.items()):
            col = col1 if i % 2 == 0 else col2
            selected[label] = col.checkbox(label, value=True, key=f"chk_{i}")

        if st.button("Extract Insights", use_container_width=False):
            chosen = {k: v for k, v in insight_questions.items() if selected.get(k)}
            if not chosen:
                st.warning("Select at least one category.")
            else:
                for label, question in chosen.items():
                    with st.spinner(f"Extracting {label}..."):
                        result = ask_question(st.session_state.qa_chain, question)
                    st.markdown(f"""
                    <div class="burg-card">
                        <div style="font-family:'Cormorant Garamond',serif;
                                    font-size:16px;font-style:italic;font-weight:600;
                                    margin-bottom:12px;color:#c8a882;">
                            {label}
                        </div>
                        <div style="font-size:14px;line-height:1.8;color:#e8d5c4;
                                    font-family:'Jost',sans-serif;">
                            {result['answer']}
                        </div>
                    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHUNKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chunks:
    if not st.session_state.all_chunks:
        st.info("Upload and process a document first.")
    else:
        chunks = st.session_state.all_chunks
        st.markdown(f"""
        <div style="font-family:'Jost',sans-serif;font-size:10px;
                    color:rgba(180,120,100,0.4);margin-bottom:16px;
                    text-transform:uppercase;letter-spacing:2px;">
            {len(chunks)} chunks indexed · Search and explore below
        </div>""", unsafe_allow_html=True)

        search = st.text_input("Search chunks", placeholder="Filter by keyword...")
        filtered = [c for c in chunks if not search or search.lower() in c.page_content.lower()]
        st.markdown(f"""
        <div style="font-family:'Courier Prime',monospace;font-size:10px;
                    color:rgba(180,120,100,0.35);margin-bottom:10px;">
            Showing {min(15, len(filtered))} of {len(filtered)} results
        </div>""", unsafe_allow_html=True)

        for i, chunk in enumerate(filtered[:15]):
            page = chunk.metadata.get("page", "—")
            src  = chunk.metadata.get("source", "document")
            with st.expander(f"Chunk {i+1}  ·  {len(chunk.page_content)} chars  ·  page {page}"):
                st.text(chunk.page_content)
