"""
app.py
------
DocMind AI — Intelligent Document Analysis Agent
Streamlit front-end that ties together all pipeline modules.

Run:
    streamlit run app.py
"""

import os
import time
import streamlit as st

from modules.loader     import load_document
from modules.chunking   import split_text
from modules.embeddings import create_embeddings
from modules.vectorstore import create_vector_db
from modules.qa_system  import create_qa_chain, ask_question
from modules.summarizer import summarize_document
from utils.helpers      import (
    format_file_size, estimate_reading_time,
    count_words, ensure_dir,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI — Document Analysis Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #07070f;
    color: #dde1ec;
}
.main { background: #07070f; }
.block-container { padding: 2rem 2.5rem 5rem; max-width: 1100px; }
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; letter-spacing: -1px !important; }
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }
[data-testid="stSidebar"] { background: #0f0f1a !important; border-right: 1px solid #1e1e30; }
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #4338ca) !important;
    color: #fff !important; border: none !important; border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    padding: 9px 22px !important; transition: opacity 0.15s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: #111120 !important; border: 1px solid #252540 !important;
    color: #dde1ec !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 14px !important;
}
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1e1e30; gap: 4px; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Sans', sans-serif; font-weight: 500; color: #6b7280; border-radius: 8px 8px 0 0; padding: 8px 18px; }
.stTabs [aria-selected="true"] { color: #a78bfa !important; background: rgba(109,40,217,0.08) !important; border-bottom: 2px solid #7c3aed !important; }
.card { background: #111120; border: 1px solid #1e1e30; border-radius: 14px; padding: 20px 24px; margin: 10px 0; }
.metric-box { flex: 1; background: #111120; border: 1px solid #1e1e30; border-radius: 12px; padding: 16px 12px; text-align: center; }
.metric-value { font-family: 'DM Mono', monospace; font-size: 22px; font-weight: 700; color: #a78bfa; display: block; }
.metric-label { font-size: 11px; color: #6b7280; margin-top: 4px; display: block; font-family: 'DM Mono', monospace; text-transform: uppercase; letter-spacing: 0.5px; }
.chat-wrap { display: flex; flex-direction: column; gap: 14px; }
.bubble-user { align-self: flex-end; background: linear-gradient(135deg, #6d28d9, #4338ca); color: #fff; border-radius: 18px 18px 4px 18px; padding: 13px 18px; max-width: 72%; font-size: 14px; line-height: 1.65; box-shadow: 0 4px 20px rgba(109,40,217,0.25); }
.bubble-bot { align-self: flex-start; background: #111120; border: 1px solid #1e1e30; color: #dde1ec; border-radius: 18px 18px 18px 4px; padding: 13px 18px; max-width: 82%; font-size: 14px; line-height: 1.65; }
.bubble-label { font-size: 10px; font-family: 'DM Mono', monospace; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 5px; color: #6b7280; }
.source-pill { display: inline-block; background: rgba(109,40,217,0.12); border: 1px solid rgba(109,40,217,0.25); color: #a78bfa; border-radius: 20px; padding: 2px 10px; font-size: 11px; font-family: 'DM Mono', monospace; margin: 5px 3px 0; }
.step-row { display: flex; align-items: flex-start; gap: 14px; padding: 10px 0; border-bottom: 1px solid #1a1a2e; }
.step-num { width: 28px; height: 28px; background: linear-gradient(135deg, #6d28d9, #4338ca); border-radius: 50%; color: #fff; font-size: 12px; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-family: 'DM Mono', monospace; }
.step-text { font-size: 13px; color: #9ca3af; padding-top: 4px; }
.divider { border: none; border-top: 1px solid #1e1e30; margin: 18px 0; }
</style>
""", unsafe_allow_html=True)

# ── API Key (hardcoded) ────────────────────────────────────────────────────────
api_key = st.secrets["GROQ_API_KEY"]

# ── Session state init ─────────────────────────────────────────────────────────
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
    st.markdown(
        "<h2 style='font-family:Syne,sans-serif;font-size:20px;margin:0 0 2px;'>🧠 DocMind AI</h2>"
        "<p style='font-size:11px;color:#6b7280;font-family:DM Mono,monospace;margin:0 0 20px;'>"
        "Document Analysis Agent</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── File upload ────────────────────────────────────────────────────────────
    st.markdown("#### 📂 Upload Document")
    uploaded_file = st.file_uploader(
        "upload",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
        help="Supported: PDF · DOCX · TXT",
    )

    if uploaded_file:
        if st.button("⚡ Process Document", use_container_width=True):
            with st.spinner("Step 1/5 — Saving file…"):
                ensure_dir("data/uploaded_docs")
                save_path = f"data/uploaded_docs/{uploaded_file.name}"
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            with st.spinner("Step 2/5 — Extracting text…"):
                raw_docs = load_document(save_path)
                full_text = " ".join(d.page_content for d in raw_docs)

            with st.spinner("Step 3/5 — Chunking text…"):
                chunks = split_text(raw_docs)

            with st.spinner("Step 4/5 — Generating embeddings…"):
                embeddings = create_embeddings()

            with st.spinner("Step 5/5 — Building vector store…"):
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
            st.success("✅ Document ready! Start asking questions.")

    # ── Loaded doc info ────────────────────────────────────────────────────────
    if st.session_state.doc_meta:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        m = st.session_state.doc_meta
        ext = m["name"].rsplit(".", 1)[-1].upper()
        icon = {"PDF": "📄", "DOCX": "📝", "TXT": "📃"}.get(ext, "📄")
        st.markdown(f"""
        <div class="card" style="padding:14px 16px;">
            <p style="font-weight:600;font-size:13px;margin:0 0 8px;color:#e2e8f0;">
                {icon} {m['name']}
            </p>
            <p style="font-size:11px;color:#6b7280;margin:0;line-height:1.8;">
                Size: <b style="color:#9ca3af">{m['size']}</b><br>
                Words: <b style="color:#9ca3af">{m['words']}</b><br>
                Pages / sections: <b style="color:#9ca3af">{m['pages']}</b><br>
                Chunks: <b style="color:#9ca3af">{m['chunks']}</b><br>
                Est. read time: <b style="color:#9ca3af">{m['read_time']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Remove Document", use_container_width=True):
            for k, v in _defaults.items():
                st.session_state[k] = v
            st.rerun()

    # ── Pipeline steps ─────────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;color:#6b7280;font-family:DM Mono,monospace;text-transform:uppercase;letter-spacing:0.5px;'>RAG Pipeline</p>", unsafe_allow_html=True)
    steps = [
        ("Upload Document",  "PDF · DOCX · TXT accepted"),
        ("Text Extraction",  "LangChain loaders extract raw text"),
        ("Text Chunking",    "Split into 600-char overlapping chunks"),
        ("Embeddings",       "HuggingFace all-MiniLM-L6-v2 model"),
        ("Vector Store",     "FAISS similarity index built"),
        ("User Question",    "Natural language query"),
        ("Similarity Search","Top-5 relevant chunks retrieved"),
        ("LLM Answer",       "GPT-3.5-turbo generates response"),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        done = st.session_state.qa_chain is not None and i <= 5
        color = "#10b981" if done else "#6d28d9"
        st.markdown(f"""
        <div class="step-row">
            <div class="step-num" style="background:{color};">{i}</div>
            <div>
                <div style="font-size:12px;font-weight:600;color:#e2e8f0;">{title}</div>
                <div class="step-text">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:4px;font-size:36px;'>🧠 DocMind AI</h1>"
    "<p style='color:#6b7280;font-size:15px;margin-bottom:28px;'>"
    "Upload any document — then ask questions, get summaries, and extract key insights using AI.</p>",
    unsafe_allow_html=True,
)

if st.session_state.doc_meta:
    m = st.session_state.doc_meta
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, m["words"],     "Total Words"),
        (c2, m["chunks"],    "Text Chunks"),
        (c3, len(st.session_state.chat_history), "Messages"),
        (c4, "✅ Ready",     "System"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <span class="metric-value">{val}</span>
                <span class="metric-label">{label}</span>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

tab_qa, tab_summary, tab_insights, tab_chunks = st.tabs([
    "💬  Q&A Chat", "📋  Summary", "🔍  Key Insights", "🗂️  Chunks Explorer",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Q&A CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_qa:
    if not st.session_state.qa_chain:
        st.markdown("""
        <div class="card" style="text-align:center;padding:40px;">
            <div style="font-size:48px;margin-bottom:16px;">📂</div>
            <h3 style="color:#e2e8f0;margin:0 0 8px;">No document loaded</h3>
            <p style="color:#6b7280;font-size:14px;margin:0;">
                Upload a PDF, DOCX, or TXT file in the sidebar and click
                <b>⚡ Process Document</b> to get started.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        if not st.session_state.chat_history:
            st.markdown(
                "<p style='font-size:13px;color:#6b7280;margin-bottom:10px;'>"
                "Try one of these questions or type your own:</p>",
                unsafe_allow_html=True,
            )
            suggestions = [
                "What is the main topic of this document?",
                "Summarize the document.",
                "What are the key points discussed?",
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
            st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="display:flex;justify-content:flex-end;">
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
                        src_pills += f'<span class="source-pill">📌 {label}</span>'
                    st.markdown(f"""
                    <div style="display:flex;justify-content:flex-start;">
                        <div>
                            <div class="bubble-label">DocMind AI</div>
                            <div class="bubble-bot">{msg['content']}{('<br>' + src_pills) if src_pills else ''}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        col_q, col_btn = st.columns([5, 1])
        with col_q:
            user_q = st.text_input(
                "question", key="q_input",
                placeholder="Ask anything about the document…",
                label_visibility="collapsed",
            )
        with col_btn:
            send = st.button("Send →", use_container_width=True)

        if hasattr(st.session_state, "_pending_q"):
            user_q = st.session_state._pending_q
            del st.session_state._pending_q
            send = True

        if send and user_q.strip():
            q = user_q.strip()
            with st.spinner("Searching document and generating answer…"):
                result = ask_question(st.session_state.qa_chain, q)
            st.session_state.chat_history.append({"role": "user", "content": q, "sources": []})
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Conversation"):
                st.session_state.chat_history = []
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    if not st.session_state.raw_docs:
        st.info("Upload and process a document first.")
    else:
        st.markdown(
            "<p style='color:#6b7280;font-size:14px;'>"
            "Generate a structured summary with overview, key points, and conclusions.</p>",
            unsafe_allow_html=True,
        )
        if st.button("✨ Generate Full Summary", use_container_width=False):
            with st.spinner("Summarizing document…"):
                summary = summarize_document(st.session_state.raw_docs, api_key)
                st.session_state.summary = summary

        if st.session_state.summary:
            st.markdown(f"""
            <div class="card" style="padding:28px 32px;">
                {st.session_state.summary.replace(chr(10), '<br>')}
            </div>""", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Summary (.txt)",
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
        st.markdown(
            "<p style='color:#6b7280;font-size:14px;'>"
            "Auto-generate structured insights about the document.</p>",
            unsafe_allow_html=True,
        )
        insight_questions = {
            "🎯 Main Topic":      "What is the main topic or purpose of this document?",
            "🔑 Key Points":      "List the most important key points discussed in this document.",
            "📊 Data & Findings": "What data, statistics, or findings are presented in this document?",
            "⚠️ Challenges":      "What challenges, limitations, or problems are discussed?",
            "✅ Conclusions":     "What are the main conclusions or recommendations of this document?",
            "🔬 Methodology":     "What methodology, approach, or techniques are described?",
        }
        col1, col2 = st.columns(2)
        selected = {}
        for i, (label, _) in enumerate(insight_questions.items()):
            col = col1 if i % 2 == 0 else col2
            selected[label] = col.checkbox(label, value=True, key=f"chk_{i}")

        if st.button("🚀 Extract Insights", use_container_width=False):
            chosen = {k: v for k, v in insight_questions.items() if selected.get(k)}
            if not chosen:
                st.warning("Select at least one insight type.")
            else:
                for label, question in chosen.items():
                    with st.spinner(f"Extracting: {label}…"):
                        result = ask_question(st.session_state.qa_chain, question)
                    st.markdown(f"""
                    <div class="card">
                        <p style="font-family:Syne,sans-serif;font-weight:700;font-size:15px;
                                  margin:0 0 10px;color:#a78bfa;">{label}</p>
                        <p style="font-size:14px;line-height:1.7;margin:0;color:#dde1ec;">
                            {result['answer']}
                        </p>
                    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHUNKS EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chunks:
    if not st.session_state.all_chunks:
        st.info("Upload and process a document first.")
    else:
        chunks = st.session_state.all_chunks
        st.markdown(
            f"<p style='color:#6b7280;font-size:14px;'>"
            f"Document split into <b style='color:#a78bfa'>{len(chunks)} chunks</b>.</p>",
            unsafe_allow_html=True,
        )
        search = st.text_input("🔎 Filter chunks by keyword", placeholder="e.g. methodology")
        filtered = [c for c in chunks if not search or search.lower() in c.page_content.lower()]
        st.markdown(
            f"<p style='font-size:12px;color:#6b7280;'>"
            f"Showing {min(15, len(filtered))} of {len(filtered)} matching chunks</p>",
            unsafe_allow_html=True,
        )
        for i, chunk in enumerate(filtered[:15]):
            page = chunk.metadata.get("page", "—")
            src  = chunk.metadata.get("source", "document")
            with st.expander(f"Chunk {i+1}   ·   {len(chunk.page_content)} chars   ·   {src}   ·   page {page}"):
                st.text(chunk.page_content)
