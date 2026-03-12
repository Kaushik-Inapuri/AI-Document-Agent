# 🧠 DocMind AI — AI-Powered Document Analysis Agent

Upload any document and interact with it through natural language.
Powered by **RAG (Retrieval-Augmented Generation)** using LangChain + OpenAI + FAISS.

---

## 🎯 Goal

Help users quickly understand large documents without reading them in full by:
- Automatically extracting and indexing document text
- Answering natural language questions grounded in document content
- Generating structured summaries and key insights
- Reducing time-to-understanding for long documents

---

## 🏗️ System Architecture

```
User Uploads Document (PDF / DOCX / TXT)
            ↓
   Text Extraction  ←─ LangChain loaders
            ↓
   Text Chunking    ←─ 600-char chunks, 120-char overlap
            ↓
   Embedding Generation ←─ HuggingFace all-MiniLM-L6-v2 (local, no API)
            ↓
   Vector Database  ←─ FAISS similarity index
            ↓
   User Asks Question
            ↓
   Similarity Search → Top-5 Relevant Chunks Retrieved
            ↓
   LLM (GPT-3.5-turbo) Generates Answer from Context
            ↓
   Response Shown to User (with source citations)
```

---

## 💡 Features

| Feature | Description |
|---------|-------------|
| 💬 Q&A Chat | Ask any question — answers sourced from document |
| 📋 Summary | Structured summary: overview + key points + conclusions |
| 🔍 Key Insights | Auto-extract topic, findings, methodology, challenges |
| 🗂️ Chunk Explorer | Inspect how the document was split; filter by keyword |
| 📌 Source Citations | Every answer shows which document sections were used |
| ⬇️ Download Summary | Save the summary as a .txt file |

---

## 🔧 Technology Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| LLM | OpenAI GPT-3.5-turbo (via LangChain) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local CPU) |
| Vector DB | FAISS |
| PDF Loader | PyPDF (LangChain) |
| DOCX Loader | Docx2txt (LangChain) |
| Framework | LangChain |

---

## 📁 Project Structure

```
AI-Document-Agent/
│
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── README.md
│
├── data/
│   └── uploaded_docs/      ← Uploaded files stored here
│
├── modules/
│   ├── loader.py           ← PDF / DOCX / TXT text extraction
│   ├── chunking.py         ← Split into overlapping chunks
│   ├── embeddings.py       ← HuggingFace embedding model
│   ├── vectorstore.py      ← FAISS vector DB (create, save, load)
│   ├── qa_system.py        ← RAG QA chain with custom grounding prompt
│   └── summarizer.py       ← Map-reduce summarization chain
│
└── utils/
    └── helpers.py          ← File size, word count, read time utils
```

---

## 🚀 Installation & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open browser
```
http://localhost:8501
```

### 4. In the sidebar
1. Paste your **OpenAI API key** (`sk-...`)
2. Upload a **PDF, DOCX, or TXT** file
3. Click **⚡ Process Document**
4. Start asking questions!

---

## 📊 Expected Output Examples

**Question:** *What is the main topic of this document?*
```
The document discusses the use of artificial intelligence techniques for
analyzing large datasets and improving decision-making processes.
```

**Question:** *Summarize the document.*
```
📌 OVERVIEW
The document explains how artificial intelligence can assist in analyzing
complex data and improve efficiency in decision-making.

🔑 KEY POINTS
1. Introduction to AI applications in data analysis.
2. Techniques used for document and information retrieval.
3. Benefits of automated extraction and summarization.
4. Challenges and limitations of current AI systems.

✅ CONCLUSION
AI-driven document analysis reduces manual effort and accelerates
understanding of large, complex information sources.
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Response accuracy | Are answers factually grounded in the document? |
| Retrieval relevance | Do retrieved chunks actually relate to the question? |
| Response time | End-to-end latency per query |
| Coverage | Are all key document sections reachable? |

---

## 🌐 Deployment

**Local:**
```bash
streamlit run app.py
```

**HuggingFace Spaces:**
- Upload all files to a new Space
- Set `OPENAI_API_KEY` as a repository secret

**Render / Railway:**
```
Start command: streamlit run app.py --server.port $PORT --server.headless true
```

---

## 🔑 Requirements

- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com))
- ~500 MB disk space for sentence-transformer model (downloaded automatically on first run)
