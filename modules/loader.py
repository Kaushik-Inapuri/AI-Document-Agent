"""
loader.py
---------
Extract text from PDF, DOCX, and TXT files.
Returns a list of LangChain Document objects.
"""

import os


def load_document(path: str) -> list:
    """
    Load a document file and return LangChain Document objects.

    Supported formats: .pdf, .docx, .txt
    """
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".pdf":
        return _load_pdf(path)
    elif ext == ".docx":
        return _load_docx(path)
    elif ext == ".txt":
        return _load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use PDF, DOCX, or TXT.")


def _load_pdf(path: str) -> list:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(path)
    pages = loader.load()
    # tag each page with source metadata
    for i, page in enumerate(pages):
        page.metadata.setdefault("page", i + 1)
        page.metadata["source"] = os.path.basename(path)
    return pages


def _load_docx(path: str) -> list:
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = os.path.basename(path)
    return docs


def _load_txt(path: str) -> list:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = os.path.basename(path)
    return docs
