"""
vectorstore.py
--------------
Build, persist, and reload a FAISS vector database from document chunks.
FAISS (Facebook AI Similarity Search) enables fast nearest-neighbor lookup
over millions of vectors in milliseconds.
"""

import os
from langchain_community.vectorstores import FAISS


def create_vector_db(chunks: list, embeddings) -> FAISS:
    """
    Build a FAISS vectorstore from document chunks.

    Each chunk is embedded and indexed so that at query time
    we can instantly retrieve the top-K most relevant chunks.

    Args:
        chunks:     List of chunked Document objects.
        embeddings: Embedding model instance.

    Returns:
        FAISS vectorstore ready for similarity search.
    """
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vector_db(vectorstore: FAISS, path: str = "data/vectorstore") -> None:
    """Persist the vectorstore to disk for reuse across sessions."""
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)


def load_vector_db(path: str, embeddings) -> FAISS:
    """Reload a previously saved vectorstore from disk."""
    return FAISS.load_local(
        path, embeddings, allow_dangerous_deserialization=True
    )
