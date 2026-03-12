"""
chunking.py
-----------
Split large documents into smaller, overlapping chunks so the
retriever can find precise answers even in long documents.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(documents: list,
               chunk_size: int = 600,
               chunk_overlap: int = 120) -> list:
    """
    Split a list of Document objects into smaller chunks.

    Args:
        documents:     LangChain Document objects from loader.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Characters shared between adjacent chunks
                       to preserve context across boundaries.

    Returns:
        List of chunked Document objects (each keeps its metadata).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split at paragraph / sentence / word boundaries first
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks
