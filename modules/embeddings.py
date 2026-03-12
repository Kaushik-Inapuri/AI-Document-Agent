"""
embeddings.py
-------------
Generate vector embeddings using a local HuggingFace model.
No external API call — runs entirely on CPU.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings


def create_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace sentence-transformer embedding model.

    'all-MiniLM-L6-v2' is a lightweight, high-quality model that
    maps text to 384-dimensional vectors for semantic similarity search.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings
