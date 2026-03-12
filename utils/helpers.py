"""
helpers.py
----------
Shared utility functions used across the application.
"""

import os
import time


def format_file_size(size_bytes: int) -> str:
    """Convert raw byte count to a human-readable string."""
    if size_bytes < 1_024:
        return f"{size_bytes} B"
    elif size_bytes < 1_048_576:
        return f"{size_bytes / 1_024:.1f} KB"
    else:
        return f"{size_bytes / 1_048_576:.1f} MB"


def get_file_extension(filename: str) -> str:
    """Return the lowercase extension of a filename (without the dot)."""
    return os.path.splitext(filename)[-1].lower().lstrip(".")


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate long text and append ellipsis."""
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def estimate_reading_time(text: str, wpm: int = 200) -> str:
    """Estimate how long it would take a human to read the full document."""
    words = len(text.split())
    minutes = max(1, round(words / wpm))
    return f"{minutes} min read" if minutes < 60 else f"{minutes // 60}h {minutes % 60}m read"


def count_words(text: str) -> int:
    """Count the number of words in a string."""
    return len(text.split())


def ensure_dir(*paths: str) -> None:
    """Create one or more directories if they do not already exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
