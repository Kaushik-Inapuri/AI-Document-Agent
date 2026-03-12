"""
summarizer.py
-------------
Smart batch summarization for large documents.
Handles any file size by splitting into tiny batches,
summarizing each one, then combining into a final summary.
"""

import time
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


_BATCH_TEMPLATE = """Summarize the following text in 2-3 sentences only.
Be very concise. Focus only on the most important idea.

TEXT:
{text}

SUMMARY:"""

_FINAL_TEMPLATE = """You are a professional document analyst.
Using the section summaries below, write a final document summary:

📌 OVERVIEW
2-3 sentences about what this document is about.

🔑 KEY POINTS
Most important ideas as numbered points (max 6 points).

✅ CONCLUSION
Main conclusions or takeaways in 2-3 sentences.

SECTION SUMMARIES:
{text}

FINAL SUMMARY:"""

BATCH_PROMPT = PromptTemplate(template=_BATCH_TEMPLATE, input_variables=["text"])
FINAL_PROMPT = PromptTemplate(template=_FINAL_TEMPLATE, input_variables=["text"])

# Very small batch size — 800 chars ≈ ~200 tokens, well within limits
BATCH_CHAR_LIMIT = 800


def _make_llm(api_key: str, max_tokens: int = 256) -> ChatGroq:
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=max_tokens,
    )


def _split_into_batches(docs: list, batch_chars: int = BATCH_CHAR_LIMIT) -> list:
    full_text = "\n\n".join(d.page_content for d in docs)
    # Remove excessive whitespace to reduce token count
    full_text = " ".join(full_text.split())
    batches = []
    start = 0
    while start < len(full_text):
        end = start + batch_chars
        batches.append(full_text[start:end])
        start = end
    return batches


def summarize_document(docs: list, api_key: str) -> str:
    """
    Summarize a document of any size using batch processing.

    Large files are split into 800-char batches.
    Each batch is summarized with a 3s delay between calls.
    All mini-summaries are combined into one final summary.
    """
    batch_llm = _make_llm(api_key, max_tokens=150)  # small output per batch
    final_llm = _make_llm(api_key, max_tokens=512)  # larger output for final

    batch_chain = BATCH_PROMPT | batch_llm | StrOutputParser()
    final_chain = FINAL_PROMPT | final_llm | StrOutputParser()

    batches = _split_into_batches(docs)

    # Limit to first 20 batches max (16,000 chars) for very large files
    batches = batches[:20]

    # If only one small batch — summarize directly
    if len(batches) == 1:
        summary = final_chain.invoke({"text": batches[0]})
        return summary.strip()

    # Summarize each batch individually
    batch_summaries = []
    for i, batch in enumerate(batches):
        try:
            mini = batch_chain.invoke({"text": batch})
            batch_summaries.append(f"[Section {i+1}]: {mini.strip()}")
            # Wait between calls to respect rate limits
            if i < len(batches) - 1:
                time.sleep(3)
        except Exception:
            batch_summaries.append(f"[Section {i+1}]: Could not summarize.")
            time.sleep(5)  # longer wait on error

    # Combine mini-summaries — trim to 2000 chars for final call
    combined = "\n".join(batch_summaries)
    combined = combined[:2000]

    time.sleep(3)  # wait before final call
    final_summary = final_chain.invoke({"text": combined})
    return final_summary.strip()
