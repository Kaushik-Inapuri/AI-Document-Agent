"""
qa_system.py
------------
RAG Question-Answering chain using Groq (free, no daily limit).
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


_QA_TEMPLATE = """You are an intelligent Document Analysis Agent.
Answer the user's question based ONLY on the document context below.

Rules:
- Answer only from the given context. Do not use outside knowledge.
- If the answer is not in the context, say: "This information is not found in the uploaded document."
- Be concise, clear, and accurate.
- For list-type answers, use numbered points.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""


def create_qa_chain(vectorstore, api_key: str):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=768,
    )
    prompt = PromptTemplate(
        template=_QA_TEMPLATE,
        input_variables=["context", "question"],
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return {"chain": chain, "retriever": retriever}


def ask_question(qa_chain, question: str) -> dict:
    chain = qa_chain["chain"]
    retriever = qa_chain["retriever"]
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return {
        "answer": answer.strip(),
        "sources": sources,
    }
