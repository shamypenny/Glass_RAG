"""Prompt templates used by the query pipeline."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from glass_rag.schemas import QueryIntent


def build_answer_prompt(intent: QueryIntent) -> ChatPromptTemplate:
    """Returns the answer prompt specialized for the chosen intent."""

    system_by_intent = {
        QueryIntent.RETRIEVAL_QA: (
            "You are the GLASS-RAG research assistant. Answer only from retrieved evidence. "
            "If evidence is insufficient, say so explicitly. Reply in the user's language."
        ),
        QueryIntent.SAMPLE_QUERY: (
            "You are the GLASS-RAG sample-data assistant. Focus on sample ids, parameters, experimental results, differences, and risks. "
            "Do not guess when the evidence is missing."
        ),
        QueryIntent.SUMMARY: (
            "You are the GLASS-RAG summarization assistant. Produce a structured summary with conclusions, evidence, and open questions."
        ),
        QueryIntent.COMPARISON: (
            "You are the GLASS-RAG comparison assistant. Compare materials, schemes, or experimental results side by side using only retrieved evidence."
        ),
        QueryIntent.KNOWLEDGE_LOOKUP: (
            "You are the GLASS-RAG lookup assistant. Answer concisely and point to the most relevant sources."
        ),
    }

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_by_intent[intent]),
            (
                "human",
                "Conversation history:\n{history}\n\n"
                "User question: {question}\n\n"
                "Retrieved context:\n{context}\n\n"
                "Provide the answer directly. End with a section named 'Citations' using the format [chunk_id] title (page).",
            ),
        ]
    )

