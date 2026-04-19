"""Main query pipeline for intent routing, retrieval, reranking, and answering."""

from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from glass_rag.config import AppConfig
from glass_rag.governance import AuditFactory, ComplianceService
from glass_rag.observability import ObservabilityManager
from glass_rag.prompts.templates import build_answer_prompt
from glass_rag.schemas import Citation, QueryIntent, QueryRequest, QueryResponse, RetrievalMode, SessionTurn
from glass_rag.storage.mongo import MongoGateway


class IntentRoute(BaseModel):
    """Structured output used by the router LLM."""

    intent: QueryIntent
    retrieval_mode: RetrievalMode
    reason: str


class QueryPipeline:
    """Executes the end-to-end answer flow for GLASS-RAG."""

    def __init__(
        self,
        config: AppConfig,
        mongo: MongoGateway,
        retrieval_service,
        reranker,
        router_llm,
        answer_llm,
        observability: ObservabilityManager,
        compliance: ComplianceService,
    ) -> None:
        self.config = config
        self.mongo = mongo
        self.retrieval_service = retrieval_service
        self.reranker = reranker
        self.router_llm = router_llm
        self.answer_llm = answer_llm
        self.observability = observability
        self.compliance = compliance
        self.router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the GLASS-RAG query router. Choose the best intent and retrieval mode for the question. "
                    "Allowed intents: retrieval_qa, sample_query, summary, comparison, knowledge_lookup. "
                    "Allowed retrieval modes: keyword, vector, hybrid.",
                ),
                ("human", "Question: {question}\nIntent hint: {intent_hint}\nRetrieval preference: {preferred_mode}"),
            ]
        )

    def answer(self, request: QueryRequest) -> QueryResponse:
        top_k = request.top_k or self.config.pipeline.default_top_k
        self.mongo.ensure_indexes()

        with self.observability.trace(
            name="query_answering",
            user_id=request.user.user_id,
            session_id=request.session_id,
            input_payload={"question": request.question},
        ) as trace_id:
            callbacks = self.observability.callbacks()
            route = self._route_request(request, callbacks=callbacks)
            history_turns = self.mongo.load_session_turns(request.session_id)
            retrieved = self.retrieval_service.retrieve(
                question=request.question,
                mode=route.retrieval_mode,
                user=request.user,
                top_k=top_k,
            )
            reranked = self.reranker.rerank(request.question, retrieved, callbacks=callbacks)
            answer_text = self._generate_answer(
                request=request,
                route=route,
                history_turns=history_turns,
                documents=reranked,
                callbacks=callbacks,
            )
            citations = self._build_citations(reranked)
            self._persist_session(request, answer_text)
            self.mongo.save_audit(
                AuditFactory.build(
                    event_type="query_answered",
                    actor_id=request.user.user_id,
                    payload={
                        "session_id": request.session_id,
                        "intent": route.intent.value,
                        "retrieval_mode": route.retrieval_mode.value,
                        "citation_count": len(citations),
                    },
                )
            )
            return QueryResponse(
                answer=answer_text,
                intent=route.intent,
                retrieval_mode=route.retrieval_mode,
                citations=citations,
                trace_id=trace_id,
            )

    def _route_request(self, request: QueryRequest, callbacks=None) -> IntentRoute:
        if request.intent_hint and request.preferred_mode:
            return IntentRoute(
                intent=request.intent_hint,
                retrieval_mode=request.preferred_mode,
                reason="user_provided",
            )

        chain = self.router_prompt | self.router_llm.with_structured_output(IntentRoute)
        try:
            return chain.invoke(
                {
                    "question": request.question,
                    "intent_hint": request.intent_hint.value if request.intent_hint else "",
                    "preferred_mode": request.preferred_mode.value if request.preferred_mode else "",
                },
                config={"callbacks": callbacks or []},
            )
        except Exception:
            question = request.question.lower()
            if any(word in question for word in ["compare", "difference", "vs", "contrast", "对比", "比较", "差异"]):
                return IntentRoute(intent=QueryIntent.COMPARISON, retrieval_mode=RetrievalMode.HYBRID, reason="fallback")
            if any(word in question for word in ["summary", "summarize", "总结", "摘要", "归纳"]):
                return IntentRoute(intent=QueryIntent.SUMMARY, retrieval_mode=RetrievalMode.HYBRID, reason="fallback")
            if any(word in question for word in ["sample", "batch", "样品", "批次"]):
                return IntentRoute(intent=QueryIntent.SAMPLE_QUERY, retrieval_mode=RetrievalMode.KEYWORD, reason="fallback")
            return IntentRoute(intent=QueryIntent.RETRIEVAL_QA, retrieval_mode=RetrievalMode.HYBRID, reason="fallback")

    def _generate_answer(
        self,
        *,
        request: QueryRequest,
        route: IntentRoute,
        history_turns: list[SessionTurn],
        documents: list[Document],
        callbacks=None,
    ) -> str:
        prompt = build_answer_prompt(route.intent)
        history_text = self._format_history(history_turns)
        context_text = self._format_context(documents)
        chain = prompt | self.answer_llm
        response = chain.invoke(
            {
                "history": history_text or "No prior history.",
                "question": request.question,
                "context": context_text or "No relevant context was retrieved. State that evidence is insufficient.",
            },
            config={"callbacks": callbacks or []},
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        return self.compliance.redact(content)

    def _format_history(self, turns: Iterable[SessionTurn]) -> str:
        return "\n".join(f"{turn.role}: {turn.content}" for turn in turns)

    def _format_context(self, documents: list[Document]) -> str:
        items: list[str] = []
        consumed = 0
        for document in documents:
            text = self.compliance.redact(document.page_content)
            block = "\n".join(
                [
                    f"[{document.metadata.get('chunk_id', 'unknown')}]",
                    f"Title: {document.metadata.get('title', '')}",
                    f"Page: {document.metadata.get('page_number', '')}",
                    f"Content: {text}",
                ]
            )
            consumed += len(block)
            if consumed > self.config.pipeline.max_context_chars:
                break
            items.append(block)
        return "\n\n".join(items)

    def _build_citations(self, documents: list[Document]) -> list[Citation]:
        citations: list[Citation] = []
        for document in documents:
            citations.append(
                Citation(
                    chunk_id=document.metadata.get("chunk_id", ""),
                    title=document.metadata.get("title", ""),
                    source_uri=document.metadata.get("source_uri"),
                    page_number=document.metadata.get("page_number"),
                    score=document.metadata.get("rerank_score") or document.metadata.get("keyword_score"),
                    quote=self.compliance.redact(document.page_content[:240]),
                )
            )
        return citations

    def _persist_session(self, request: QueryRequest, answer_text: str) -> None:
        self.mongo.append_session_turn(
            SessionTurn(session_id=request.session_id, role="user", content=request.question)
        )
        self.mongo.append_session_turn(
            SessionTurn(session_id=request.session_id, role="assistant", content=answer_text)
        )
