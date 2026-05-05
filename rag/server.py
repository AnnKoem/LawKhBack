"""
FastAPI server for Cambodian legal RAG assistant.
Provides chat, law lookup, and health endpoints.
"""

import asyncio
import hashlib
import json
import os
import sys
import uuid
from typing import Literal, Optional

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from query import (
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    generate_answer,
    generate_answer_stream,
    generate_answer_with_review,
    generate_chat_answer,
    query_law,
    should_self_check,
)

app = FastAPI(
    title="Cambodian Legal Assistant",
    description="Bilingual (Khmer/English) legal Q&A powered by RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Legal question")
    language: str = Field(default="all", pattern="^(en|kh|all)$")
    top_k: int = Field(default=5, ge=1, le=20)
    provider: str = Field(default=DEFAULT_LLM_PROVIDER, pattern="^(ollama|openrouter)$")
    ollama_url: str = Field(default="http://localhost:11434")
    model: str = Field(default=DEFAULT_OPENROUTER_MODEL if DEFAULT_LLM_PROVIDER == "openrouter" else DEFAULT_MODEL)
    self_check: bool = Field(default=True, description="Review the answer for high-risk questions")


class SourceChunk(BaseModel):
    text: str
    doc_id: str
    language: str
    category: str
    source_file: str
    verified: bool = True
    corrected: bool = False
    quality: str = "unknown"
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    question: str
    language: str
    review: Optional[dict] = None


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RagFilters(BaseModel):
    categoryIds: list[str] = Field(default_factory=list)
    documentIds: list[str] = Field(default_factory=list)


class RagChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chatId: Optional[str] = None
    history: list[ChatHistoryMessage] = Field(default_factory=list)
    filters: Optional[RagFilters] = None


class RagCitation(BaseModel):
    id: str
    title: str
    fullCitation: str
    documentId: Optional[str] = None
    categoryId: Optional[str] = None
    page: Optional[int] = None
    excerpt: Optional[str] = None
    score: Optional[float] = None


class RagChatResponse(BaseModel):
    chatId: str
    answer: str
    citations: list[RagCitation]


class LawCategory(BaseModel):
    id: str
    title: str
    sourceCategory: str


class LawDocument(BaseModel):
    id: str
    title: str
    categoryId: str
    sourceFile: str


class CitationDetail(RagCitation):
    text: Optional[str] = None


_CITATION_CACHE: dict[str, CitationDetail] = {}

CATEGORY_LABELS = {
    "banking_ocr": ("banking", "Banking and Finance"),
    "tax_ocr": ("tax", "Tax"),
    "Finance_ocr": ("finance", "Public Finance and Tax"),
    "labour_ocr": ("labour", "Labour"),
    "CouncilForDevelopmentOfCambodia_ocr": ("investment", "Investment"),
    "RegistrationBusiness_ocr": ("business-registration", "Business Registration"),
}


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "cambodian-legal-rag", "provider": DEFAULT_LLM_PROVIDER}


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "cambodian-legal-rag",
        "provider": DEFAULT_LLM_PROVIDER,
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def root_health():
    return await health()


@app.post("/chat", response_model=RagChatResponse)
async def chat_endpoint(req: RagChatRequest):
    filters = req.filters or RagFilters()
    chat_id = req.chatId or f"chat_{uuid.uuid4().hex[:12]}"

    try:
        chunks = query_law(
            req.question,
            language="kh",
            top_k=7,
            category_ids=filters.categoryIds,
            document_ids=filters.documentIds,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    if not chunks:
        return RagChatResponse(
            chatId=chat_id,
            answer="I could not find relevant Cambodian legal source documents for that question. Please try a more specific question or remove filters.",
            citations=[],
        )

    try:
        history = [message.model_dump() for message in req.history]
        answer = await asyncio.to_thread(
            generate_chat_answer,
            req.question,
            chunks,
            history,
            DEFAULT_LLM_PROVIDER,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Generation failed with provider '{DEFAULT_LLM_PROVIDER}': {exc}") from exc

    citations = [_chunk_to_citation(chunk) for chunk in chunks]
    return RagChatResponse(chatId=chat_id, answer=answer, citations=citations)


@app.post("/api/chat", response_model=RagChatResponse)
async def api_chat_endpoint(req: RagChatRequest):
    return await chat_endpoint(req)


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    try:
        chunks = query_law(req.question, language=req.language, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    if not chunks:
        return QueryResponse(
            answer="No relevant legal documents were found for your question.",
            sources=[],
            question=req.question,
            language=req.language,
        )

    try:
        if should_self_check(req.question, req.self_check):
            answer, review = await asyncio.to_thread(
                generate_answer_with_review,
                req.question,
                chunks,
                req.provider,
                ollama_url=req.ollama_url,
                model=req.model,
            )
        else:
            answer = await asyncio.to_thread(
                generate_answer,
                req.question,
                chunks,
                req.provider,
                ollama_url=req.ollama_url,
                model=req.model,
            )
            review = None
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Generation failed with provider '{req.provider}': {exc}") from exc

    return QueryResponse(
        answer=answer,
        sources=[SourceChunk(**chunk) for chunk in chunks],
        question=req.question,
        language=req.language,
        review=review,
    )


@app.get("/chats")
async def chats_endpoint():
    return []


@app.get("/law/categories", response_model=list[LawCategory])
async def law_categories_endpoint():
    return [
        LawCategory(id=frontend_id, title=title, sourceCategory=source)
        for source, (frontend_id, title) in CATEGORY_LABELS.items()
    ]


@app.get("/law/categories/{category_id}/documents", response_model=list[LawDocument])
async def law_category_documents_endpoint(category_id: str):
    source_categories = _frontend_category_to_sources(category_id)
    try:
        from query import _get_collection

        data = _get_collection().get(include=["metadatas"], limit=20000)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document metadata lookup failed: {exc}") from exc

    docs: dict[str, LawDocument] = {}
    for meta in data.get("metadatas", []):
        if not meta:
            continue
        source_category = meta.get("category", "")
        if source_categories and source_category not in source_categories:
            continue
        doc_id = meta.get("doc_id", "")
        if not doc_id or doc_id in docs:
            continue
        frontend_cat, _title = CATEGORY_LABELS.get(source_category, (source_category, source_category))
        docs[doc_id] = LawDocument(
            id=doc_id,
            title=_title_from_source(meta.get("source_file", doc_id)),
            categoryId=frontend_cat,
            sourceFile=meta.get("source_file", doc_id),
        )
    return sorted(docs.values(), key=lambda doc: doc.title)


@app.get("/law/documents/{document_id:path}")
async def law_document_endpoint(document_id: str):
    try:
        from query import _get_collection

        data = _get_collection().get(where={"doc_id": document_id}, include=["documents", "metadatas"], limit=2000)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document lookup failed: {exc}") from exc

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    if not documents:
        raise HTTPException(status_code=404, detail="Document not found")

    meta = metadatas[0] if metadatas else {}
    return {
        "id": document_id,
        "title": _title_from_source(meta.get("source_file", document_id)),
        "categoryId": CATEGORY_LABELS.get(meta.get("category", ""), (meta.get("category", ""), ""))[0],
        "sourceFile": meta.get("source_file", document_id),
        "chunks": [
            {"index": item_meta.get("chunk_index", index) if item_meta else index, "text": text}
            for index, (text, item_meta) in enumerate(zip(documents, metadatas))
        ],
    }


@app.get("/citations/{citation_id}", response_model=CitationDetail)
async def citation_detail_endpoint(citation_id: str):
    citation = _CITATION_CACHE.get(citation_id)
    if not citation:
        raise HTTPException(status_code=404, detail="Citation not found in current server memory")
    return citation


@app.post("/api/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    try:
        chunks = query_law(req.question, language=req.language, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    if not chunks:
        async def empty_stream():
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'answer', 'content': 'No relevant legal documents were found for your question.'})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    async def event_stream():
        sources_data = {
            "type": "sources",
            "sources": [
                {
                    "doc_id": chunk["doc_id"],
                    "language": chunk["language"],
                    "category": chunk["category"],
                    "source_file": chunk["source_file"],
                    "score": chunk["score"],
                    "text": chunk["text"][:300],
                }
                for chunk in chunks
            ],
        }
        yield f"data: {json.dumps(sources_data, ensure_ascii=False)}\n\n"

        try:
            generator = generate_answer_stream(
                req.question,
                chunks,
                req.provider,
                ollama_url=req.ollama_url,
                model=req.model,
            )
            for token in generator:
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _chunk_to_citation(chunk: dict) -> RagCitation:
    source_file = chunk.get("source_file", "")
    doc_id = chunk.get("doc_id", "")
    source_category = chunk.get("category", "")
    category_id, category_title = CATEGORY_LABELS.get(source_category, (source_category, source_category))
    title = _title_from_source(source_file or doc_id)
    citation_id = "cite_" + hashlib.sha1(
        f"{doc_id}|{chunk.get('score', 0)}|{chunk.get('text', '')[:80]}".encode("utf-8")
    ).hexdigest()[:12]
    excerpt = _clean_excerpt(chunk.get("text", ""))
    full = f"{title}"
    if category_title:
        full += f" ({category_title})"
    if not chunk.get("verified", True):
        full += " - unverified OCR source"

    detail = CitationDetail(
        id=citation_id,
        title=title,
        fullCitation=full,
        documentId=doc_id,
        categoryId=category_id,
        excerpt=excerpt,
        score=round(float(chunk.get("score", 0.0)), 4),
        text=chunk.get("text", ""),
    )
    _CITATION_CACHE[citation_id] = detail
    return RagCitation(**detail.model_dump(exclude={"text"}))


def _title_from_source(source: str) -> str:
    title = os.path.splitext(os.path.basename(source))[0] or source
    title = title.replace("_", " ").replace("-", " ")
    return " ".join(title.split())[:120]


def _clean_excerpt(text: str, max_len: int = 360) -> str:
    excerpt = " ".join((text or "").split())
    if len(excerpt) > max_len:
        return excerpt[: max_len - 1].rstrip() + "..."
    return excerpt


def _frontend_category_to_sources(category_id: str) -> list[str]:
    mapping = {
        "banking": ["banking_ocr"],
        "tax": ["tax_ocr", "Finance_ocr"],
        "finance": ["Finance_ocr"],
        "labour": ["labour_ocr"],
        "labor": ["labour_ocr"],
        "business-registration": ["RegistrationBusiness_ocr"],
        "commercial": ["RegistrationBusiness_ocr", "banking_ocr"],
        "investment": ["CouncilForDevelopmentOfCambodia_ocr"],
        "cdc": ["CouncilForDevelopmentOfCambodia_ocr"],
    }
    return mapping.get(category_id, [category_id])


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
