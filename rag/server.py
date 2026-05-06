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
from datetime import datetime, timezone
from typing import Literal, Optional

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, EmailStr, Field

from auth_store import (
    chats_collection,
    create_access_token,
    create_reset_token,
    decode_access_token,
    get_user_by_id,
    hash_password,
    normalize_email,
    public_user,
    reset_password,
    users_collection,
    verify_password,
)
from law_assets import (
    build_law_index,
    get_document,
    get_document_path,
    list_categories,
    list_documents,
    prepare_assets_from_source,
)

try:
    from query import (
        DEFAULT_LLM_PROVIDER,
        DEFAULT_DEEPSEEK_MODEL,
        DEFAULT_MODEL,
        DEFAULT_OPENROUTER_MODEL,
        generate_answer,
        generate_answer_stream,
        generate_answer_with_review,
        generate_chat_answer,
        query_law,
        should_self_check,
    )
except ModuleNotFoundError:
    from .query import (
        DEFAULT_LLM_PROVIDER,
        DEFAULT_DEEPSEEK_MODEL,
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
    provider: str = Field(default=DEFAULT_LLM_PROVIDER, pattern="^(ollama|openrouter|deepseek)$")
    ollama_url: str = Field(default="http://localhost:11434")
    model: str = Field(
        default=DEFAULT_OPENROUTER_MODEL
        if DEFAULT_LLM_PROVIDER == "openrouter"
        else DEFAULT_DEEPSEEK_MODEL
        if DEFAULT_LLM_PROVIDER == "deepseek"
        else DEFAULT_MODEL
    )
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


class AuthUser(BaseModel):
    id: str
    name: str
    email: EmailStr
    preferences: dict = Field(default_factory=lambda: {"darkMode": True})


class AuthResponse(BaseModel):
    accessToken: str
    user: AuthUser


class SignupRequest(BaseModel):
    name: str = Field(..., min_length=1)
    email: EmailStr
    password: str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    ok: bool
    resetToken: Optional[str] = None
    resetUrl: Optional[str] = None


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=10)
    password: str = Field(..., min_length=8)


class OkResponse(BaseModel):
    ok: bool


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1)
    preferences: Optional[dict] = None


class ChatSummaryResponse(BaseModel):
    id: str
    title: str
    preview: str
    updatedAt: str


class StoredMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    createdAt: str
    citations: list[RagCitation] = Field(default_factory=list)


class ChatDetailResponse(BaseModel):
    id: str
    title: str
    updatedAt: str
    messages: list[StoredMessage]


class LawCategory(BaseModel):
    id: str
    icon: Optional[str] = None
    name: str
    description: str
    documentCount: int


class LawDocument(BaseModel):
    id: str
    title: str
    categoryId: str
    subtitle: str
    year: str
    pages: Optional[int] = None
    size: Optional[str] = None


class LawDocumentDetail(LawDocument):
    content: str


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _auth_user_from_header(authorization: str | None) -> dict | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_access_token(token)
    except Exception:
        return None
    return get_user_by_id(payload.get("sub", ""))


def _require_user(authorization: str | None) -> dict:
    user = _auth_user_from_header(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or missing access token")
    return user


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


@app.post("/auth/signup", response_model=AuthResponse)
async def signup_endpoint(req: SignupRequest):
    email = normalize_email(req.email)
    now = datetime.now(timezone.utc)
    user_doc = {
        "name": req.name.strip(),
        "email": email,
        "passwordHash": hash_password(req.password),
        "preferences": {"darkMode": True},
        "createdAt": now,
        "updatedAt": now,
    }
    try:
        result = users_collection().insert_one(user_doc)
    except Exception as exc:
        if "duplicate" in str(exc).lower():
            raise HTTPException(status_code=409, detail="Email is already registered") from exc
        raise HTTPException(status_code=503, detail="User database is unavailable") from exc
    user_doc["_id"] = result.inserted_id
    return AuthResponse(accessToken=create_access_token(user_doc), user=AuthUser(**public_user(user_doc)))


@app.post("/auth/login", response_model=AuthResponse)
async def login_endpoint(req: LoginRequest):
    try:
        user = users_collection().find_one({"email": normalize_email(req.email)})
    except Exception as exc:
        raise HTTPException(status_code=503, detail="User database is unavailable") from exc
    if not user or not verify_password(req.password, user.get("passwordHash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return AuthResponse(accessToken=create_access_token(user), user=AuthUser(**public_user(user)))


@app.post("/auth/password/forgot", response_model=ForgotPasswordResponse)
async def forgot_password_endpoint(req: ForgotPasswordRequest):
    try:
        reset = create_reset_token(req.email)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="User database is unavailable") from exc
    if not reset:
        return ForgotPasswordResponse(ok=True)
    return ForgotPasswordResponse(ok=True, resetToken=reset["token"], resetUrl=reset["resetUrl"])


@app.post("/auth/password/reset", response_model=OkResponse)
async def reset_password_endpoint(req: ResetPasswordRequest):
    try:
        ok = reset_password(req.token, req.password)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="User database is unavailable") from exc
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    return OkResponse(ok=True)


@app.get("/auth/me", response_model=AuthUser)
async def auth_me_endpoint(authorization: Optional[str] = Header(default=None)):
    return AuthUser(**public_user(_require_user(authorization)))


@app.get("/me", response_model=AuthUser)
async def me_endpoint(authorization: Optional[str] = Header(default=None)):
    return await auth_me_endpoint(authorization)


@app.patch("/auth/me", response_model=AuthUser)
async def update_me_endpoint(req: UpdateProfileRequest, authorization: Optional[str] = Header(default=None)):
    user = _require_user(authorization)
    update: dict = {"updatedAt": datetime.now(timezone.utc)}
    if req.name is not None:
        update["name"] = req.name.strip()
    if req.preferences is not None:
        update["preferences"] = req.preferences
    try:
        users_collection().update_one({"_id": user["_id"]}, {"$set": update})
        updated = users_collection().find_one({"_id": user["_id"]})
    except Exception as exc:
        raise HTTPException(status_code=503, detail="User database is unavailable") from exc
    return AuthUser(**public_user(updated))


@app.patch("/me", response_model=AuthUser)
async def update_me_alias_endpoint(req: UpdateProfileRequest, authorization: Optional[str] = Header(default=None)):
    return await update_me_endpoint(req, authorization)


@app.post("/auth/logout", response_model=OkResponse)
async def logout_endpoint():
    return OkResponse(ok=True)


@app.post("/chat", response_model=RagChatResponse)
async def chat_endpoint(req: RagChatRequest, authorization: Optional[str] = Header(default=None)):
    filters = req.filters or RagFilters()
    chat_id = req.chatId or f"chat_{uuid.uuid4().hex[:12]}"

    try:
        chunks = query_law(
            req.question,
            language="all",
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
    user = _auth_user_from_header(authorization)
    try:
        _store_chat_turn(chat_id, req.question, answer, citations, user_id=str(user["_id"]) if user else None)
    except Exception as exc:
        print(f"[chat] failed to persist chat history: {exc}", file=sys.stderr)
    return RagChatResponse(chatId=chat_id, answer=answer, citations=citations)


@app.post("/api/chat", response_model=RagChatResponse)
async def api_chat_endpoint(req: RagChatRequest, authorization: Optional[str] = Header(default=None)):
    return await chat_endpoint(req, authorization)


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


@app.get("/chats", response_model=list[ChatSummaryResponse])
async def chats_endpoint(authorization: Optional[str] = Header(default=None)):
    user = _auth_user_from_header(authorization)
    query = {"userId": str(user["_id"])} if user else {}
    try:
        docs = chats_collection().find(query).sort("updatedAt", -1).limit(100)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Chat history database is unavailable") from exc
    return [
        ChatSummaryResponse(
            id=doc["id"],
            title=doc.get("title", "New Chat"),
            preview=doc.get("preview", ""),
            updatedAt=_mongo_dt_to_iso(doc.get("updatedAt")),
        )
        for doc in docs
    ]


@app.get("/chats/{chat_id}", response_model=ChatDetailResponse)
async def chat_detail_endpoint(chat_id: str, authorization: Optional[str] = Header(default=None)):
    user = _auth_user_from_header(authorization)
    query = {"id": chat_id}
    if user:
        query["userId"] = str(user["_id"])
    try:
        doc = chats_collection().find_one(query)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Chat history database is unavailable") from exc
    if not doc:
        raise HTTPException(status_code=404, detail="Chat not found")
    return ChatDetailResponse(
        id=doc["id"],
        title=doc.get("title", "New Chat"),
        updatedAt=_mongo_dt_to_iso(doc.get("updatedAt")),
        messages=[
            StoredMessage(
                id=message["id"],
                role=message["role"],
                content=message["content"],
                createdAt=_mongo_dt_to_iso(message.get("createdAt")),
                citations=[RagCitation(**citation) for citation in message.get("citations", [])],
            )
            for message in doc.get("messages", [])
        ],
    )


@app.get("/law/categories", response_model=list[LawCategory])
async def law_categories_endpoint():
    return [LawCategory(**category) for category in list_categories()]


@app.get("/law/categories/{category_id}/documents", response_model=list[LawDocument])
async def law_category_documents_endpoint(category_id: str):
    return [LawDocument(**doc) for doc in list_documents(category_id)]


@app.get("/law/documents/{document_id:path}/download")
async def law_document_download_endpoint(document_id: str):
    path = get_document_path(document_id)
    if not path:
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(path, filename=path.name)


@app.get("/law/documents/{document_id:path}", response_model=LawDocumentDetail)
async def law_document_endpoint(document_id: str):
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return LawDocumentDetail(**doc)


@app.post("/admin/law-assets/prepare")
async def prepare_law_assets_endpoint():
    prepared = await asyncio.to_thread(prepare_assets_from_source)
    index = await asyncio.to_thread(build_law_index)
    return {"ok": True, **prepared, "documentCount": len(index.get("documents", []))}


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


def _mongo_dt_to_iso(value) -> str:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, str):
        return value
    return _utc_now_iso()


def _store_chat_turn(
    chat_id: str,
    question: str,
    answer: str,
    citations: list[RagCitation],
    user_id: str | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    user_message = {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "role": "user",
        "content": question,
        "createdAt": now,
        "citations": [],
    }
    assistant_message = {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "role": "assistant",
        "content": answer,
        "createdAt": now,
        "citations": [citation.model_dump() for citation in citations],
    }
    title = _clean_excerpt(question, max_len=72)
    preview = _clean_excerpt(answer, max_len=160)
    insert_fields = {
        "id": chat_id,
        "title": title,
        "createdAt": now,
    }

    update = {
        "$setOnInsert": insert_fields,
        "$set": {"preview": preview, "updatedAt": now},
        "$push": {"messages": {"$each": [user_message, assistant_message]}},
    }
    if user_id:
        update["$set"]["userId"] = user_id
    chats_collection().update_one({"id": chat_id}, update, upsert=True)


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
