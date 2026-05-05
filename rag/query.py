"""
Query engine for Cambodian legal RAG system.
Retrieves relevant law chunks from ChromaDB and generates answers via either
OpenRouter or Ollama.
"""

import json
import os
import shutil
import sys
import time

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import chromadb
import httpx
from sentence_transformers import SentenceTransformer

PACKAGED_CHROMA_DIR = os.path.join(PROJECT_ROOT, "rag", "chroma_db")
DEFAULT_CHROMA_DIR = os.path.join(PROJECT_ROOT, ".runtime", "chroma_db")
ACTIVE_CHROMA_DIR = os.path.join(PROJECT_ROOT, "rag", "chroma_db_active")
FRESH_ACTIVE_CHROMA_DIR = os.path.join(PROJECT_ROOT, "rag", "chroma_db_active_fresh", "chroma_db")
CHROMA_SOURCE_DIR = os.getenv("CHROMA_SOURCE_DIR", PACKAGED_CHROMA_DIR)
CHROMA_DIR = os.getenv("CHROMA_DIR", DEFAULT_CHROMA_DIR)
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "cambodian_laws")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
DEFAULT_OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Feasible")

SYSTEM_PROMPT = (
    "You are a legal expert specializing in Cambodian law. "
    "Answer the question using the provided legal text as your primary source. "
    "Always cite the specific law name and article number. "
    "If the provided text partially covers the question, answer what you can from it "
    "and supplement with your own knowledge, clearly marking which parts come from "
    "the provided text and which come from your general knowledge. "
    "If the provided text does not cover the question at all, answer from your "
    "general knowledge of Cambodian law but note: 'Based on general legal knowledge, "
    "not from the provided documents.' "
    "Do not make up information."
)

REVIEW_SYSTEM_PROMPT = (
    "You are an independent Cambodian legal answer reviewer. "
    "Review the answer without using the retrieved RAG context. "
    "Check whether the answer is plausible under Cambodian law, whether cited "
    "law/article references look suspicious, and whether exact numbers or dates "
    "need manual verification. Be strict and concise."
)


def _is_high_risk_question(question: str) -> bool:
    lowered = (question or "").lower()
    risk_terms = [
        "rate",
        "rates",
        "tax rate",
        "tax rates",
        "penalty",
        "penalties",
        "fine",
        "fines",
        "deadline",
        "deadlines",
        "article",
        "articles",
        "threshold",
        "thresholds",
        "age limit",
        "minimum age",
        "maximum age",
        "registration fee",
        "interest rate",
        "percent",
        "%",
        "days",
        "months",
        "years",
        "riel",
        "usd",
    ]
    return any(term in lowered for term in risk_terms)


_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        _ensure_runtime_chroma_dir()
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _ensure_runtime_chroma_dir() -> None:
    if os.getenv("CHROMA_DIR"):
        return

    source_sqlite = os.path.join(CHROMA_SOURCE_DIR, "chroma.sqlite3")
    target_sqlite = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if not os.path.exists(source_sqlite):
        fallback = (
            ACTIVE_CHROMA_DIR
            if os.path.exists(os.path.join(ACTIVE_CHROMA_DIR, "chroma.sqlite3"))
            else FRESH_ACTIVE_CHROMA_DIR
        )
        if os.path.exists(os.path.join(fallback, "chroma.sqlite3")):
            _copy_chroma_dir(fallback, CHROMA_DIR)
        return

    should_copy = not os.path.exists(target_sqlite)
    if not should_copy:
        source_stat = os.stat(source_sqlite)
        target_stat = os.stat(target_sqlite)
        should_copy = source_stat.st_size != target_stat.st_size or source_stat.st_mtime > target_stat.st_mtime

    if should_copy:
        _copy_chroma_dir(CHROMA_SOURCE_DIR, CHROMA_DIR)
    else:
        journal = os.path.join(CHROMA_DIR, "chroma.sqlite3-journal")
        if os.path.exists(journal):
            os.remove(journal)


def _copy_chroma_dir(source: str, target: str) -> None:
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree(source, target)


def query_law(
    question: str,
    language: str = "all",
    top_k: int = 5,
    category_ids: list[str] | None = None,
    document_ids: list[str] | None = None,
) -> list[dict]:
    started_at = time.perf_counter()
    if not category_ids:
        category_ids = _infer_category_ids(question)

    _debug_log(f"query_law start language={language} top_k={top_k} category_ids={category_ids or []} document_ids={document_ids or []}")
    model_started = time.perf_counter()
    model = _get_model()
    _debug_log(f"model ready in {time.perf_counter() - model_started:.2f}s")

    collection_started = time.perf_counter()
    collection = _get_collection()
    _debug_log(f"collection ready in {time.perf_counter() - collection_started:.2f}s")

    category_set = {c for cid in (category_ids or []) for c in _expand_category_id(cid)}
    document_set = set(document_ids or [])

    embedding_started = time.perf_counter()
    embed = model.encode(
        f"query: {question}",
        normalize_embeddings=True,
    ).tolist()
    _debug_log(f"embedding generated in {time.perf_counter() - embedding_started:.2f}s")

    final = _query_law_once(
        collection=collection,
        embed=embed,
        language=language,
        top_k=top_k,
        category_set=category_set,
        document_set=document_set,
        use_category_filter=True,
    )

    if not final and category_set:
        _debug_log("category-filtered query returned no clean chunks; retrying without category constraint")
        final = _query_law_once(
            collection=collection,
            embed=embed,
            language=language,
            top_k=top_k,
            category_set=category_set,
            document_set=document_set,
            use_category_filter=False,
        )

    _debug_log(f"query_law done chunks={len(final)} total={time.perf_counter() - started_at:.2f}s")
    return final


def _query_law_once(
    *,
    collection,
    embed: list[float],
    language: str,
    top_k: int,
    category_set: set[str],
    document_set: set[str],
    use_category_filter: bool,
) -> list[dict]:
    has_filters = bool(category_set or document_set)
    fetch_k = top_k * (20 if has_filters else 12)
    query_params = {
        "query_embeddings": [embed],
        "n_results": fetch_k,
    }
    if use_category_filter and category_set and language == "all":
        categories = sorted(category_set)
        query_params["where"] = {"category": categories[0]} if len(categories) == 1 else {"category": {"$in": categories}}
    elif language != "all":
        query_params["where"] = {"language": language}

    chroma_started = time.perf_counter()
    results = collection.query(**query_params)
    raw_count = len(results["ids"][0]) if results.get("ids") else 0
    _debug_log(
        f"chroma query returned raw={raw_count} in {time.perf_counter() - chroma_started:.2f}s "
        f"where={query_params.get('where')}"
    )

    seen_docs = set()
    deduped = []
    remaining = []
    skipped_bad = 0
    skipped_category = 0
    skipped_document = 0

    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        doc_id = meta["doc_id"]
        category = meta["category"]
        if category_set and category not in category_set:
            skipped_category += 1
            continue
        if document_set and doc_id not in document_set:
            skipped_document += 1
            continue
        text = results["documents"][0][i]
        if _is_bad_source_text(text):
            skipped_bad += 1
            continue

        chunk = {
            "text": text,
            "doc_id": doc_id,
            "language": meta["language"],
            "category": category,
            "source_file": meta["source_file"],
            "verified": meta.get("verified", True),
            "corrected": meta.get("corrected", False),
            "quality": meta.get("quality", "unknown"),
            "score": 1 - results["distances"][0][i],
        }
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            deduped.append(chunk)
        else:
            remaining.append(chunk)

    final = deduped[:top_k]
    if len(final) < top_k:
        final.extend(remaining[: top_k - len(final)])

    _debug_log(
        f"post-filter clean={len(final)} skipped_bad={skipped_bad} "
        f"skipped_category={skipped_category} skipped_document={skipped_document}"
    )
    for index, chunk in enumerate(final, start=1):
        preview = " ".join(chunk["text"].split())[:220]
        _debug_log(
            f"chunk[{index}] category={chunk['category']} source={chunk['source_file']} "
            f"score={chunk['score']:.4f} text={preview}"
        )

    return final


def _debug_log(message: str) -> None:
    if os.getenv("RAG_DEBUG", "").lower() in {"1", "true", "yes", "on"}:
        print(f"[rag] {message}", file=sys.stderr)


def _is_bad_source_text(text: str) -> bool:
    lowered = (text or "").lower()
    bad_markers = [
        "failed to authenticate",
        "authentication_error",
        "invalid authentication credentials",
        "api error: 401",
    ]
    return any(marker in lowered for marker in bad_markers)


def _infer_category_ids(question: str) -> list[str]:
    lowered = (question or "").lower()
    keyword_map = [
        (("working hour", "working hours", "overtime", "employee", "employment", "labour", "labor", "wage", "salary"), ["labour"]),
        (("tax", "vat", "withholding", "income tax", "taxation"), ["tax"]),
        (("bank", "banking", "loan", "credit", "deposit"), ["banking"]),
        (("company registration", "business registration", "register company", "commercial"), ["business-registration"]),
        (("investment", "cdc", "qualified investment project", "qip"), ["investment"]),
    ]
    for keywords, categories in keyword_map:
        if any(keyword in lowered for keyword in keywords):
            return categories
    return []


def _expand_category_id(category_id: str) -> list[str]:
    mapping = {
        "banking": ["banking_ocr"],
        "banking_ocr": ["banking_ocr"],
        "tax": ["tax_ocr", "Finance_ocr"],
        "tax_ocr": ["tax_ocr"],
        "finance": ["Finance_ocr"],
        "labour": ["labour_ocr"],
        "labor": ["labour_ocr"],
        "labour_ocr": ["labour_ocr"],
        "business-registration": ["RegistrationBusiness_ocr"],
        "registration": ["RegistrationBusiness_ocr"],
        "commercial": ["RegistrationBusiness_ocr", "banking_ocr"],
        "investment": ["CouncilForDevelopmentOfCambodia_ocr"],
        "cdc": ["CouncilForDevelopmentOfCambodia_ocr"],
    }
    return mapping.get(category_id, [category_id])


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    has_unverified = False
    for i, chunk in enumerate(chunks, 1):
        corrected_tag = " | corrected" if chunk.get("corrected", False) else ""
        quality_tag = f" | quality={chunk.get('quality', 'unknown')}"
        verified_tag = "" if chunk.get("verified", True) else " | UNVERIFIED"
        if not chunk.get("verified", True):
            has_unverified = True
        context_parts.append(
            f"[Source {i}: {chunk['source_file']} | {chunk['category']} | {chunk['language'].upper()}"
            f"{corrected_tag}{quality_tag}{verified_tag}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    warning = ""
    if has_unverified:
        warning = (
            "\n\nNote: Some sources above are marked UNVERIFIED - their OCR text may contain "
            "errors (garbled numbers, corrupted characters). Cross-check any specific numbers, "
            "dates, or article references from unverified sources against your own knowledge "
            "before including them in your answer.\n"
        )

    return (
        f"Legal context:\n\n{context}\n\n"
        f"---{warning}\n\n"
        f"Question: {question}\n\n"
        f"Provide a detailed answer citing specific law names and article numbers."
    )


def _build_chat_prompt(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    context_parts = []
    has_unverified = False
    for i, chunk in enumerate(chunks, 1):
        verified_tag = "" if chunk.get("verified", True) else " | UNVERIFIED OCR"
        if not chunk.get("verified", True):
            has_unverified = True
        context_parts.append(
            f"[Source {i}: {chunk['source_file']} | documentId={chunk['doc_id']} | "
            f"categoryId={chunk['category']} | score={chunk['score']:.4f}{verified_tag}]\n"
            f"{chunk['text']}"
        )

    history_lines = []
    for msg in (history or [])[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            history_lines.append(f"{role}: {content}")

    warning = ""
    if has_unverified:
        warning = (
            "\nSome retrieved sources are marked UNVERIFIED OCR. Be cautious with exact "
            "numbers, dates, penalties, and article references from those sources. "
            "State when manual verification is needed."
        )

    return (
        "Recent chat history:\n"
        + ("\n".join(history_lines) if history_lines else "(none)")
        + "\n\nRetrieved legal context:\n\n"
        + "\n\n---\n\n".join(context_parts)
        + warning
        + "\n\nUser question:\n"
        + question
        + "\n\nReturn a concise legal-assistant answer. Cite source law/article details when present. "
        "Do not fabricate citations. Include a brief legal-information disclaimer only when useful."
    )


def _resolve_provider(provider: str | None = None) -> str:
    selected = (provider or DEFAULT_LLM_PROVIDER or "ollama").strip().lower()
    if selected not in {"openrouter", "ollama"}:
        raise ValueError(f"Unsupported LLM provider: {selected}")
    return selected


def _build_generation_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _generate_via_ollama(
    messages: list[dict[str, str]],
    *,
    ollama_url: str,
    model: str,
    response_format: str | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "top_p": 1},
    }
    if response_format:
        payload["format"] = response_format

    response = httpx.post(
        f"{ollama_url}/api/chat",
        json=payload,
        timeout=300.0,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _generate_via_openrouter(
    messages: list[dict[str, str]],
    *,
    model: str,
    openrouter_base_url: str,
) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    response = httpx.post(
        f"{openrouter_base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not content:
        raise ValueError("OpenRouter returned an empty message.")
    return content


def generate_chat_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    response_format: str | None = None,
) -> str:
    selected_provider = _resolve_provider(provider)
    messages = _build_generation_messages(system_prompt, user_prompt)

    if selected_provider == "openrouter":
        return _generate_via_openrouter(
            messages,
            model=model or DEFAULT_OPENROUTER_MODEL,
            openrouter_base_url=openrouter_base_url,
        )

    return _generate_via_ollama(
        messages,
        ollama_url=ollama_url,
        model=model or DEFAULT_MODEL,
        response_format=response_format,
    )


def generate_answer(
    question: str,
    chunks: list[dict],
    provider: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str | None = None,
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
) -> str:
    prompt = _build_prompt(question, chunks)
    return generate_chat_completion(
        SYSTEM_PROMPT,
        prompt,
        provider=provider,
        model=model,
        ollama_url=ollama_url,
        openrouter_base_url=openrouter_base_url,
    )


def generate_chat_answer(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    provider: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str | None = None,
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
) -> str:
    prompt = _build_chat_prompt(question, chunks, history=history)
    return generate_chat_completion(
        SYSTEM_PROMPT,
        prompt,
        provider=provider,
        model=model,
        ollama_url=ollama_url,
        openrouter_base_url=openrouter_base_url,
    )


def review_answer(
    question: str,
    answer: str,
    provider: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str | None = None,
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
) -> dict:
    prompt = f"""Review this Cambodian legal Q&A answer independently.

Question:
{question}

Answer:
{answer}

Return ONLY valid JSON with this shape:
{{
  "verdict": "agree" | "uncertain" | "disagree",
  "confidence": 0.0,
  "needs_manual_review": true,
  "issues": ["brief issue, if any"],
  "note": "one short sentence"
}}

Use "uncertain" when the answer depends on exact numbers, dates, article numbers, or legal status that you cannot independently verify."""

    raw = generate_chat_completion(
        REVIEW_SYSTEM_PROMPT,
        prompt,
        provider=provider,
        model=model,
        ollama_url=ollama_url,
        openrouter_base_url=openrouter_base_url,
        response_format="json",
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "verdict": "uncertain",
            "confidence": 0.0,
            "needs_manual_review": True,
            "issues": ["Reviewer returned malformed JSON."],
            "note": raw[:300],
        }
    parsed.setdefault("verdict", "uncertain")
    parsed.setdefault("confidence", 0.0)
    parsed.setdefault("needs_manual_review", parsed.get("verdict") != "agree")
    parsed.setdefault("issues", [])
    parsed.setdefault("note", "")
    return parsed


def generate_answer_with_review(
    question: str,
    chunks: list[dict],
    provider: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str | None = None,
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
) -> tuple[str, dict]:
    answer = generate_answer(
        question,
        chunks,
        provider=provider,
        ollama_url=ollama_url,
        model=model,
        openrouter_base_url=openrouter_base_url,
    )
    review = review_answer(
        question,
        answer,
        provider=provider,
        ollama_url=ollama_url,
        model=model,
        openrouter_base_url=openrouter_base_url,
    )
    return answer, review


def generate_answer_stream(
    question: str,
    chunks: list[dict],
    provider: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str | None = None,
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
):
    prompt = _build_prompt(question, chunks)
    selected_provider = _resolve_provider(provider)

    if selected_provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        with httpx.stream(
            "POST",
            f"{openrouter_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": OPENROUTER_SITE_URL,
                "X-Title": OPENROUTER_APP_NAME,
            },
            json={
                "model": model or DEFAULT_OPENROUTER_MODEL,
                "messages": _build_generation_messages(SYSTEM_PROMPT, prompt),
                "temperature": 0,
                "top_p": 1,
                "stream": True,
            },
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                if content:
                    yield content
        return

    with httpx.stream(
        "POST",
        f"{ollama_url}/api/chat",
        json={
            "model": model or DEFAULT_MODEL,
            "messages": _build_generation_messages(SYSTEM_PROMPT, prompt),
            "stream": True,
        },
        timeout=300.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            if content:
                yield content
            if data.get("done", False):
                break


def should_self_check(question: str, requested: bool = True) -> bool:
    return bool(requested and _is_high_risk_question(question))
