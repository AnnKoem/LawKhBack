"""
Client for the Cambodian Law RAG API.
"""

import httpx

from config import RAG_API_BASE_URL, RAG_REQUEST_TIMEOUT_SECONDS


async def query_rag(question: str, user_id: int) -> str:
    """Send a question to the RAG API and return the answer."""
    async with httpx.AsyncClient(timeout=RAG_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{RAG_API_BASE_URL}/chat",
            json={
                "question": question,
                "chatId": f"telegram_{user_id}",
                "history": [],
            },
        )
        response.raise_for_status()
        data = response.json()
        return _format_answer(data)


def _format_answer(payload: dict) -> str:
    answer = (payload.get("answer") or "No answer returned from the API.").strip()
    citations = payload.get("citations") or []

    if not citations:
        return answer

    citation_lines = []
    for index, citation in enumerate(citations[:3], start=1):
        title = citation.get("title") or "Untitled source"
        full_citation = citation.get("fullCitation") or title
        citation_lines.append(f"{index}. {title} - {full_citation}")

    return f"{answer}\n\nSources:\n" + "\n".join(citation_lines)
