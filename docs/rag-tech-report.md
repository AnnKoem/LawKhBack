# LawKh RAG Pipeline: 1-Page Tech Report

## Project Summary

LawKh is a Cambodian law assistant that answers legal questions through a Retrieval-Augmented Generation (RAG) backend. The system is used by two frontends: an Expo React Native mobile app and a Telegram bot. The backend keeps model/API keys private, retrieves relevant law sources, calls a configurable LLM provider, and returns answers with structured citations and document links.

## Data Sources

The legal corpus comes from Cambodian law document collections prepared from the project OCR/data pipeline. The source materials are organized into categories such as labour, tax, banking, finance, business registration, Council for Development of Cambodia, and general law documents. The local law library contains 1,189 prepared source documents, including original PDFs exposed through `/law/documents/{documentId}/download`. The RAG vector database uses the `cambodian_laws` Chroma collection, built from OCR/text chunks of Cambodian legal documents.

## Why RAG

We use RAG because legal answers must be grounded in source material instead of relying only on a model's general knowledge. A base LLM can hallucinate, miss local Cambodian legal context, or fail to cite evidence. RAG improves reliability by retrieving relevant legal chunks before generation, placing those chunks in the prompt, and returning citations that connect answers back to source documents. This is especially important for a legal assistant where users need traceable answers, not just fluent text.

## Pipeline

1. The Expo app or Telegram bot sends a question to `POST /chat`.
2. The backend validates the request and optionally receives chat history and filters.
3. The question is embedded with `intfloat/multilingual-e5-large`.
4. ChromaDB searches the local `cambodian_laws` collection for relevant chunks.
5. Retrieval applies category inference, bad-chunk filtering, deduplication, and fallback search.
6. The backend builds a legal-assistant prompt with the retrieved context and disclaimer.
7. The configured LLM provider generates the answer. The current provider can be DeepSeek, OpenRouter, or Ollama depending on `.env`.
8. The backend returns `answer`, `chatId`, and rich `citations` with document IDs, excerpts, scores, PDF URLs, and location labels.
9. If the user is authenticated, MongoDB stores the chat history for later retrieval or deletion.

## Core Technology Choices

- FastAPI: simple HTTP API for mobile, Telegram, and direct testing.
- ChromaDB: local vector database for demo-friendly RAG retrieval.
- `intfloat/multilingual-e5-large`: multilingual embeddings suitable for English and Khmer queries.
- DeepSeek/OpenRouter/Ollama provider switch: keeps generation flexible across cloud APIs and local model testing.
- MongoDB Atlas: stores users, password reset tokens, and chat history.
- Git LFS: stores the large ChromaDB files so a cloned repo can run local RAG without separately transferring the vector database.

## Current Limitations

The backend does not invent PDF page numbers. If page metadata is unavailable from OCR/indexing, citations return `page: null` and a `locationLabel` such as an OCR chunk. This avoids misleading users with fake page jumps. Future work should rebuild the corpus with exact `pageStart`, `pageEnd`, and section metadata for every chunk.
