# LawKhBack

Local-first backend for a Cambodian law assistant app.

This repository runs the backend API used by the Expo mobile app and the Telegram bot interface. It uses local ChromaDB retrieval, local sentence-transformer embeddings, and a configurable LLM provider for answer generation.

## Overview

LawKhBack is designed for a local demo or a normal server host where the Chroma database can live beside the backend. The API accepts legal questions, retrieves relevant Cambodian law chunks from ChromaDB, sends the retrieved context to the configured model provider, and returns an answer with structured citations.

Supported clients:

- Expo React Native APK / Android emulator
- Telegram bot
- Direct HTTP requests to the backend API

Default model provider:

```txt
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=openrouter/free
```

Supported generation providers:

```txt
openrouter
deepseek
ollama
```

## Architecture

```txt
Expo app / Telegram bot
  -> FastAPI backend
    -> local ChromaDB
    -> local sentence-transformers embedding model
    -> OpenRouter / DeepSeek / Ollama chat completion API
    -> answer + citations
```

## Repository Layout

```txt
LawKhBack/
  README.md
  .env.example
  requirements.txt
  run_rag_api.ps1
  rag/
    server.py
    query.py
    chroma_db/
    auth_store.py
    law_assets.py
  scripts/
    prepare_law_assets.py
  law_assets/
    library/       (gitignored, generated locally)
    index/         (gitignored, generated locally)
  telegram_bot/
    bot.py
    api_client.py
    config.py
  .runtime/          (gitignored, created at startup)
```

Important paths:

- `rag/server.py` contains the FastAPI endpoints.
- `rag/query.py` contains retrieval, prompt building, OpenRouter/DeepSeek/Ollama provider switching, and self-check logic.
- `rag/chroma_db/` is where the local Chroma database should be placed.
- `scripts/prepare_law_assets.py` prepares source law ZIP/material folders for the law library API.
- `telegram_bot/` contains the Telegram polling bot that calls the same `/chat` backend endpoint.

## Requirements

- Python 3.10 or newer
- Git LFS, used for the included ChromaDB files
- OpenRouter API key or DeepSeek API key, depending on provider
- MongoDB Atlas URI for signup, login, password reset, and saved chat history
- Telegram bot token if using the Telegram interface

## Setup

Clone the repository and enter the project folder:

```powershell
git lfs install
git clone <your-repo-url>
cd LawKhBack
git lfs pull
```

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create your local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least one provider key:

```txt
OPENROUTER_API_KEY=your_openrouter_api_key
```

or:

```txt
DEEPSEEK_API_KEY=your_deepseek_api_key
```

If using Telegram, also set:

```txt
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

For auth and saved chat history, set:

```txt
MONGODB_URI=your_mongodb_connection_string
MONGODB_DB=lawkh
JWT_SECRET=replace_this_with_a_long_random_secret
```

If Atlas rejects local connections, check the Atlas Network Access IP allowlist first. The backend uses MongoDB for auth and chat history; without a reachable Mongo URI, `/chat` still works but auth/history endpoints cannot persist data.

## Law Documents

The law document ZIPs/source folders are intentionally not committed. Keep them outside git, then prepare them into the local gitignored library.

Set this in `.env`:

```txt
LAW_ASSETS_SOURCE_DIR=D:\Users\aykay\SProjects\KhmerOCR\Original Materials
LAW_ASSETS_DIR=law_assets/library
LAW_ASSETS_INDEX=law_assets/index/documents.json
```

Prepare the law library:

```powershell
python -B scripts\prepare_law_assets.py
```

Expected output includes:

```txt
Indexed 1189 documents
```

The preparation step copies/extracts documents into short local filenames so Windows long paths do not break testing. Generated files stay ignored by git.

## ChromaDB

This repo includes the packaged Chroma database under:

```txt
rag/chroma_db/
```

The folder contains `chroma.sqlite3` and the related Chroma index files for the `cambodian_laws` collection. These large files are stored with Git LFS.

At runtime, the backend copies the packaged database into:

```txt
.runtime/chroma_db/
```

The runtime copy is ignored by git. This keeps the committed database clean while allowing Chroma/SQLite to create temporary journal files locally.

Default collection name:

```txt
cambodian_laws
```

If you want to use a different packaged database, set `CHROMA_SOURCE_DIR` in `.env`:

```txt
CHROMA_SOURCE_DIR=D:\path\to\your\chroma_db
```

If you want to bypass the runtime copy and point Chroma directly at a database, set `CHROMA_DIR`:

```txt
CHROMA_DIR=D:\path\to\your\runtime_chroma_db
```

## Run Locally

Start the backend:

```powershell
.\run_rag_api.ps1
```

By default this starts:

- FastAPI backend on `http://localhost:8000`
- Telegram bot in the background when `RUN_TELEGRAM_BOT=true` and `TELEGRAM_BOT_TOKEN` is set

Health check:

```txt
GET http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "cambodian-legal-rag",
  "provider": "openrouter"
}
```

The `provider` value follows your `LLM_PROVIDER` setting.

## Environment Variables

Default `.env` shape:

```txt
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=http://localhost:3000
OPENROUTER_APP_NAME=Feasible
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
RUN_TELEGRAM_BOT=true
RAG_API_BASE_URL=http://localhost:8000
RAG_REQUEST_TIMEOUT_SECONDS=60

MONGODB_URI=mongodb+srv://username:password@cluster.example.mongodb.net/
MONGODB_DB=lawkh
MONGODB_TIMEOUT_MS=5000
JWT_SECRET=change_this_for_local_dev
JWT_ACCESS_TOKEN_MINUTES=1440
PASSWORD_RESET_TOKEN_MINUTES=30

LAW_ASSETS_DIR=law_assets/library
LAW_ASSETS_INDEX=law_assets/index/documents.json
LAW_ASSETS_SOURCE_DIR=D:\path\to\Original Materials

CHROMA_SOURCE_DIR=rag/chroma_db
CHROMA_DIR=
CHROMA_COLLECTION_NAME=cambodian_laws
EMBEDDING_MODEL=intfloat/multilingual-e5-large

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
```

To run only the backend without Telegram:

```txt
RUN_TELEGRAM_BOT=false
```

To switch generation to Ollama:

```txt
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
```

To switch generation to DeepSeek:

```txt
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

DeepSeek currently exposes these model IDs through `/models`:

```txt
deepseek-v4-flash
deepseek-v4-pro
```

## API

### `GET /health`

Returns service status and active provider.

### `POST /chat`

Main endpoint for the Expo app and Telegram bot.

This endpoint is public. If the client sends `Authorization: Bearer <token>`, the backend attempts to save the turn under that user. If no token is sent, RAG answer generation still works.

Request:

```json
{
  "question": "What are the tax registration requirements for a new company in Cambodia?",
  "chatId": "chat_optional",
  "history": [
    {
      "role": "user",
      "content": "I am setting up a company."
    }
  ],
  "filters": {
    "categoryIds": ["tax", "business-registration"],
    "documentIds": []
  }
}
```

Response:

```json
{
  "chatId": "chat_abc123",
  "answer": "Generated legal answer...",
  "citations": [
    {
      "id": "cite_001",
      "title": "Law on Taxation",
      "fullCitation": "Law on Taxation (Tax)",
      "documentId": "tax/law-on-taxation-2023-pdf",
      "categoryId": "tax",
      "page": null,
      "excerpt": "Relevant source excerpt...",
      "score": 0.82,
      "pdfUrl": "http://localhost:8000/law/documents/tax%2Flaw-on-taxation-2023-pdf/download",
      "downloadUrl": "http://localhost:8000/law/documents/tax%2Flaw-on-taxation-2023-pdf/download",
      "chunkId": "123",
      "locationLabel": "OCR chunk 123"
    }
  ]
}
```

Citation metadata is intentionally source-driven. The backend does not ask the LLM to invent PDF pages. If a real PDF page is unavailable, `page`, `pageStart`, and `pageEnd` are `null`, and `locationLabel` describes the OCR chunk/source location instead.

Citation `documentId` values match the law-library document IDs accepted by:

```txt
GET /law/documents/{documentId}
GET /law/documents/{documentId}/download
```

Document IDs may contain `/`, so frontend clients should URL-encode them before calling document endpoints, for example `tax%2Flaw-on-taxation-2023-pdf`.

Other available endpoints:

- `POST /auth/signup`
- `POST /auth/login`
- `POST /auth/password/forgot`
- `POST /auth/password/reset`
- `GET /auth/me`
- `PATCH /auth/me`
- `GET /me`
- `PATCH /me`
- `GET /chats`
- `GET /chats/{chatId}`
- `GET /law/categories`
- `GET /law/categories/{categoryId}/documents`
- `GET /law/documents/{documentId}`
- `GET /law/documents/{documentId}/download`
- `GET /citations/{citationId}`
- `POST /api/query`
- `POST /api/query/stream`

### Auth Flow

Signup request:

```json
{
  "name": "User Name",
  "email": "user@example.com",
  "password": "Password123!"
}
```

Signup/login response:

```json
{
  "accessToken": "jwt_token",
  "user": {
    "id": "mongo_user_id",
    "name": "User Name",
    "email": "user@example.com",
    "preferences": {
      "darkMode": true
    }
  }
}
```

Forgot password is intentionally simple for the MVP. `POST /auth/password/forgot` returns a reset token directly when the email exists, then `POST /auth/password/reset` changes the password with that token.

### Law Document Viewer

`GET /law/documents/{documentId}` returns document metadata plus direct file URLs:

```json
{
  "id": "labour/example-pdf",
  "categoryId": "labour",
  "title": "Example Labour Document",
  "subtitle": "Employment and worker protections",
  "year": "2024",
  "pages": null,
  "size": "1.2 MB",
  "pdfUrl": "http://localhost:8000/law/documents/labour%2Fexample-pdf/download",
  "fileUrl": "http://localhost:8000/law/documents/labour%2Fexample-pdf/download",
  "downloadUrl": "http://localhost:8000/law/documents/labour%2Fexample-pdf/download",
  "content": "This document is available for download: example.pdf"
}
```

`GET /law/documents/{documentId}/download` supports URL-encoded IDs and returns PDFs with inline browser-friendly headers:

```txt
Content-Type: application/pdf
Content-Disposition: inline
Accept-Ranges: bytes
```

## Expo App Usage

Use these backend URLs from the Expo app:

- Web on same machine: `http://localhost:8000`
- Android emulator: `http://10.0.2.2:8000`
- Physical device: `http://<your-lan-ip>:8000`

The Expo app should call:

```txt
POST /chat
```

The OpenRouter/DeepSeek key and Telegram token stay in this backend repo's `.env`, not inside the APK.

## Telegram Bot Usage

The Telegram bot is started by `run_rag_api.ps1` when:

```txt
RUN_TELEGRAM_BOT=true
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

The bot uses polling and sends user questions to:

```txt
http://localhost:8000/chat
```

Only one polling instance can use the same Telegram bot token at a time. The startup script skips creating a duplicate local `bot.py` process, but Telegram will still return a `409 Conflict` if the same token is already running from another machine or hosting service.

Bot logs are written to:

```txt
telegram_bot_stdout.log
telegram_bot_stderr.log
```

These log files are ignored by git.

## Git Notes

Do not commit `.env`.

The Chroma DB files are intentionally committed through Git LFS so a clone can run local RAG without receiving the database through a separate channel.

Do not commit law document ZIPs or generated law library files. The source ZIP/material folder should be prepared locally with `python -B scripts\prepare_law_assets.py`; generated `law_assets/library/` and `law_assets/index/` are gitignored.
