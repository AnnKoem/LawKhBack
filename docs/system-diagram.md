# LawKh System Diagram

## Architecture Map

```mermaid
flowchart LR
  app["Expo React Native App<br/>Primary frontend"]
  bot["Telegram Bot<br/>Secondary frontend"]

  api["FastAPI Backend<br/>http://localhost:8000"]

  auth["MongoDB Atlas<br/>users, reset tokens,<br/>chat history"]
  chroma["Local ChromaDB<br/>collection: cambodian_laws"]
  embed["Embedding Model<br/>intfloat/multilingual-e5-large"]
  llm["LLM Provider<br/>DeepSeek / OpenRouter / Ollama"]
  files["Law Document Library<br/>prepared PDFs + index"]

  app -->|"POST /chat<br/>GET /chats<br/>DELETE /chats/{chatId}<br/>GET /law/*<br/>POST /auth/*"| api
  bot -->|"POST /chat"| api

  api -->|"signup/login/reset<br/>chat persistence"| auth
  api -->|"embed query"| embed
  embed -->|"query vector"| chroma
  chroma -->|"top legal chunks"| api
  api -->|"prompt + retrieved context"| llm
  llm -->|"answer text"| api
  api -->|"citation documentId / pdfUrl"| files
  files -->|"GET /law/documents/{documentId}/download"| app
  api -->|"answer + citations"| app
  api -->|"answer + citations"| bot
```

## Main API Surface

- `POST /chat`: RAG answer generation. Public endpoint; saves history when a bearer token is provided.
- `GET /chats`: list saved chat history.
- `GET /chats/{chatId}`: open a saved chat.
- `DELETE /chats/{chatId}`: delete a saved chat.
- `POST /auth/signup`: create account in MongoDB.
- `POST /auth/login`: return JWT access token.
- `POST /auth/password/forgot`: MVP reset-token flow.
- `POST /auth/password/reset`: update password by reset token.
- `GET /law/categories`: list law categories.
- `GET /law/categories/{categoryId}/documents`: list source documents in a category.
- `GET /law/documents/{documentId}`: return document metadata and file URLs.
- `GET /law/documents/{documentId}/download`: return original PDF inline.
- `GET /citations/{citationId}`: return rich citation metadata and source document link.

## Deployment / Demo Notes

For the local demo, the backend runs on `http://localhost:8000`, the Expo Android emulator can call `http://10.0.2.2:8000`, and the Telegram bot polls Telegram while calling the same backend `/chat` endpoint. Secrets such as DeepSeek/OpenRouter keys, Telegram token, and MongoDB URI stay in backend `.env`; they are not placed inside the Expo APK.
