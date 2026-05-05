import { embedQuestion } from "./embeddings.js";
import { generateAnswer } from "./openrouter.js";
import type { AppConfig, Citation, RagChatRequest, RagChatResponse } from "./types.js";
import { searchRelevantChunks } from "./vectorStore.js";

function createChatId(existingChatId?: string): string {
  return existingChatId || `chat_${crypto.randomUUID()}`;
}

export async function runRagChat(
  request: RagChatRequest,
  config: AppConfig,
): Promise<RagChatResponse> {
  const embedding = await embedQuestion(request.question, config);
  const chunks = await searchRelevantChunks(embedding, request.filters, config);
  const answer = await generateAnswer({
    question: request.question,
    history: request.history,
    chunks,
    config,
  });

  const citations: Citation[] = chunks.map((chunk) => ({
    id: chunk.id,
    title: chunk.title,
    fullCitation: chunk.fullCitation,
    documentId: chunk.documentId,
    categoryId: chunk.categoryId,
    page: chunk.page,
    excerpt: chunk.text,
    score: chunk.score,
  }));

  return {
    chatId: createChatId(request.chatId),
    answer,
    citations,
  };
}
