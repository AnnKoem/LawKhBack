export type ChatRole = "user" | "assistant";

export interface RagChatMessage {
  role: ChatRole;
  content: string;
}

export interface RagFilters {
  categoryIds?: string[];
  documentIds?: string[];
}

export interface RagChatRequest {
  question: string;
  chatId?: string;
  history: RagChatMessage[];
  filters?: RagFilters;
}

export interface Citation {
  id: string;
  title: string;
  fullCitation: string;
  documentId?: string;
  categoryId?: string;
  page?: number;
  excerpt?: string;
  score?: number;
}

export interface RagChatResponse {
  chatId: string;
  answer: string;
  citations: Citation[];
}

export interface RetrievedChunk {
  id: string;
  title: string;
  fullCitation: string;
  text: string;
  documentId?: string;
  categoryId?: string;
  page?: number;
  score?: number;
}

export interface AppConfig {
  openRouterApiKey: string;
  openRouterModel: string;
  openRouterBaseUrl: string;
  embeddingApiKey: string;
  embeddingModel: string;
  embeddingApiUrl: string;
  vectorDbUrl: string;
  vectorDbApiKey: string;
  allowedOrigin?: string;
  modelTimeoutMs: number;
  embeddingTimeoutMs: number;
  vectorDbTimeoutMs: number;
  vectorDbTopK: number;
}
