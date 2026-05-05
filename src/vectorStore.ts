import { AppError } from "./errors.js";
import { createTimeoutSignal } from "./http.js";
import type { AppConfig, RagFilters, RetrievedChunk } from "./types.js";

interface VectorSearchResponse {
  matches?: Array<{
    id: string;
    score?: number;
    payload?: {
      title?: string;
      fullCitation?: string;
      text?: string;
      documentId?: string;
      categoryId?: string;
      page?: number;
      excerpt?: string;
    };
  }>;
}

export async function searchRelevantChunks(
  embedding: number[],
  filters: RagFilters | undefined,
  config: AppConfig,
): Promise<RetrievedChunk[]> {
  const response = await fetch(config.vectorDbUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.vectorDbApiKey}`,
      "x-api-key": config.vectorDbApiKey,
    },
    body: JSON.stringify({
      vector: embedding,
      topK: config.vectorDbTopK,
      filters: filters || {},
    }),
    signal: createTimeoutSignal(config.vectorDbTimeoutMs),
  }).catch((cause) => {
    throw new AppError("VECTOR_DB_ERROR", 502, `Failed to query vector DB: ${String(cause)}`);
  });

  if (!response.ok) {
    throw new AppError("VECTOR_DB_ERROR", 502, "Vector DB returned an error.");
  }

  const data = (await response.json()) as VectorSearchResponse;
  const matches = data.matches ?? [];

  return matches
    .filter((match) => match.payload?.text && match.payload?.title && match.payload?.fullCitation)
    .map((match) => ({
      id: match.id,
      title: match.payload!.title!,
      fullCitation: match.payload!.fullCitation!,
      text: match.payload!.text!,
      documentId: match.payload?.documentId,
      categoryId: match.payload?.categoryId,
      page: match.payload?.page,
      score: match.score,
    }));
}
