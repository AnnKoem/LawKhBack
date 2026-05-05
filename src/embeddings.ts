import { AppError } from "./errors.js";
import { createTimeoutSignal } from "./http.js";
import type { AppConfig } from "./types.js";

interface EmbeddingApiResponse {
  data?: Array<{ embedding?: number[] }>;
}

export async function embedQuestion(question: string, config: AppConfig): Promise<number[]> {
  const response = await fetch(config.embeddingApiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.embeddingApiKey}`,
    },
    body: JSON.stringify({
      model: config.embeddingModel,
      input: question,
    }),
    signal: createTimeoutSignal(config.embeddingTimeoutMs),
  }).catch((cause) => {
    throw new AppError("EMBEDDING_ERROR", 502, `Failed to call embedding provider: ${String(cause)}`);
  });

  if (!response.ok) {
    throw new AppError("EMBEDDING_ERROR", 502, "Embedding provider returned an error.");
  }

  const data = (await response.json()) as EmbeddingApiResponse;
  const embedding = data.data?.[0]?.embedding;
  if (!embedding || embedding.length === 0) {
    throw new AppError("EMBEDDING_ERROR", 502, "Embedding provider returned no embedding.");
  }

  return embedding;
}
