import { AppError } from "./errors.js";
import type { AppConfig } from "./types.js";

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new AppError("CONFIG_ERROR", 500, `Missing required environment variable: ${name}`);
  }

  return value;
}

function parseNumber(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function getConfig(): AppConfig {
  return {
    openRouterApiKey: requireEnv("OPENROUTER_API_KEY"),
    openRouterModel: requireEnv("OPENROUTER_MODEL"),
    openRouterBaseUrl: process.env.OPENROUTER_BASE_URL?.trim() || "https://openrouter.ai/api/v1",
    embeddingApiKey: requireEnv("EMBEDDING_API_KEY"),
    embeddingModel: requireEnv("EMBEDDING_MODEL"),
    embeddingApiUrl: process.env.EMBEDDING_API_URL?.trim() || "https://api.openai.com/v1/embeddings",
    vectorDbUrl: requireEnv("VECTOR_DB_URL"),
    vectorDbApiKey: requireEnv("VECTOR_DB_API_KEY"),
    allowedOrigin: process.env.ALLOWED_ORIGIN?.trim() || undefined,
    modelTimeoutMs: parseNumber(process.env.MODEL_TIMEOUT_MS, 25000),
    embeddingTimeoutMs: parseNumber(process.env.EMBEDDING_TIMEOUT_MS, 15000),
    vectorDbTimeoutMs: parseNumber(process.env.VECTOR_DB_TIMEOUT_MS, 10000),
    vectorDbTopK: parseNumber(process.env.VECTOR_DB_TOP_K, 6),
  };
}

export function getAllowedOrigin(): string | undefined {
  return process.env.ALLOWED_ORIGIN?.trim() || undefined;
}
