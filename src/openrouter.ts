import { AppError } from "./errors.js";
import { createTimeoutSignal } from "./http.js";
import type { AppConfig, RagChatMessage, RetrievedChunk } from "./types.js";

interface OpenRouterResponse {
  choices?: Array<{
    message?: {
      content?: string;
    };
  }>;
}

function buildSystemPrompt(chunks: RetrievedChunk[]): string {
  const context = chunks
    .map((chunk, index) => {
      const metadata = [
        `source_id=${chunk.id}`,
        `title=${chunk.title}`,
        `citation=${chunk.fullCitation}`,
        chunk.page ? `page=${chunk.page}` : undefined,
      ]
        .filter(Boolean)
        .join(", ");

      return `[${index + 1}] ${metadata}\n${chunk.text}`;
    })
    .join("\n\n");

  return [
    "You are a Cambodian law assistant.",
    "Answer only from the provided legal context when possible.",
    "If the context is insufficient, say so clearly and avoid fabricating legal claims.",
    "Do not claim to be a lawyer and do not present the response as formal legal advice.",
    "Prefer concise, practical language.",
    "When you rely on context, mention the relevant source numbers like [1] or [2] in the answer.",
    "",
    "Legal disclaimer: This answer is for general informational purposes and is not legal advice.",
    "",
    "Retrieved context:",
    context || "No context retrieved.",
  ].join("\n");
}

export async function generateAnswer(params: {
  question: string;
  history: RagChatMessage[];
  chunks: RetrievedChunk[];
  config: AppConfig;
}): Promise<string> {
  const { question, history, chunks, config } = params;
  const systemPrompt = buildSystemPrompt(chunks);

  const response = await fetch(`${config.openRouterBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.openRouterApiKey}`,
      "HTTP-Referer": "https://lawkh-backend.local",
      "X-Title": "LawKh Backend",
    },
    body: JSON.stringify({
      model: config.openRouterModel,
      messages: [
        { role: "system", content: systemPrompt },
        ...history,
        { role: "user", content: question },
      ],
      temperature: 0.2,
    }),
    signal: createTimeoutSignal(config.modelTimeoutMs),
  }).catch((cause) => {
    const message = String(cause);
    if (message.includes("TimeoutError")) {
      throw new AppError("MODEL_TIMEOUT", 504, "The model took too long to respond. Please try again.");
    }

    throw new AppError("MODEL_ERROR", 502, `Failed to call OpenRouter: ${message}`);
  });

  if (!response.ok) {
    throw new AppError("MODEL_ERROR", 502, "The model provider returned an error.");
  }

  const data = (await response.json()) as OpenRouterResponse;
  const answer = data.choices?.[0]?.message?.content?.trim();
  if (!answer) {
    throw new AppError("MODEL_ERROR", 502, "The model provider returned an empty answer.");
  }

  return answer;
}
