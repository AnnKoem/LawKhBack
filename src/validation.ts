import { z } from "zod";
import { AppError } from "./errors.js";
import type { RagChatRequest } from "./types.js";

const chatMessageSchema = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.string().trim().min(1).max(12000),
});

const ragChatRequestSchema = z.object({
  question: z.string().trim().min(1).max(4000),
  chatId: z.string().trim().min(1).max(128).optional(),
  history: z.array(chatMessageSchema).max(30).default([]),
  filters: z
    .object({
      categoryIds: z.array(z.string().trim().min(1).max(128)).max(20).optional(),
      documentIds: z.array(z.string().trim().min(1).max(128)).max(50).optional(),
    })
    .optional(),
});

export function validateChatRequest(payload: unknown): RagChatRequest {
  const result = ragChatRequestSchema.safeParse(payload);
  if (!result.success) {
    throw new AppError("BAD_REQUEST", 400, "Invalid /chat request payload.");
  }

  return result.data;
}
