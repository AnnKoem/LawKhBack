import type { VercelRequest, VercelResponse } from "@vercel/node";
import { getAllowedOrigin, getConfig } from "../src/config.js";
import { toErrorResponse, AppError } from "../src/errors.js";
import { setCorsHeaders } from "../src/http.js";
import { runRagChat } from "../src/rag.js";
import { validateChatRequest } from "../src/validation.js";

export default async function handler(req: VercelRequest, res: VercelResponse): Promise<void> {
  setCorsHeaders(req, res, getAllowedOrigin());

  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }

  try {
    const config = getConfig();

    if (req.method !== "POST") {
      throw new AppError("METHOD_NOT_ALLOWED", 405, "Only POST is allowed for /chat.");
    }

    const payload = validateChatRequest(req.body);
    const result = await runRagChat(payload, config);
    res.status(200).json(result);
  } catch (error) {
    const { status, body } = toErrorResponse(error);
    res.status(status).json(body);
  }
}
