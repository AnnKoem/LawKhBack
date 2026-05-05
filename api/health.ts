import type { VercelRequest, VercelResponse } from "@vercel/node";
import { getAllowedOrigin } from "../src/config.js";
import { toErrorResponse } from "../src/errors.js";
import { setCorsHeaders } from "../src/http.js";

export default function handler(req: VercelRequest, res: VercelResponse): void {
  try {
    setCorsHeaders(req, res, getAllowedOrigin());

    if (req.method === "OPTIONS") {
      res.status(204).end();
      return;
    }

    res.status(200).json({
      ok: true,
      service: "law-kh-backend",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    const { status, body } = toErrorResponse(error);
    res.status(status).json(body);
  }
}
