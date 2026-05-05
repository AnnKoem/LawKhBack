import type { VercelRequest, VercelResponse } from "@vercel/node";

export function setCorsHeaders(req: VercelRequest, res: VercelResponse, allowedOrigin?: string): void {
  const requestOrigin = req.headers.origin;
  const origin = allowedOrigin || requestOrigin || "*";

  res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

export function createTimeoutSignal(timeoutMs: number): AbortSignal {
  return AbortSignal.timeout(timeoutMs);
}
