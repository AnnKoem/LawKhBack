export type ErrorCode =
  | "BAD_REQUEST"
  | "METHOD_NOT_ALLOWED"
  | "CONFIG_ERROR"
  | "EMBEDDING_ERROR"
  | "VECTOR_DB_ERROR"
  | "MODEL_TIMEOUT"
  | "MODEL_ERROR"
  | "INTERNAL_ERROR";

export class AppError extends Error {
  constructor(
    public readonly code: ErrorCode,
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "AppError";
  }
}

export function toErrorResponse(error: unknown): {
  status: number;
  body: { error: { code: ErrorCode; message: string } };
} {
  if (error instanceof AppError) {
    return {
      status: error.status,
      body: {
        error: {
          code: error.code,
          message: error.message,
        },
      },
    };
  }

  return {
    status: 500,
    body: {
      error: {
        code: "INTERNAL_ERROR",
        message: "An unexpected error occurred.",
      },
    },
  };
}
