import os

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RAG_API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://localhost:8000")
RAG_REQUEST_TIMEOUT_SECONDS = float(os.getenv("RAG_REQUEST_TIMEOUT_SECONDS", "60"))

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set. Create a .env file from .env.example.")
