import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import certifi
import jwt
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "lawkh")
MONGODB_TIMEOUT_MS = int(os.getenv("MONGODB_TIMEOUT_MS", "5000"))
JWT_SECRET = os.getenv("JWT_SECRET", "lawkh_local_dev_change_me")
JWT_ACCESS_TOKEN_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_MINUTES", "1440"))
PASSWORD_RESET_TOKEN_MINUTES = int(os.getenv("PASSWORD_RESET_TOKEN_MINUTES", "30"))

_client: MongoClient | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get_db():
    global _client
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set.")
    if _client is None:
        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=MONGODB_TIMEOUT_MS,
            connectTimeoutMS=MONGODB_TIMEOUT_MS,
            socketTimeoutMS=MONGODB_TIMEOUT_MS,
            tlsCAFile=certifi.where(),
        )
        _client.admin.command("ping")
    return _client[MONGODB_DB]


def users_collection() -> Collection:
    collection = get_db()["users"]
    collection.create_index("email", unique=True)
    return collection


def chats_collection() -> Collection:
    collection = get_db()["chats"]
    collection.create_index([("updatedAt", -1)])
    return collection


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(user["_id"]),
        "name": user.get("name", ""),
        "email": user.get("email", ""),
        "preferences": user.get("preferences", {"darkMode": True}),
    }


def create_access_token(user: dict[str, Any]) -> str:
    expires_at = _now() + timedelta(minutes=JWT_ACCESS_TOKEN_MINUTES)
    payload = {
        "sub": str(user["_id"]),
        "email": user.get("email", ""),
        "exp": expires_at,
        "iat": _now(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    try:
        return users_collection().find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None


def create_reset_token(email: str) -> dict[str, str] | None:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires_at = _now() + timedelta(minutes=PASSWORD_RESET_TOKEN_MINUTES)
    result = users_collection().update_one(
        {"email": normalize_email(email)},
        {
            "$set": {
                "passwordResetTokenHash": token_hash,
                "passwordResetExpiresAt": expires_at,
                "updatedAt": _now(),
            }
        },
    )
    if result.matched_count == 0:
        return None
    return {"token": token, "resetUrl": f"lawkh://reset-password?token={token}"}


def reset_password(token: str, password: str) -> bool:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    result = users_collection().update_one(
        {
            "passwordResetTokenHash": token_hash,
            "passwordResetExpiresAt": {"$gt": _now()},
        },
        {
            "$set": {
                "passwordHash": hash_password(password),
                "updatedAt": _now(),
            },
            "$unset": {
                "passwordResetTokenHash": "",
                "passwordResetExpiresAt": "",
            },
        },
    )
    return result.modified_count == 1
