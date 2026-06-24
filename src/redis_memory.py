"""
Redis-backed short-term memory for EduPilot API sessions.

This module is intentionally independent from Streamlit session_state and
LangGraph's in-process checkpointer. It stores only recent ReAct chat turns so
FastAPI requests with the same session_id can recover short-term context.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

try:
    import redis
except ImportError:  # keep Streamlit usable before redis dependency is installed
    redis = None


DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_MAX_ROUNDS = int(os.getenv("EDUPILOT_REDIS_MAX_ROUNDS", "10"))
KEY_PREFIX = os.getenv("EDUPILOT_REDIS_KEY_PREFIX", "edupilot:session")

_client = None


def _sanitize_session_id(session_id: str) -> str:
    """
    Keep Redis keys predictable and avoid accidental separators/control chars.
    """

    safe = "".join(
        char if char.isalnum() or char in {"-", "_", ":"} else "_"
        for char in str(session_id or "default")
    )
    return safe[:120] or "default"


def _react_key(session_id: str) -> str:
    return f"{KEY_PREFIX}:{_sanitize_session_id(session_id)}:react"


def get_redis_client():
    """
    Lazily create a Redis client.

    decode_responses=True lets lrange return str instead of bytes, which keeps
    JSON serialization simple.
    """

    global _client

    if redis is None:
        raise RuntimeError("redis package is not installed. Please run: pip install redis")

    if _client is None:
        _client = redis.Redis.from_url(
            DEFAULT_REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

    return _client


def check_redis() -> dict[str, Any]:
    """
    Return Redis health information for /health.
    """

    try:
        client = get_redis_client()
        ok = bool(client.ping())
        return {
            "ok": ok,
            "url": DEFAULT_REDIS_URL,
            "max_rounds": DEFAULT_MAX_ROUNDS,
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": DEFAULT_REDIS_URL,
            "max_rounds": DEFAULT_MAX_ROUNDS,
            "error": str(exc),
        }


def load_react_history(session_id: str, max_rounds: int | None = None) -> list[dict[str, Any]]:
    """
    Load recent ReAct chat turns from Redis in chronological order.
    """

    max_rounds = int(max_rounds or DEFAULT_MAX_ROUNDS)

    try:
        raw_items = get_redis_client().lrange(_react_key(session_id), -max_rounds, -1)
    except Exception:
        return []

    history: list[dict[str, Any]] = []
    for raw in raw_items:
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if isinstance(item, dict):
            history.append(item)

    return history


def append_react_turn(
    session_id: str,
    question: str,
    final_answer: str,
    draft_answer: str = "",
    trace: list[dict[str, Any]] | None = None,
    matched_skills: list[str] | None = None,
    max_rounds: int | None = None,
) -> dict[str, Any]:
    """
    Append one ReAct chat turn and keep only the latest N rounds.
    """

    max_rounds = int(max_rounds or DEFAULT_MAX_ROUNDS)
    key = _react_key(session_id)

    item = {
        "question": question,
        "final_answer": final_answer,
        "draft_answer": draft_answer,
        "trace": trace or [],
        "matched_skills": matched_skills or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    client = get_redis_client()
    client.rpush(key, json.dumps(item, ensure_ascii=False))
    client.ltrim(key, -max_rounds, -1)
    ttl_seconds = int(os.getenv("EDUPILOT_REDIS_TTL_SECONDS", "0"))
    if ttl_seconds > 0:
        client.expire(key, ttl_seconds)

    return {
        "saved": True,
        "key": key,
        "max_rounds": max_rounds,
        "history_rounds": client.llen(key),
    }


def clear_react_history(session_id: str) -> dict[str, Any]:
    """
    Clear one session's ReAct short-term memory. Kept for local debugging/tests.
    """

    key = _react_key(session_id)
    deleted = get_redis_client().delete(key)
    return {"deleted": int(deleted), "key": key}
