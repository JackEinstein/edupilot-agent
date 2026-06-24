from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.long_term_memory import record_react_memory
from src.react_agent import run_react_agent
from src.redis_memory import (
    DEFAULT_MAX_ROUNDS,
    append_react_turn,
    check_redis,
    load_react_history,
)
from src.retriever import DEFAULT_RETRIEVAL_MODE
from src.short_term_memory import format_react_agent_history
from src.skills import detect_skills


app = FastAPI(
    title="EduPilot Agent API",
    description="FastAPI service layer for EduPilot ReAct chat with Redis short-term memory.",
    version="0.1.0",
)


class ReactChatRequest(BaseModel):
    session_id: str = Field(default="default", description="Client-side conversation/session id")
    question: str = Field(..., min_length=1, description="Student question for ReAct Agent")

    goal: str = Field(default="Build an AI Agent project in 10 days.")
    level: str = Field(default="Beginner")
    hours: int = Field(default=4, ge=1, le=12)

    enable_reflection: bool = True
    rag_top_k: int = Field(default=3, ge=1, le=10)
    rag_fetch_k: int = Field(default=8, ge=1, le=30)
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE
    max_memory_rounds: int = Field(default=DEFAULT_MAX_ROUNDS, ge=1, le=50)


class ReactChatResponse(BaseModel):
    session_id: str
    final_answer: str
    draft_answer: str = ""
    reflection: str = ""
    used_reflection: bool = False
    trace: list[dict[str, Any]] = []
    matched_skills: list[str] = []
    redis_memory: dict[str, Any] = {}
    long_term_memory_result: dict[str, Any] = {}


@app.get("/health")
def health() -> dict[str, Any]:
    redis_status = check_redis()
    return {
        "status": "ok" if redis_status.get("ok") else "degraded",
        "service": "edupilot-api",
        "redis": redis_status,
    }


@app.post("/react/chat", response_model=ReactChatResponse)
def react_chat(payload: ReactChatRequest) -> ReactChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    redis_history = load_react_history(
        session_id=payload.session_id,
        max_rounds=payload.max_memory_rounds,
    )
    short_term_memory = format_react_agent_history(
        redis_history,
        max_rounds=payload.max_memory_rounds,
    )
    matched_skills = [skill.name for skill in detect_skills(question)]

    context = {
        "goal": payload.goal,
        "level": payload.level,
        "hours": payload.hours,
        "qa_history": [],
        "react_agent_history": redis_history,
        "short_term_memory": short_term_memory,
        "rag_top_k": payload.rag_top_k,
        "rag_fetch_k": payload.rag_fetch_k,
        "retrieval_mode": payload.retrieval_mode,
        "matched_skills": matched_skills,
    }

    try:
        react_result = run_react_agent(
            context=context,
            question=question,
            thread_id=payload.session_id,
            enable_reflection=payload.enable_reflection,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ReAct Agent failed: {exc}") from exc

    final_answer = react_result.get("final_answer", "")
    draft_answer = react_result.get("draft_answer", "")
    trace = react_result.get("trace", [])

    try:
        redis_memory_result = append_react_turn(
            session_id=payload.session_id,
            question=question,
            final_answer=final_answer,
            draft_answer=draft_answer,
            trace=trace,
            matched_skills=matched_skills,
            max_rounds=payload.max_memory_rounds,
        )
    except Exception as exc:
        redis_memory_result = {
            "saved": False,
            "error": str(exc),
            "history_rounds_before_request": len(redis_history),
        }

    try:
        long_term_memory_result = record_react_memory(
            goal=payload.goal,
            level=payload.level,
            question=question,
            answer=final_answer,
        )
    except Exception as exc:
        long_term_memory_result = {
            "saved": False,
            "action": "error",
            "reason": str(exc),
        }

    return ReactChatResponse(
        session_id=payload.session_id,
        final_answer=final_answer,
        draft_answer=draft_answer,
        reflection=react_result.get("reflection", ""),
        used_reflection=react_result.get("used_reflection", False),
        trace=trace,
        matched_skills=matched_skills,
        redis_memory=redis_memory_result,
        long_term_memory_result=long_term_memory_result,
    )
