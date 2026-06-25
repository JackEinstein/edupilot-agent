from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.graph import run_graph
from src.long_term_memory import record_react_memory
from src.qa import answer_followup_question
from src.quiz import grade_quiz
from src.react_agent import run_react_agent
from src.redis_memory import (
    DEFAULT_MAX_ROUNDS,
    append_react_turn,
    check_redis,
    clear_react_history,
    load_react_history,
)
from src.retriever import DEFAULT_RETRIEVAL_MODE
from src.short_term_memory import format_react_agent_history
from src.skills import detect_skills


app = FastAPI(
    title="EduPilot Agent API",
    description=(
        "FastAPI service layer for EduPilot Workflow, Follow-up QA, "
        "Quiz grading, ReAct Agent and Redis short-term memory."
    ),
    version="0.2.0",
)


def _to_jsonable(value: Any) -> Any:
    """
    Convert LangGraph / LangChain objects into JSON-safe values.

    LangGraph result contains HumanMessage / AIMessage objects in `messages`.
    FastAPI cannot directly serialize these objects, so we convert them into
    simple dictionaries.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "content"):
        return {
            "type": value.__class__.__name__,
            "content": getattr(value, "content", ""),
        }

    return str(value)


class WorkflowRunRequest(BaseModel):
    session_id: str = Field(default="default", description="Client-side workflow/session id")
    goal: str = Field(..., min_length=1, description="Student learning goal")
    level: str = Field(default="Beginner")
    hours: int = Field(default=4, ge=1, le=12)

    enable_reflection: bool = True
    rag_top_k: int = Field(default=3, ge=1, le=10)
    rag_fetch_k: int = Field(default=8, ge=1, le=30)
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE


class WorkflowRunResponse(BaseModel):
    session_id: str
    result: dict[str, Any]


class FollowupQARequest(BaseModel):
    session_id: str = Field(default="default")
    question: str = Field(..., min_length=1)

    goal: str = Field(..., min_length=1)
    level: str = Field(default="Beginner")
    learning_plan: str = ""
    tutor_explanation: str = ""
    retrieved_context: str = ""
    qa_history: list[dict[str, str]] = Field(default_factory=list)

    rag_top_k: int = Field(default=3, ge=1, le=10)
    rag_fetch_k: int = Field(default=8, ge=1, le=30)
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE


class FollowupQAResponse(BaseModel):
    session_id: str
    answer: str


class QuizGradeRequest(BaseModel):
    session_id: str = Field(default="default")
    goal: str = Field(..., min_length=1)
    level: str = Field(default="Beginner")
    tutor_explanation: str = ""
    quiz: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)


class QuizGradeResponse(BaseModel):
    session_id: str
    feedback: str


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
    trace: list[dict[str, Any]] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    redis_memory: dict[str, Any] = Field(default_factory=dict)
    long_term_memory_result: dict[str, Any] = Field(default_factory=dict)


class MemorySessionRequest(BaseModel):
    session_id: str = Field(default="default", description="Client-side conversation/session id")
    max_memory_rounds: int = Field(default=DEFAULT_MAX_ROUNDS, ge=1, le=50)


@app.get("/health")
def health() -> dict[str, Any]:
    redis_status = check_redis()
    return {
        "status": "ok" if redis_status.get("ok") else "degraded",
        "service": "edupilot-api",
        "redis": redis_status,
    }


@app.get("/memory/history")
def memory_history(
    session_id: str = Query(default="default"),
    max_memory_rounds: int = Query(default=DEFAULT_MAX_ROUNDS, ge=1, le=50),
) -> dict[str, Any]:
    """Return recent Redis ReAct memory for one session. Useful for Streamlit debug panel."""

    history = load_react_history(
        session_id=session_id,
        max_rounds=max_memory_rounds,
    )
    return {
        "session_id": session_id,
        "max_memory_rounds": max_memory_rounds,
        "history_rounds": len(history),
        "history": history,
    }


@app.post("/memory/clear")
def memory_clear(payload: MemorySessionRequest) -> dict[str, Any]:
    """Clear Redis ReAct short-term memory for one session."""

    try:
        result = clear_react_history(payload.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Redis memory clear failed: {exc}") from exc

    return {
        "session_id": payload.session_id,
        **result,
    }


@app.post("/workflow/run", response_model=WorkflowRunResponse)
def workflow_run(payload: WorkflowRunRequest) -> WorkflowRunResponse:
    """Run the fixed LangGraph workflow for one Streamlit session."""

    goal = payload.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal cannot be empty")

    try:
        result = run_graph(
            goal=goal,
            level=payload.level,
            hours=payload.hours,
            thread_id=payload.session_id,
            enable_reflection=payload.enable_reflection,
            rag_top_k=payload.rag_top_k,
            rag_fetch_k=payload.rag_fetch_k,
            retrieval_mode=payload.retrieval_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow run failed: {exc}") from exc

    return WorkflowRunResponse(
        session_id=payload.session_id,
        result=_to_jsonable(result),
    )


@app.post("/qa/followup", response_model=FollowupQAResponse)
def qa_followup(payload: FollowupQARequest) -> FollowupQAResponse:
    """Answer a follow-up question using current workflow context and RAG."""

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        answer = answer_followup_question(
            goal=payload.goal,
            level=payload.level,
            question=question,
            learning_plan=payload.learning_plan,
            tutor_explanation=payload.tutor_explanation,
            retrieved_context=payload.retrieved_context,
            qa_history=payload.qa_history,
            rag_top_k=payload.rag_top_k,
            rag_fetch_k=payload.rag_fetch_k,
            retrieval_mode=payload.retrieval_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Follow-up QA failed: {exc}") from exc

    return FollowupQAResponse(
        session_id=payload.session_id,
        answer=answer,
    )


@app.post("/quiz/grade", response_model=QuizGradeResponse)
def quiz_grade(payload: QuizGradeRequest) -> QuizGradeResponse:
    """Grade a student's answer for the current workflow quiz."""

    try:
        feedback = grade_quiz(
            goal=payload.goal,
            level=payload.level,
            quiz=payload.quiz,
            student_answer=payload.student_answer,
            tutor_explanation=payload.tutor_explanation,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Quiz grading failed: {exc}") from exc

    return QuizGradeResponse(
        session_id=payload.session_id,
        feedback=feedback,
    )


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
