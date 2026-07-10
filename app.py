from __future__ import annotations

import inspect
import os
import uuid
from typing import Any

import requests
import streamlit as st

from src.long_term_memory import clear_long_term_memory
from src.prompts import get_prompt_template, list_prompt_specs
from src.retriever import rebuild_vectorstore, save_uploaded_files
from src.skills import detect_skills, format_skills_for_display, list_skills
from src.vector_memory import apply_forgetting_policy, get_memory_stats

try:
    from src.retriever import search_knowledge as _search_knowledge
except ImportError:
    _search_knowledge = None


# =========================================================
# Constants
# =========================================================
RETRIEVAL_ROUGH = "rough"
RETRIEVAL_LIGHT_RERANK = "light_rerank"
RETRIEVAL_MODEL_RERANK = "model_rerank"
DEFAULT_RETRIEVAL_MODE = RETRIEVAL_LIGHT_RERANK

DEFAULT_API_BASE_URL = os.getenv("EDUPILOT_API_BASE_URL", "http://127.0.0.1:8000")

MODE_WORKFLOW = "学习闭环 Workflow"
MODE_REACT = "智能助教 ReAct"


# =========================================================
# FastAPI client helpers
# =========================================================
def _api_url(path: str) -> str:
    base_url = st.session_state.get("api_base_url", DEFAULT_API_BASE_URL).rstrip("/")
    return f"{base_url}/{path.lstrip('/')}"


def _call_api(method: str, path: str, timeout: int = 180, **kwargs) -> dict[str, Any]:
    try:
        response = requests.request(method, _api_url(path), timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("FastAPI 服务未启动或地址不正确。请先运行：uvicorn api_server:app --reload") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("FastAPI 请求超时，请检查后端日志，或适当增大 timeout。") from exc
    except requests.exceptions.HTTPError as exc:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"FastAPI 返回错误：{detail}") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"FastAPI 请求失败：{exc}") from exc


# =========================================================
# Compatibility helpers for retriever versions
# =========================================================
def _accepts_var_kwargs(func) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _supports_param(func, param_name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return param_name in signature.parameters or _accepts_var_kwargs(func)


def _call_with_supported_kwargs(func, **kwargs):
    if _accepts_var_kwargs(func):
        return func(**kwargs)
    supported_kwargs = {key: value for key, value in kwargs.items() if _supports_param(func, key)}
    return func(**supported_kwargs)


def _enable_rerank_from_mode(retrieval_mode: str) -> bool:
    return retrieval_mode in {RETRIEVAL_LIGHT_RERANK, RETRIEVAL_MODEL_RERANK}


def _search_knowledge_with_rerank(query: str, k: int, fetch_k: int, retrieval_mode: str):
    if _search_knowledge is None:
        return []

    return _call_with_supported_kwargs(
        _search_knowledge,
        query=query,
        k=int(k),
        fetch_k=int(fetch_k),
        retrieval_mode=retrieval_mode,
        enable_rerank=_enable_rerank_from_mode(retrieval_mode),
    )


def _skill_display_names(user_input: str) -> list[str]:
    return [skill.display_name for skill in detect_skills(user_input)]


def _skill_internal_names(user_input: str) -> list[str]:
    return [skill.name for skill in detect_skills(user_input)]


# =========================================================
# Streamlit page setup
# =========================================================
st.set_page_config(
    page_title="EduPilot Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    .hero-card {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(250,250,252,0.95), rgba(244,247,251,0.95));
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 760;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: rgba(49, 51, 63, 0.72);
        font-size: 1rem;
        line-height: 1.65;
    }
    .status-pill {
        display: inline-block;
        padding: 0.18rem 0.6rem;
        margin-right: 0.35rem;
        margin-top: 0.45rem;
        border-radius: 999px;
        background: rgba(49, 51, 63, 0.08);
        color: rgba(49, 51, 63, 0.82);
        font-size: 0.8rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.10);
        border-radius: 14px;
        padding: 0.7rem 0.85rem;
        background: rgba(250, 250, 252, 0.72);
    }
    .small-muted {
        color: rgba(49, 51, 63, 0.62);
        font-size: 0.88rem;
        line-height: 1.6;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# Session state
# =========================================================
def _init_state() -> None:
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "latest_result": None,
        "quiz_feedback": "",
        "qa_history": [],
        "react_agent_history": [],
        "forgetting_result": None,
        "api_base_url": DEFAULT_API_BASE_URL,
        "api_debug_result": None,
        "last_request": "Build an AI Agent project in 10 days.",
        "goal_context": "Build an AI Agent project in 10 days.",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def _reset_session() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.latest_result = None
    st.session_state.quiz_feedback = ""
    st.session_state.qa_history = []
    st.session_state.react_agent_history = []
    st.rerun()


# =========================================================
# Sidebar: compact control center
# =========================================================
with st.sidebar:
    st.markdown("## EduPilot 控制台")
    st.caption(f"Session：`{st.session_state.thread_id[:8]}`")

    mode = st.radio(
        "运行模式",
        [MODE_WORKFLOW, MODE_REACT],
        index=0,
        help="Workflow 适合完整生成学习闭环；ReAct 适合自由追问和工具调用。",
    )

    st.markdown("---")
    st.markdown("### 学习参数")

    level = st.selectbox(
        "当前基础",
        ["零基础", "Beginner", "有一点基础", "中等水平"],
        index=1,
    )

    hours = st.slider("今日学习时间 / 小时", min_value=1, max_value=12, value=4, step=1)

    enable_reflection = st.toggle(
        "Reflection 自检",
        value=True,
        help="开启后，Workflow 和 ReAct 都会执行回答质量自检。",
    )

    with st.expander("高级检索设置", expanded=False):
        retrieval_mode = st.selectbox(
            "RAG 检索模式",
            [RETRIEVAL_ROUGH, RETRIEVAL_LIGHT_RERANK, RETRIEVAL_MODEL_RERANK],
            index=[RETRIEVAL_ROUGH, RETRIEVAL_LIGHT_RERANK, RETRIEVAL_MODEL_RERANK].index(DEFAULT_RETRIEVAL_MODE),
        )
        rag_top_k = st.number_input("top_k", min_value=1, max_value=10, value=3, step=1)
        rag_fetch_k = st.number_input(
            "fetch_k",
            min_value=int(rag_top_k),
            max_value=30,
            value=max(8, int(rag_top_k) * 3),
            step=1,
        )
    st.caption(f"RAG：{retrieval_mode} · top_k={rag_top_k} · fetch_k={rag_fetch_k}")

    st.markdown("---")
    with st.expander("服务与记忆", expanded=False):
        st.session_state.api_base_url = st.text_input("FastAPI 地址", value=st.session_state.api_base_url)
        max_memory_rounds = st.number_input("Redis 读取轮数", min_value=1, max_value=50, value=10, step=1)

        col_api_1, col_api_2 = st.columns(2)
        if col_api_1.button("检查 API", use_container_width=True):
            try:
                st.session_state.api_debug_result = _call_api("GET", "/health", timeout=5)
                st.success("API 正常")
            except RuntimeError as exc:
                st.session_state.api_debug_result = {"error": str(exc)}
                st.error(str(exc))

        if col_api_2.button("查看 Redis", use_container_width=True):
            try:
                st.session_state.api_debug_result = _call_api(
                    "GET",
                    "/memory/history",
                    timeout=10,
                    params={
                        "session_id": st.session_state.thread_id,
                        "max_memory_rounds": int(max_memory_rounds),
                    },
                )
                st.success("已读取 Redis")
            except RuntimeError as exc:
                st.session_state.api_debug_result = {"error": str(exc)}
                st.error(str(exc))

        with st.expander("API / Redis 返回结果", expanded=False):
            st.write(st.session_state.api_debug_result or "暂无结果")

        confirm_clear_redis = st.checkbox("确认清空当前 Redis 短期记忆", value=False)
        if st.button("清空 Redis 记忆", disabled=not confirm_clear_redis, use_container_width=True):
            try:
                st.session_state.api_debug_result = _call_api(
                    "POST",
                    "/memory/clear",
                    timeout=10,
                    json={
                        "session_id": st.session_state.thread_id,
                        "max_memory_rounds": int(max_memory_rounds),
                    },
                )
                st.session_state.react_agent_history = []
                st.success("Redis 短期记忆已清空")
            except RuntimeError as exc:
                st.error(str(exc))

    with st.expander("知识库", expanded=False):
        uploaded_files = st.file_uploader(
            "上传学习资料（.md / .txt）",
            type=["md", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("保存资料", use_container_width=True):
            saved_files = save_uploaded_files(uploaded_files)
            if saved_files:
                st.success(f"已保存 {len(saved_files)} 个文件")
                st.write(saved_files)
            else:
                st.warning("没有保存成功的文件，请检查文件类型。")

        if st.button("重建知识库", use_container_width=True):
            with st.spinner("正在重建 Chroma 知识库..."):
                vectorstore_result = rebuild_vectorstore()
            if vectorstore_result.get("success"):
                st.success(f"{vectorstore_result.get('message')}，切片数：{vectorstore_result.get('chunk_count')}")
            else:
                st.warning(vectorstore_result.get("message"))

    with st.expander("长期记忆", expanded=False):
        memory_stats = get_memory_stats()
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("有效", memory_stats.get("active_count", memory_stats.get("count", 0)))
        col_m2.metric("归档", memory_stats.get("archived_count", 0))
        st.caption(f"总数：{memory_stats.get('count', 0)} 条")

        if st.button("执行遗忘检查", use_container_width=True):
            with st.spinner("正在检查长期记忆..."):
                st.session_state.forgetting_result = apply_forgetting_policy(max_idle_days=30, max_access_count=1)
            st.success("遗忘检查完成")

        if st.session_state.forgetting_result:
            st.write(st.session_state.forgetting_result)

        confirm_clear_memory = st.checkbox("确认清空长期记忆", value=False)
        if st.button("清空长期记忆", disabled=not confirm_clear_memory, use_container_width=True):
            clear_long_term_memory()
            st.session_state.forgetting_result = None
            st.success("长期记忆已清空")
            st.rerun()

    st.markdown("---")
    if st.button("开启新学习会话", use_container_width=True):
        _reset_session()


# =========================================================
# Header
# =========================================================
st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">🎓 EduPilot Agent</div>
    <div class="hero-subtitle">
        Personalized Learning Agent based on <b>LangGraph Workflow</b>, <b>ReAct Tool Calling</b>,
        <b>RAG</b>, <b>Reflection</b>, <b>Redis Short-term Memory</b> and <b>Vector Long-term Memory</b>.
    </div>
    <span class="status-pill">LangGraph Workflow</span>
    <span class="status-pill">ReAct Agent</span>
    <span class="status-pill">RAG + Rerank</span>
    <span class="status-pill">Reflection</span>
    <span class="status-pill">Memory</span>
</div>
""",
    unsafe_allow_html=True,
)

col_status_1, col_status_2, col_status_3, col_status_4 = st.columns(4)
col_status_1.metric("Mode", "Workflow" if mode == MODE_WORKFLOW else "ReAct")
col_status_2.metric("Reflection", "On" if enable_reflection else "Off")
col_status_3.metric("Study Hours", f"{hours}h")
col_status_4.metric("RAG", retrieval_mode)


# =========================================================
# Unified input area
# =========================================================
main_placeholder = (
    "例如：我想学习 LangGraph 的短期记忆机制，帮我生成今天的学习闭环。"
    if mode == MODE_WORKFLOW
    else "例如：老师，请根据我的长期记忆，总结 EduPilot 项目的当前进度和下一步任务。"
)

with st.container(border=True):
    st.markdown("### 任务输入")
    user_request = st.text_area(
        "只需要在这里输入你的学习目标或问题",
        value=st.session_state.last_request,
        height=150,
        placeholder=main_placeholder,
        label_visibility="collapsed",
    )

    if mode == MODE_REACT:
        with st.expander("ReAct 项目背景 / Goal Context", expanded=False):
            st.session_state.goal_context = st.text_area(
                "传给 ReAct Agent 的学习目标背景",
                value=st.session_state.goal_context,
                height=90,
                help="ReAct 模式下，上方主输入作为 question；这里作为 goal/context。一般保持默认即可。",
            )

    col_run, col_skill = st.columns([1, 3])
    run_label = "生成完整学习闭环" if mode == MODE_WORKFLOW else "发送给智能助教"
    run_button = col_run.button(run_label, type="primary", use_container_width=True)

    matched_skill_names = _skill_display_names(user_request) if user_request.strip() else []
    col_skill.caption("命中 Skill：" + ("、".join(matched_skill_names) if matched_skill_names else "暂无"))


# =========================================================
# Main run logic
# =========================================================
if run_button:
    if not user_request.strip():
        st.warning("请先输入学习目标或问题。")
        st.stop()

    st.session_state.last_request = user_request

    if mode == MODE_WORKFLOW:
        payload = {
            "session_id": st.session_state.thread_id,
            "goal": user_request,
            "level": level,
            "hours": int(hours),
            "enable_reflection": bool(enable_reflection),
            "rag_top_k": int(rag_top_k),
            "rag_fetch_k": int(rag_fetch_k),
            "retrieval_mode": retrieval_mode,
        }

        with st.spinner("EduPilot 正在生成学习计划、导师讲解、小测、复盘与 Reflection..."):
            try:
                workflow_response = _call_api("POST", "/workflow/run", json=payload, timeout=240)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        st.session_state.latest_result = workflow_response.get("result", {})
        st.session_state.quiz_feedback = ""
        st.success("Workflow 运行完成。")
        st.rerun()

    else:
        matched_skill_internal_names = _skill_internal_names(user_request)
        payload = {
            "session_id": st.session_state.thread_id,
            "question": user_request,
            "goal": st.session_state.goal_context or user_request,
            "level": level,
            "hours": int(hours),
            "enable_reflection": bool(enable_reflection),
            "rag_top_k": int(rag_top_k),
            "rag_fetch_k": int(rag_fetch_k),
            "retrieval_mode": retrieval_mode,
            "max_memory_rounds": int(max_memory_rounds),
        }

        with st.spinner("ReAct Agent 正在选择工具、生成回答并执行 Reflection..."):
            try:
                react_result = _call_api("POST", "/react/chat", json=payload, timeout=240)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        st.session_state.react_agent_history.append(
            {
                "question": user_request,
                "final_answer": react_result.get("final_answer", ""),
                "draft_answer": react_result.get("draft_answer", ""),
                "reflection": react_result.get("reflection", ""),
                "used_reflection": react_result.get("used_reflection", False),
                "trace": react_result.get("trace", []),
                "matched_skill_names": matched_skill_names,
                "matched_skills": react_result.get("matched_skills", matched_skill_internal_names),
                "recommended_tools": react_result.get("recommended_tools", []),
                "routing_reason": react_result.get("routing_reason", ""),
                "skill_route": react_result.get("skill_route", {}),
                "redis_memory": react_result.get("redis_memory", {}),
                "memory_result": react_result.get("long_term_memory_result", {}),
            }
        )
        st.rerun()


# =========================================================
# Workflow results
# =========================================================
if mode == MODE_WORKFLOW:
    result = st.session_state.latest_result

    if not result:
        st.info("输入学习目标后，点击“生成完整学习闭环”。建议面试展示时先跑 Workflow，再切换到 ReAct 做自由追问。")
    else:
        st.markdown("## 学习闭环结果")

        tab_overview, tab_learning, tab_practice, tab_memory, tab_debug = st.tabs(
            ["总览", "学习内容", "练习与追问", "记忆", "调试细节"]
        )

        with tab_overview:
            final_answer = result.get("final_answer") or result.get("workflow_draft_answer")
            if final_answer:
                st.markdown(final_answer)
            else:
                st.markdown("### 今日学习计划")
                st.markdown(result.get("learning_plan", "暂无结果。"))
                st.markdown("### 导师讲解")
                st.markdown(result.get("tutor_explanation", "暂无结果。"))
                st.markdown("### 复盘验收")
                st.markdown(result.get("review", "暂无结果。"))

            with st.expander("本轮命中的 Skill", expanded=False):
                st.markdown(format_skills_for_display(st.session_state.last_request))

        with tab_learning:
            st.markdown("### 今日学习计划")
            st.markdown(result.get("learning_plan", "暂无学习计划。"))
            st.markdown("---")
            st.markdown("### 导师讲解")
            st.markdown(result.get("tutor_explanation", "暂无导师讲解。"))
            st.markdown("---")
            st.markdown("### 复盘验收")
            st.markdown(result.get("review", "暂无复盘。"))

        with tab_practice:
            st.markdown("### 本轮小测")
            st.markdown(result.get("quiz", "暂无小测。"))

            with st.form("quiz_grade_form", clear_on_submit=False):
                student_answer = st.text_area(
                    "作答区",
                    height=180,
                    placeholder="请按题号作答，例如：\n1. ...\n2. ...\n3. ...",
                    key="student_answer",
                )
                submit_grade = st.form_submit_button("提交并批改")

            if submit_grade:
                if not student_answer.strip():
                    st.warning("请先填写答案。")
                else:
                    with st.spinner("正在批改答案..."):
                        try:
                            grade_response = _call_api(
                                "POST",
                                "/quiz/grade",
                                json={
                                    "session_id": st.session_state.thread_id,
                                    "goal": st.session_state.last_request,
                                    "level": level,
                                    "quiz": result.get("quiz", ""),
                                    "student_answer": student_answer,
                                    "tutor_explanation": result.get("tutor_explanation", ""),
                                },
                                timeout=180,
                            )
                        except RuntimeError as exc:
                            st.error(str(exc))
                            st.stop()
                    st.session_state.quiz_feedback = grade_response.get("feedback", "")
                    st.success("批改完成。")

            if st.session_state.quiz_feedback:
                st.markdown("#### 批改反馈")
                st.markdown(st.session_state.quiz_feedback)

            st.markdown("---")
            st.markdown("### 继续追问")

            for item in st.session_state.qa_history:
                with st.chat_message("user"):
                    st.markdown(item["question"])
                with st.chat_message("assistant"):
                    st.markdown(item["answer"])

            with st.form("followup_qa_form", clear_on_submit=True):
                followup_question = st.text_area(
                    "追问区",
                    height=110,
                    placeholder="例如：老师，为什么 checkpointer 和 thread_id 要一起用？",
                    label_visibility="collapsed",
                )
                submit_followup = st.form_submit_button("提交追问")

            if submit_followup:
                if not followup_question.strip():
                    st.warning("请先输入追问。")
                else:
                    with st.spinner("正在回答追问..."):
                        try:
                            qa_response = _call_api(
                                "POST",
                                "/qa/followup",
                                json={
                                    "session_id": st.session_state.thread_id,
                                    "question": followup_question,
                                    "goal": st.session_state.last_request,
                                    "level": level,
                                    "learning_plan": result.get("learning_plan", ""),
                                    "tutor_explanation": result.get("tutor_explanation", ""),
                                    "retrieved_context": result.get("retrieved_context", ""),
                                    "qa_history": st.session_state.qa_history,
                                    "rag_top_k": int(rag_top_k),
                                    "rag_fetch_k": int(rag_fetch_k),
                                    "retrieval_mode": retrieval_mode,
                                },
                                timeout=180,
                            )
                        except RuntimeError as exc:
                            st.error(str(exc))
                            st.stop()
                    st.session_state.qa_history.append(
                        {"question": followup_question, "answer": qa_response.get("answer", "")}
                    )
                    st.rerun()

        with tab_memory:
            st.markdown("### 长期记忆召回")
            st.markdown(result.get("long_term_memory", "暂无相关长期记忆。"))

            st.markdown("### 本轮记忆写入")
            memory_result = result.get("memory_result", {})
            if memory_result:
                if memory_result.get("saved"):
                    st.success("本轮内容已写入长期记忆。")
                elif memory_result.get("action") == "discard":
                    st.info("Memory Reflection 判断本轮内容不需要写入长期记忆。")
                else:
                    st.warning("长期记忆未写入或写入异常。")
                st.write(memory_result)
            else:
                st.info("暂无记忆写入结果。")

            st.markdown("### 当前记忆库统计")
            st.write(get_memory_stats())

        with tab_debug:
            with st.expander("Reflection 过程", expanded=True):
                st.markdown("#### Tutor Reflection")
                st.markdown(result.get("tutor_reflection") or "未启用或暂无结果。")
                st.markdown("#### Quiz Reflection")
                st.markdown(result.get("quiz_reflection") or "未启用或暂无结果。")
                st.markdown("#### Reviewer Reflection")
                st.markdown(result.get("review_reflection") or "未启用或暂无结果。")
                st.markdown("#### Global Workflow Reflection")
                st.markdown(result.get("workflow_reflection") or "未启用或暂无结果。")

            with st.expander("RAG / Memory 检索上下文", expanded=False):
                st.markdown(result.get("retrieved_context", "暂无检索上下文。"))

            with st.expander("原始 State 数据", expanded=False):
                st.write(result)


# =========================================================
# ReAct results
# =========================================================
else:
    st.markdown("## 智能助教对话")

    if not st.session_state.react_agent_history:
        st.info("输入问题后，ReAct Agent 会自动判断是否调用 RAG、Planner、Tutor、Quiz、QA、Reviewer 或长期记忆工具。")
    else:
        for item in st.session_state.react_agent_history:
            with st.chat_message("user"):
                st.markdown(item["question"])

            with st.chat_message("assistant"):
                st.markdown(item.get("final_answer", ""))

            with st.expander("本轮过程与调试信息", expanded=False):
                col_r1, col_r2 = st.columns(2)
                col_r1.markdown("**Matched Skills**")
                col_r1.write(item.get("matched_skill_names") or item.get("matched_skills") or [])
                col_r2.markdown("**Recommended Tools**")
                col_r2.write(item.get("recommended_tools") or [])

                if item.get("routing_reason"):
                    st.markdown("**Routing Reason**")
                    st.markdown(item.get("routing_reason"))

                if item.get("draft_answer") or item.get("reflection"):
                    st.markdown("#### Reflection")
                    st.markdown("**Draft Answer**")
                    st.markdown(item.get("draft_answer") or "暂无草稿。")
                    st.markdown("**Reflection**")
                    st.markdown(item.get("reflection") or "暂无 Reflection。")

                if item.get("trace"):
                    st.markdown("#### Tool Trace")
                    for step in item["trace"]:
                        step_type = step.get("type", "")
                        step_name = step.get("name", "")
                        step_content = step.get("content", "")
                        if step_type == "tool_call":
                            st.markdown(f"调用工具：`{step_name}`")
                        elif step_type == "tool_result":
                            st.markdown(f"工具返回：`{step_name}`")
                        else:
                            st.markdown(f"步骤：`{step_type}`")
                        st.code(step_content, language="text")

                if item.get("redis_memory"):
                    st.markdown("#### Redis Memory")
                    st.write(item.get("redis_memory"))

                if item.get("memory_result"):
                    st.markdown("#### Long-term Memory Write")
                    st.write(item.get("memory_result"))


# =========================================================
# Advanced engineering panels
# =========================================================
st.markdown("---")
with st.expander("工程化展示：Prompt Registry / Skill Registry / RAG Debug", expanded=False):
    registry_tab, rag_tab = st.tabs(["Prompt & Skill", "RAG Debug"])

    with registry_tab:
        col_s, col_p = st.columns(2)

        with col_s:
            st.markdown("### Skill Registry")
            st.markdown(format_skills_for_display(st.session_state.last_request))
            with st.expander("查看全部 Skill", expanded=False):
                for skill in list_skills():
                    st.markdown(f"#### {skill['display_name']}")
                    st.write(skill["description"])
                    st.markdown(f"关联 Prompt：`{skill['prompt_name']}`")
                    st.markdown("关联 Tools：" + ", ".join(f"`{tool}`" for tool in skill["related_tools"]))
                    st.caption(f"示例：{skill['demo_query']}")
                    st.divider()

        with col_p:
            st.markdown("### Prompt Registry")
            prompt_specs = list_prompt_specs()
            prompt_names = [item["name"] for item in prompt_specs]
            selected_prompt = st.selectbox("选择 Prompt", prompt_names)
            selected_spec = next((item for item in prompt_specs if item["name"] == selected_prompt), None)
            if selected_spec:
                st.markdown(f"版本：`{selected_spec['version']}`")
                st.markdown(f"说明：{selected_spec['description']}")
                st.markdown("变量：" + ", ".join(f"`{var}`" for var in selected_spec["variables"]))
            with st.expander("查看 Prompt 模板", expanded=False):
                st.code(get_prompt_template(selected_prompt), language="text")

    with rag_tab:
        st.markdown("### RAG 召回 / Rerank 调试")
        debug_query = st.text_input(
            "检索 Query",
            value=st.session_state.last_request,
            placeholder="例如：LangGraph 的 thread_id 有什么作用？",
        )

        if st.button("运行 RAG 检索调试"):
            if not debug_query.strip():
                st.warning("请先输入检索 Query。")
            else:
                with st.spinner("正在执行 RAG 检索与 rerank..."):
                    debug_results = _search_knowledge_with_rerank(
                        query=debug_query,
                        k=int(rag_top_k),
                        fetch_k=int(rag_fetch_k),
                        retrieval_mode=retrieval_mode,
                    )

                if not debug_results:
                    st.warning("没有检索到结果。请确认已经上传资料并重建知识库。")
                else:
                    st.success(f"检索完成，共返回 {len(debug_results)} 条结果。")
                    summary_rows = []
                    for i, item in enumerate(debug_results, start=1):
                        summary_rows.append(
                            {
                                "rank": i,
                                "source": item.get("source"),
                                "chunk_id": item.get("chunk_id"),
                                "mode": item.get("retrieval_mode"),
                                "initial_rank": item.get("initial_rank"),
                                "distance": item.get("distance"),
                                "vector_score": item.get("vector_score"),
                                "keyword_score": item.get("keyword_score"),
                                "rerank_score": item.get("rerank_score"),
                                "model_rerank_score": item.get("model_rerank_score"),
                            }
                        )
                    st.dataframe(summary_rows, use_container_width=True)

                    for i, item in enumerate(debug_results, start=1):
                        with st.expander(f"结果 {i}：{item.get('source')} / chunk {item.get('chunk_id')}"):
                            st.write(
                                {
                                    "retrieval_mode": item.get("retrieval_mode"),
                                    "initial_rank": item.get("initial_rank"),
                                    "distance": item.get("distance"),
                                    "vector_score": item.get("vector_score"),
                                    "keyword_score": item.get("keyword_score"),
                                    "rerank_score": item.get("rerank_score"),
                                    "model_rerank_score": item.get("model_rerank_score"),
                                    "model_rerank_error": item.get("model_rerank_error"),
                                }
                            )
                            st.markdown(item.get("content", ""))
