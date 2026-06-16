import inspect
import uuid

import streamlit as st

from src.graph import run_graph
from src.long_term_memory import clear_long_term_memory, record_react_memory
from src.qa import answer_followup_question
from src.quiz import grade_quiz
from src.react_agent import run_react_agent
from src.retriever import rebuild_vectorstore, save_uploaded_files

try:
    from src.retriever import search_knowledge as _search_knowledge
except ImportError:
    _search_knowledge = None
from src.vector_memory import apply_forgetting_policy, get_memory_stats


# =========================
# RAG 检索模式兼容配置
# =========================
RETRIEVAL_ROUGH = "rough"
RETRIEVAL_LIGHT_RERANK = "light_rerank"
RETRIEVAL_MODEL_RERANK = "model_rerank"
DEFAULT_RETRIEVAL_MODE = RETRIEVAL_LIGHT_RERANK


def _accepts_var_kwargs(func) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _supports_param(func, param_name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    return param_name in signature.parameters or _accepts_var_kwargs(func)


def _call_with_supported_kwargs(func, **kwargs):
    if _accepts_var_kwargs(func):
        return func(**kwargs)

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if _supports_param(func, key)
    }
    return func(**supported_kwargs)


def _enable_rerank_from_mode(retrieval_mode: str) -> bool:
    return retrieval_mode in {RETRIEVAL_LIGHT_RERANK, RETRIEVAL_MODEL_RERANK}


def _run_graph_with_rerank(**kwargs):
    retrieval_mode = kwargs.get("retrieval_mode", DEFAULT_RETRIEVAL_MODE)
    kwargs["enable_rerank"] = _enable_rerank_from_mode(retrieval_mode)
    return _call_with_supported_kwargs(run_graph, **kwargs)


def _answer_followup_with_rerank(**kwargs):
    retrieval_mode = kwargs.get("retrieval_mode", DEFAULT_RETRIEVAL_MODE)
    kwargs["enable_rerank"] = _enable_rerank_from_mode(retrieval_mode)
    return _call_with_supported_kwargs(answer_followup_question, **kwargs)


def _search_knowledge_with_rerank(query: str, k: int, fetch_k: int, retrieval_mode: str):
    if _search_knowledge is None:
        return []

    kwargs = {
        "query": query,
        "k": int(k),
        "fetch_k": int(fetch_k),
        "retrieval_mode": retrieval_mode,
        "enable_rerank": _enable_rerank_from_mode(retrieval_mode),
    }
    return _call_with_supported_kwargs(_search_knowledge, **kwargs)


# =========================
# 页面基础配置
# =========================
st.set_page_config(
    page_title="EduPilot Agent",
    page_icon="🎓",
    layout="wide",
)


# =========================
# Session State 初始化
# =========================

# 当前学习会话 ID：用于 Workflow / ReAct Agent 的短期记忆隔离
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 保存最近一次 Workflow Mode 的完整运行结果
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# 保存 Quiz 批改反馈
if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = ""

# 保存 Follow-up QA 追问历史
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# 保存 ReAct Agent 对话历史，包括问题、回答、草稿、Reflection、工具调用轨迹和长期记忆写入结果
if "react_agent_history" not in st.session_state:
    st.session_state.react_agent_history = []

# 保存手动执行遗忘机制的结果，方便在侧边栏展示
if "forgetting_result" not in st.session_state:
    st.session_state.forgetting_result = None


# =========================
# 页面标题与说明
# =========================
st.title("🎓 EduPilot Agent")
st.caption("一个面向个性化学习计划、导师讲解、测验批改、Reflection、ReAct Tool Calling 与长期记忆的 AI 学习助手")

st.markdown(
    """
EduPilot Agent 当前版本支持：

1. 根据学习目标生成今日学习计划；
2. 从本地知识库检索相关学习资料；
3. 根据学习计划和参考资料生成导师讲解；
4. 自动生成本轮小测验；
5. 支持用户作答并生成智能批改反馈；
6. 支持学生基于本轮学习内容继续追问答疑；
7. 支持将 RAG / Planner / Tutor / Quiz / Grading / QA / Reviewer / Long-term Memory 封装成 tools；
8. 支持 Workflow Mode 和 ReAct Agent Mode 两种运行模式；
9. 使用 LangGraph 串联 Retriever、Planner、Tutor、Quiz、Reviewer、Workflow Reflection 多节点工作流；
10. 支持轻量级 Workflow Reflection：Tutor / Quiz / Reviewer 节点级自检 + 全局 Workflow 自检；
11. 支持 ReAct Agent 最终回答后的 Reflection 自检改写；
12. 支持基于 Chroma 的向量数据库长期记忆，并提供轻量级遗忘机制。
13. 支持 3 种 rerank 模式：粗召回、轻量重排序、模型重排序
"""
)


# =========================
# Sidebar：参数、模式、知识库、会话、长期记忆
# =========================
with st.sidebar:
    st.header("学习参数设置")

    # 学生基础水平，用于生成学习计划、讲解、小测和复盘
    level = st.selectbox(
        "你的当前基础",
        ["零基础", "Beginner", "有一点基础", "中等水平"],
        index=1,
    )

    # 今日学习时长，用于控制学习计划粒度
    hours = st.number_input(
        "今天可学习时间 / 小时",
        min_value=1,
        max_value=12,
        value=4,
        step=1,
    )

    st.markdown("---")
    st.header("运行模式")

    # 双模式入口：固定 Workflow 与动态 ReAct Agent
    mode = st.radio(
        "请选择 EduPilot 运行模式",
        ["Workflow Mode", "ReAct Agent Mode"],
        index=0,
    )

    # Reflection 总开关：Workflow 和 ReAct 共用
    enable_reflection = st.checkbox(
        "启用 Reflection 反思模块",
        value=True,
        help="开启后，Workflow Mode 会执行轻量级节点自检和全局自检；ReAct Agent Mode 会对最终回答进行自检改写。",
    )

    st.markdown("---")
    st.header("🔎 RAG 召回 / Rerank 设置")

    retrieval_mode = st.selectbox(
        "检索模式",
        options=[
            RETRIEVAL_ROUGH,
            RETRIEVAL_LIGHT_RERANK,
            RETRIEVAL_MODEL_RERANK,
        ],
        index=[
            RETRIEVAL_ROUGH,
            RETRIEVAL_LIGHT_RERANK,
            RETRIEVAL_MODEL_RERANK,
        ].index(DEFAULT_RETRIEVAL_MODE),
        help=(
            "rough：只做 Chroma 向量粗召回；"
            "light_rerank：先扩大召回候选，再用轻量规则重排；"
            "model_rerank：尝试使用 CrossEncoder 重排，失败时自动降级到 light_rerank。"
        ),
    )

    rag_top_k = st.number_input(
        "最终返回 top_k",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="最终传给 LLM 的知识库片段数量。",
    )

    rag_fetch_k = st.number_input(
        "粗召回候选 fetch_k",
        min_value=int(rag_top_k),
        max_value=30,
        value=max(8, int(rag_top_k) * 3),
        step=1,
        help="rerank 前先从 Chroma 召回的候选片段数量。rough 模式下主要使用 top_k。",
    )

    st.caption(f"当前检索配置：{retrieval_mode} | top_k={rag_top_k} | fetch_k={rag_fetch_k}")

    # 根据当前模式展示不同的执行流程
    if mode == "Workflow Mode":
        st.markdown("### 当前工作流")
        st.code(
            """
User Input
    ↓
Retriever: RAG Retrieval / Rerank + Long-term Memory
    ↓
Planner
    ↓
Tutor + Lightweight Reflection
    ↓
Quiz + Lightweight Reflection
    ↓
Reviewer + Lightweight Reflection
    ↓
Global Workflow Reflection
    ↓
Memory Reflection → Chroma Long-term Memory
    ↓
Final Answer
            """,
            language="text",
        )
    else:
        st.markdown("### 当前 Agent Loop")
        st.code(
            """
User Question
    ↓
LLM decides whether to use tools
    ↓
Tool Call: RAG / Planner / Tutor / Quiz / Grading / QA / Reviewer / Long-term Memory
    ↓
Tool Result
    ↓
Draft Answer
    ↓
Reflection
    ↓
Memory Reflection → Chroma Long-term Memory
    ↓
Final Answer
            """,
            language="text",
        )

    st.markdown("---")
    st.header("📚 知识库管理")

    # 上传本地学习资料，供 RAG 检索使用
    uploaded_files = st.file_uploader(
        "上传学习资料（支持 .md / .txt）",
        type=["md", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("保存上传资料"):
            saved_files = save_uploaded_files(uploaded_files)

            if saved_files:
                st.success(f"已保存 {len(saved_files)} 个文件：")
                for filename in saved_files:
                    st.write(f"- {filename}")
            else:
                st.warning("没有保存成功的文件，请检查文件类型。")

    # 重新构建 Chroma 知识库
    if st.button("重新构建知识库"):
        with st.spinner("正在重新构建 Chroma 知识库..."):
            vectorstore_result = rebuild_vectorstore()

        if vectorstore_result["success"]:
            st.success(
                f'{vectorstore_result["message"]}，切片数：{vectorstore_result["chunk_count"]}'
            )
        else:
            st.warning(vectorstore_result["message"])

    st.markdown("---")
    st.markdown("### 🧠 学习会话记忆")
    st.caption(f"当前会话 ID：{st.session_state.thread_id[:8]}...")

    # 开启新会话时，只清空短期会话状态；不会清空向量长期记忆
    if st.button("开启新学习会话"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.latest_result = None
        st.session_state.quiz_feedback = ""
        st.session_state.qa_history = []
        st.session_state.react_agent_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 🗂️ 长期记忆管理")

    memory_stats = get_memory_stats()
    col_a, col_b = st.columns(2)
    col_a.metric("有效记忆", memory_stats.get("active_count", memory_stats.get("count", 0)))
    col_b.metric("已归档", memory_stats.get("archived_count", 0))
    st.caption(f"总数：{memory_stats.get('count', 0)} 条")
    st.caption(f"路径：{memory_stats.get('path', '')}")

    if st.button("执行一次遗忘检查"):
        with st.spinner("正在根据访问时间和访问次数检查长期记忆..."):
            forgetting_result = apply_forgetting_policy(
                max_idle_days=30,
                max_access_count=1,
            )
        st.session_state.forgetting_result = forgetting_result
        st.success(
            f"遗忘检查完成：检查 {forgetting_result.get('checked', 0)} 条，归档 {forgetting_result.get('archived', 0)} 条。"
        )
        st.rerun()

    if st.session_state.forgetting_result:
        with st.expander("查看最近一次遗忘检查结果"):
            st.write(st.session_state.forgetting_result)

    confirm_clear_memory = st.checkbox("确认要清空向量长期记忆", value=False)
    if st.button("清空长期记忆", disabled=not confirm_clear_memory):
        clear_long_term_memory()
        st.session_state.forgetting_result = None
        st.success("长期记忆已清空。")
        st.rerun()


# 学习目标是 Workflow 和 ReAct Agent 共用的核心输入
goal = st.text_area(
    "请输入你的学习目标",
    value="Build an AI Agent project in 10 days.",
    height=120,
)


# =========================
# RAG Rerank 调试面板：只用于验证三种检索模式
# =========================
with st.expander("🔎 RAG 召回 / Rerank 调试面板", expanded=False):
    st.caption(
        "用于单独测试 Day 9 的三种检索模式：rough、light_rerank、model_rerank。"
        "这里不会写入短期记忆或长期记忆。"
    )

    debug_query = st.text_input(
        "调试检索 Query",
        value=goal,
        placeholder="例如：LangGraph 的 thread_id 有什么作用？",
    )

    if st.button("运行 RAG 检索调试"):
        if not debug_query.strip():
            st.warning("请先输入调试 Query。")
        else:
            with st.spinner("正在执行 RAG 检索与 rerank..."):
                debug_results = _search_knowledge_with_rerank(
                    query=debug_query,
                    k=int(rag_top_k),
                    fetch_k=int(rag_fetch_k),
                    retrieval_mode=retrieval_mode,
                )

            if not debug_results:
                st.warning("没有检索到结果。请确认已经上传资料并重新构建知识库。")
            else:
                st.success(f"检索完成，共返回 {len(debug_results)} 条结果。")

                summary_rows = []
                for i, item in enumerate(debug_results, start=1):
                    summary_rows.append(
                        {
                            "final_rank": i,
                            "source": item.get("source"),
                            "chunk_id": item.get("chunk_id"),
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

                st.dataframe(summary_rows, use_container_width=True)

                for i, item in enumerate(debug_results, start=1):
                    with st.expander(f"查看结果 {i}：{item.get('source')} / chunk {item.get('chunk_id')}"):
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


# =========================
# Workflow Mode：固定 LangGraph 学习闭环
# =========================
if mode == "Workflow Mode":
    run_button = st.button("生成学习方案", type="primary")

    if run_button:
        if not goal.strip():
            st.warning("请先输入学习目标。")
        else:
            with st.spinner("EduPilot Agent 正在生成学习计划、导师讲解、小测验、复盘验收、Reflection 自检和长期记忆... "):
                try:
                    # 调用固定 LangGraph 工作流
                    workflow_result = _run_graph_with_rerank(
                        goal=goal,
                        level=level,
                        hours=hours,
                        thread_id=st.session_state.thread_id,
                        enable_reflection=enable_reflection,
                        rag_top_k=int(rag_top_k),
                        rag_fetch_k=int(rag_fetch_k),
                        retrieval_mode=retrieval_mode,
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()

            # 保存 Workflow 结果，供页面展示和 ReAct Agent 复用
            st.session_state.latest_result = workflow_result
            st.session_state.quiz_feedback = ""
            st.success("生成完成！")

    result = st.session_state.latest_result

    if result:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            [
                "📅 学习计划",
                "🧑‍🏫 导师讲解",
                "💬 追问答疑",
                "📝 小测验",
                "✅ 复盘验收",
                "🧠 Reflection",
                "📚 检索资料",
                "🗂️ 长期记忆",
            ]
        )

        with tab1:
            st.subheader("📅 今日学习计划")
            st.markdown(result.get("learning_plan", ""))

        with tab2:
            st.subheader("🧑‍🏫 导师讲解")
            st.markdown(result.get("tutor_explanation", ""))

        with tab3:
            st.subheader("💬 追问答疑")
            st.caption("你可以针对今天的学习计划、导师讲解、RAG 资料和长期记忆继续追问。")

            # 展示 Follow-up QA 历史
            if not st.session_state.qa_history:
                st.info("目前还没有追问记录。你可以在下面输入问题。")
            else:
                for item in st.session_state.qa_history:
                    with st.chat_message("user"):
                        st.markdown(item["question"])

                    with st.chat_message("assistant"):
                        st.markdown(item["answer"])

            # 固定 QA 表单：基于当前 Workflow 结果继续追问
            with st.form("followup_qa_form", clear_on_submit=True):
                followup_question = st.text_area(
                    "请输入你的追问",
                    height=120,
                    placeholder="例如：老师，为什么 checkpointer 和 thread_id 要一起用？",
                )

                submit_followup = st.form_submit_button("提交追问")

            if submit_followup:
                if not followup_question.strip():
                    st.warning("请先输入你的追问问题。")
                else:
                    with st.spinner("EduPilot Agent 正在结合本轮内容、知识库和长期记忆回答..."):
                        answer = _answer_followup_with_rerank(
                            goal=goal,
                            level=level,
                            question=followup_question,
                            learning_plan=result.get("learning_plan", ""),
                            tutor_explanation=result.get("tutor_explanation", ""),
                            retrieved_context=result.get("retrieved_context", ""),
                            qa_history=st.session_state.qa_history,
                            rag_top_k=int(rag_top_k),
                            rag_fetch_k=int(rag_fetch_k),
                            retrieval_mode=retrieval_mode,
                        )

                    # 将本轮追问加入 QA 历史
                    st.session_state.qa_history.append(
                        {
                            "question": followup_question,
                            "answer": answer,
                        }
                    )

                    st.rerun()

        with tab4:
            st.subheader("📝 本轮小测验")
            st.markdown(result.get("quiz", ""))

            student_answer = st.text_area(
                "请在这里作答",
                height=220,
                placeholder="请按 1、2、3 的顺序作答，例如：\n1. ...\n2. ...\n3. ...",
                key="student_answer",
            )

            # 固定 Quiz 批改：批改当前 Workflow 生成的小测验
            if st.button("提交答案并批改"):
                if not student_answer.strip():
                    st.warning("请先填写你的答案。")
                else:
                    with st.spinner("EduPilot Agent 正在批改你的答案..."):
                        feedback = grade_quiz(
                            goal=goal,
                            level=level,
                            quiz=result.get("quiz", ""),
                            student_answer=student_answer,
                            tutor_explanation=result.get("tutor_explanation", ""),
                        )

                    st.session_state.quiz_feedback = feedback
                    st.success("批改完成！")

            if st.session_state.quiz_feedback:
                st.markdown("---")
                st.subheader("✅ Quiz 批改反馈")
                st.markdown(st.session_state.quiz_feedback)

        with tab5:
            st.subheader("✅ 复盘与验收")
            st.markdown(result.get("review", ""))

        with tab6:
            st.subheader("🧠 轻量级 Workflow Reflection")
            st.caption("这里展示 Tutor / Quiz / Reviewer 的节点级自检，以及最终 Global Workflow Reflection。")

            if not enable_reflection:
                st.info("当前关闭了 Reflection。你可以在左侧侧边栏开启。")

            st.markdown("### Tutor Node Reflection")
            tutor_reflection = result.get("tutor_reflection", "")
            st.markdown(tutor_reflection if tutor_reflection else "未启用或暂无结果。")

            st.markdown("### Quiz Node Reflection")
            quiz_reflection = result.get("quiz_reflection", "")
            st.markdown(quiz_reflection if quiz_reflection else "未启用或暂无结果。")

            st.markdown("### Reviewer Node Reflection")
            review_reflection = result.get("review_reflection", "")
            st.markdown(review_reflection if review_reflection else "未启用或暂无结果。")

            st.markdown("### Global Workflow Reflection")
            workflow_reflection = result.get("workflow_reflection", "")
            st.markdown(workflow_reflection if workflow_reflection else "未启用或暂无结果。")

            with st.expander("查看 Workflow 草稿总输出"):
                st.markdown(result.get("workflow_draft_answer", ""))

            with st.expander("查看最终输出 final_answer"):
                st.markdown(result.get("final_answer", ""))

        with tab7:
            st.subheader("📚 RAG 与长期记忆检索上下文")
            st.caption("retrieved_context 中同时包含本地知识库 RAG 检索结果和向量长期记忆召回结果。")
            st.info(f"当前 RAG 配置：retrieval_mode={retrieval_mode}，top_k={rag_top_k}，fetch_k={rag_fetch_k}")
            st.markdown(result.get("retrieved_context", ""))

        with tab8:
            st.subheader("🗂️ 向量长期记忆")
            st.caption("Workflow 完成后，系统会通过 Memory Reflection 判断本轮学习事件是否值得写入长期记忆。")

            memory_result = result.get("memory_result", {})
            if memory_result:
                if memory_result.get("saved"):
                    st.success("本轮 Workflow 已写入长期记忆。")
                elif memory_result.get("action") == "discard":
                    st.info("Memory Reflection 判断本轮内容不需要写入长期记忆。")
                else:
                    st.warning("本轮长期记忆未写入或写入时发生异常。")

                st.markdown("### 本轮记忆写入结果")
                st.write(memory_result)
            else:
                st.info("暂无本轮长期记忆写入结果。")

            st.markdown("### 本轮召回的长期记忆")
            st.markdown(result.get("long_term_memory", "暂无相关长期记忆。"))

            st.markdown("### 当前长期记忆库统计")
            st.write(get_memory_stats())

        # 调试入口：查看 LangGraph State 原始结果
        with st.expander("查看原始 State 数据"):
            st.write(result)


# =========================
# ReAct Agent Mode：动态 Tool Calling Agent
# =========================
else:
    st.subheader("🛠️ ReAct Agent Mode")
    st.caption("这个模式会让模型根据你的问题自主选择 RAG / Planner / Tutor / Quiz / Grading / QA / Reviewer / Long-term Memory 等工具，并在最终回答后进行 Reflection 自检改写。")

    # 读取最近一次 Workflow 结果；没有运行过 Workflow 时为空字典
    latest_result = st.session_state.latest_result or {}

    # ReAct Agent 可独立运行；如果有 Workflow 上下文，则回答更贴合当前学习内容
    if latest_result:
        st.success("已检测到 Workflow Mode 生成过的学习上下文，ReAct Agent 可以读取并调用相关工具。")
    else:
        st.info("当前还没有 Workflow Mode 生成的学习上下文。ReAct Agent 仍可独立调用 Planner / Tutor / RAG / Quiz / QA / Reviewer / Long-term Memory 等工具，建议先运行 Workflow，回答会更贴合本轮学习内容。")

    st.markdown("### 可用工具")
    st.markdown(
        """
- `rag_tool`：封装 RAG 检索；
- `plan_tool`：封装 Planner 学习计划；
- `tutor_tool`：封装 Tutor 导师讲解；
- `quiz_tool`：封装 Quiz 生成；
- `grade_quiz_answer_tool`：封装 Quiz 批改；
- `qa_tool`：封装 Follow-up QA；
- `review_tool`：封装 Reviewer 复盘验收；
- `get_current_context_tool`：读取当前会话上下文；
- `long_term_memory_tool`：检索 Chroma 向量长期记忆。
        """
    )

    st.info(f"当前 ReAct RAG 配置：retrieval_mode={retrieval_mode}，top_k={rag_top_k}，fetch_k={rag_fetch_k}")

    # 展示 ReAct Agent 历史对话、Reflection、工具调用轨迹和记忆写入结果
    if not st.session_state.react_agent_history:
        st.info("目前还没有 ReAct Agent 对话。可以先问：老师，请根据我的长期记忆总结我目前 EduPilot 项目的进度和下一步任务。")
    else:
        for item in st.session_state.react_agent_history:
            with st.chat_message("user"):
                st.markdown(item["question"])

            with st.chat_message("assistant"):
                st.markdown(item["final_answer"])

            # 展示本轮 Reflection，便于调试和项目演示
            if item.get("used_reflection") or item.get("reflection") or item.get("draft_answer"):
                with st.expander("查看本轮 Reflection 自检过程"):
                    st.markdown("### ReAct 草稿回答")
                    st.markdown(item.get("draft_answer", "暂无草稿记录。"))

                    st.markdown("### Reflection 审查意见")
                    st.markdown(item.get("reflection", "暂无 Reflection 记录。"))

            # 展示本轮工具调用 trace，便于调试和项目演示
            if item.get("trace"):
                with st.expander("查看本轮工具调用过程"):
                    for step in item["trace"]:
                        if step.get("type") == "tool_call":
                            st.markdown(f"**调用工具：`{step.get('name', '')}`**")
                            st.code(step.get("content", ""), language="text")

                        elif step.get("type") == "tool_result":
                            st.markdown(f"**工具返回：`{step.get('name', '')}`**")
                            st.code(step.get("content", ""), language="text")

            # 展示 ReAct 交互后的长期记忆写入结果
            if item.get("memory_result"):
                with st.expander("查看本轮长期记忆写入结果"):
                    st.write(item.get("memory_result"))

    # ReAct Agent 输入表单：用户自由提问，由模型决定是否调用工具
    with st.form("react_agent_form", clear_on_submit=True):
        react_question = st.text_area(
            "请输入你想让 ReAct Agent 处理的问题",
            height=140,
            placeholder="例如：老师，请根据我的长期记忆，总结我目前 EduPilot 项目的进度和下一步任务。",
        )

        submit_react_question = st.form_submit_button("提交给 ReAct Agent")

    if submit_react_question:
        if not react_question.strip():
            st.warning("请先输入问题。")
        else:
            # 构建 ReAct Agent 上下文：
            # 有 Workflow 结果时复用；没有时让 tools 自己补齐 Plan / Tutor / RAG 等内容
            current_context = {
                "goal": goal,
                "level": level,
                "hours": hours,
                "learning_plan": latest_result.get("learning_plan", ""),
                "tutor_explanation": latest_result.get("tutor_explanation", ""),
                "quiz": latest_result.get("quiz", ""),
                "review": latest_result.get("review", ""),
                "retrieved_context": latest_result.get("retrieved_context", ""),
                "workflow_reflection": latest_result.get("workflow_reflection", ""),
                "final_answer": latest_result.get("final_answer", ""),
                "long_term_memory": latest_result.get("long_term_memory", ""),
                "qa_history": st.session_state.qa_history,
                "react_agent_history": st.session_state.react_agent_history,
                "rag_top_k": int(rag_top_k),
                "rag_fetch_k": int(rag_fetch_k),
                "retrieval_mode": retrieval_mode,
            }

            with st.spinner("ReAct Agent 正在判断是否需要调用工具，并生成回答..."):
                try:
                    # 调用 ReAct Agent，返回最终回答、草稿、Reflection 和工具调用轨迹
                    react_result = run_react_agent(
                        question=react_question,
                        context=current_context,
                        thread_id=st.session_state.thread_id,
                        enable_reflection=enable_reflection,
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()

            # ReAct 完成后，尝试把本轮交互沉淀为长期记忆。
            # 失败不影响 ReAct 主回答。
            try:
                react_memory_result = record_react_memory(
                    goal=goal,
                    level=level,
                    question=react_question,
                    answer=react_result.get("final_answer", ""),
                )
            except Exception as exc:
                react_memory_result = {
                    "saved": False,
                    "action": "error",
                    "reason": str(exc),
                }

            # 保存本轮 ReAct 对话，供页面展示和后续上下文使用
            st.session_state.react_agent_history.append(
                {
                    "question": react_question,
                    "final_answer": react_result.get("final_answer", ""),
                    "draft_answer": react_result.get("draft_answer", ""),
                    "reflection": react_result.get("reflection", ""),
                    "used_reflection": react_result.get("used_reflection", False),
                    "trace": react_result.get("trace", []),
                    "memory_result": react_memory_result,
                }
            )

            st.rerun()