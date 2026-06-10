import uuid

import streamlit as st

from src.qa import answer_followup_question
from src.quiz import grade_quiz
from src.react_agent import run_react_agent
from src.retriever import save_uploaded_files, rebuild_vectorstore
from src.graph import run_graph


# 页面基础配置
st.set_page_config(
    page_title="EduPilot Agent",
    page_icon="🎓",
    layout="wide",
)


# 初始化当前学习会话 ID，用于 Workflow / ReAct Agent 的短期记忆隔离
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

# 保存 ReAct Agent 对话历史，包括问题、回答和工具调用轨迹
if "react_agent_history" not in st.session_state:
    st.session_state.react_agent_history = []


st.title("🎓 EduPilot Agent")
st.caption("一个面向个性化学习计划、导师讲解、测验批改与 ReAct Tool Calling 的 AI 学习助手")

st.markdown(
    """
    EduPilot Agent 当前版本支持：

    1. 根据学习目标生成今日学习计划；
    2. 从本地知识库检索相关学习资料；
    3. 根据学习计划和参考资料生成导师讲解；
    4. 自动生成本轮小测验；
    5. 支持用户作答并生成智能批改反馈；
    6. 支持学生基于本轮学习内容继续追问答疑；
    7. 支持将 RAG / Planner / Tutor / Quiz / Grading / QA / Reviewer 封装成 tools；
    8. 支持 Workflow Mode 和 ReAct Agent Mode 两种运行模式；
    9. 使用 LangGraph 串联 Retriever、Planner、Tutor、Quiz、Reviewer 多节点工作流。
"""
)


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

    # 根据当前模式展示不同的执行流程
    if mode == "Workflow Mode":
        st.markdown("### 当前工作流")
        st.code(
            """
User Input
    ↓
Retriever
    ↓
Planner
    ↓
Tutor
    ↓
Quiz
    ↓
Reviewer
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
Tool Call: RAG / Planner / Tutor / Quiz / Grading / QA / Reviewer
    ↓
Tool Result
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

    # 重新构建 Chroma 向量库
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

    # 开启新会话时，清空 Workflow、QA、Quiz、ReAct Agent 历史
    if st.button("开启新学习会话"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.latest_result = None
        st.session_state.quiz_feedback = ""
        st.session_state.qa_history = []
        st.session_state.react_agent_history = []
        st.rerun()


# 学习目标是 Workflow 和 ReAct Agent 共用的核心输入
goal = st.text_area(
    "请输入你的学习目标",
    value="Build an AI Agent project in 10 days.",
    height=120,
)


# =========================
# Workflow Mode：固定 LangGraph 学习闭环
# =========================
if mode == "Workflow Mode":
    run_button = st.button("生成学习方案", type="primary")

    if run_button:
        if not goal.strip():
            st.warning("请先输入学习目标。")
        else:
            with st.spinner("EduPilot Agent 正在生成学习计划、导师讲解、小测验和复盘验收..."):
                try:
                    # 调用固定 LangGraph 工作流
                    workflow_result = run_graph(
                        goal=goal,
                        level=level,
                        hours=hours,
                        thread_id=st.session_state.thread_id,
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["📅 学习计划", "🧑‍🏫 导师讲解", "💬 追问答疑", "📝 小测验", "✅ 复盘验收", "📚 检索资料"]
        )

        with tab1:
            st.subheader("📅 今日学习计划")
            st.markdown(result["learning_plan"])

        with tab2:
            st.subheader("🧑‍🏫 导师讲解")
            st.markdown(result["tutor_explanation"])

        with tab3:
            st.subheader("💬 追问答疑")
            st.caption("你可以针对今天的学习计划、导师讲解、RAG 资料继续追问。")

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
                    with st.spinner("EduPilot Agent 正在结合本轮内容和知识库回答..."):
                        answer = answer_followup_question(
                            goal=goal,
                            level=level,
                            question=followup_question,
                            learning_plan=result["learning_plan"],
                            tutor_explanation=result["tutor_explanation"],
                            retrieved_context=result["retrieved_context"],
                            qa_history=st.session_state.qa_history,
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
            st.markdown(result["quiz"])

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
                            quiz=result["quiz"],
                            student_answer=student_answer,
                            tutor_explanation=result["tutor_explanation"],
                        )

                    st.session_state.quiz_feedback = feedback
                    st.success("批改完成！")

            if st.session_state.quiz_feedback:
                st.markdown("---")
                st.subheader("✅ Quiz 批改反馈")
                st.markdown(st.session_state.quiz_feedback)

        with tab5:
            st.subheader("✅ 复盘与验收")
            st.markdown(result["review"])

        with tab6:
            st.subheader("📚 RAG 检索到的参考资料")
            st.markdown(result["retrieved_context"])

        # 调试入口：查看 LangGraph State 原始结果
        with st.expander("查看原始 State 数据"):
            st.write(result)


# =========================
# ReAct Agent Mode：动态 Tool Calling Agent
# =========================
else:
    st.subheader("🛠️ ReAct Agent Mode")
    st.caption("这个模式会让模型根据你的问题自主选择 RAG / Planner / Tutor / Quiz / Grading / QA / Reviewer 等工具。")

    # 读取最近一次 Workflow 结果；没有运行过 Workflow 时为空字典
    latest_result = st.session_state.latest_result or {}

    # ReAct Agent 可独立运行；如果有 Workflow 上下文，则回答更贴合当前学习内容
    if latest_result:
        st.success("已检测到 Workflow Mode 生成过的学习上下文，ReAct Agent 可以读取并调用相关工具。")
    else:
        st.info("当前还没有 Workflow Mode 生成的学习上下文。ReAct Agent 仍可独立调用 Planner / Tutor / RAG / Quiz / QA / Reviewer 等工具，建议先运行 Workflow，回答会更贴合本轮学习内容。")

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
- `get_current_context_tool`：读取当前会话上下文.
        """
    )

    # 展示 ReAct Agent 历史对话和工具调用轨迹
    if not st.session_state.react_agent_history:
        st.info("目前还没有 ReAct Agent 对话。可以先问：老师，我今天想学习 ReAct Tool Calling，请你直接给我一份学习计划和小测验。")
    else:
        for item in st.session_state.react_agent_history:
            with st.chat_message("user"):
                st.markdown(item["question"])

            with st.chat_message("assistant"):
                st.markdown(item["answer"])

            # 展示本轮工具调用 trace，便于调试和项目演示
            if item.get("trace"):
                with st.expander("查看本轮工具调用过程"):
                    for step in item["trace"]:
                        if step["type"] == "tool_call":
                            st.markdown(f"**调用工具：`{step['name']}`**")
                            st.code(step["content"], language="text")

                        elif step["type"] == "tool_result":
                            st.markdown(f"**工具返回：`{step['name']}`**")
                            st.code(step["content"], language="text")

    # ReAct Agent 输入表单：用户自由提问，由模型决定是否调用工具
    with st.form("react_agent_form", clear_on_submit=True):
        react_question = st.text_area(
            "请输入你想让 ReAct Agent 处理的问题",
            height=140,
            placeholder="例如：老师，请重新生成一套关于 Tool Calling 的小测验。",
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
                "qa_history": st.session_state.qa_history,
                "react_agent_history": st.session_state.react_agent_history,
            }

            with st.spinner("ReAct Agent 正在判断是否需要调用工具，并生成回答..."):
                try:
                    # 调用 ReAct Agent，返回最终回答和工具调用轨迹
                    react_result = run_react_agent(
                        question=react_question,
                        context=current_context,
                        thread_id=st.session_state.thread_id,
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()

            # 保存本轮 ReAct 对话，供页面展示和后续上下文使用
            st.session_state.react_agent_history.append(
                {
                    "question": react_question,
                    "answer": react_result["answer"],
                    "trace": react_result["trace"],
                }
            )

            st.rerun()