import streamlit as st

from src.retriever import save_uploaded_files, rebuild_vectorstore
from src.graph import run_graph


st.set_page_config(
    page_title="EduPilot Agent",
    page_icon="🎓",
    layout="wide",
)


st.title("🎓 EduPilot Agent")
st.caption("一个面向个性化学习计划、导师讲解与复盘验收的 AI 学习助手")

st.markdown(
    """
EduPilot Agent 当前版本支持：

1. 根据学习目标生成今日学习计划；
2. 从本地知识库检索相关学习资料；
3. 根据学习计划和参考资料生成导师讲解；
4. 根据学习内容生成复盘问题与验收标准；
5. 使用 LangGraph 串联 Retriever、Planner、Tutor、Reviewer 多节点工作流。
"""
)


with st.sidebar:
    st.header("学习参数设置")

    level = st.selectbox(
        "你的当前基础",
        ["零基础", "Beginner", "有一点基础", "中等水平"],
        index=1,
    )

    hours = st.number_input(
        "今天可学习时间 / 小时",
        min_value=1,
        max_value=12,
        value=4,
        step=1,
    )

    st.markdown("---")
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
        Reviewer
        """,
        language="text",
    )

    st.header("📚 知识库管理")

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

    if st.button("重新构建知识库"):
        with st.spinner("正在重新构建 Chroma 知识库..."):
            result = rebuild_vectorstore()

        if result["success"]:
            st.success(
                f'{result["message"]}，切片数：{result["chunk_count"]}'
            )
        else:
            st.warning(result["message"])



goal = st.text_area(
    "请输入你的学习目标",
    value="Build an AI Agent project before June 10",
    height=120,
)

run_button = st.button("生成学习方案", type="primary")


if run_button:
    if not goal.strip():
        st.warning("请先输入学习目标。")
    else:
        with st.spinner("EduPilot Agent 正在生成学习计划、导师讲解和复盘验收..."):
            try:
                result = run_graph(
                    goal=goal,
                    level=level,
                    hours=hours,
                )
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        st.success("生成完成！")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📅 学习计划", "🧑‍🏫 导师讲解", "✅ 复盘验收", "📚 检索资料"]
        )

        with tab1:
            st.subheader("📅 今日学习计划")
            st.markdown(result["learning_plan"])

        with tab2:
            st.subheader("🧑‍🏫 导师讲解")
            st.markdown(result["tutor_explanation"])

        with tab3:
            st.subheader("✅ 复盘与验收")
            st.markdown(result["review"])

        with tab4:
            st.subheader("📚 RAG 检索到的参考资料")
            st.markdown(result["retrieved_context"])

        with st.expander("查看原始 State 数据"):
            st.json(result)
