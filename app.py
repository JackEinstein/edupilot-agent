import streamlit as st

from src.graph import run_graph


st.set_page_config(
    page_title="EduPilot Agent",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 EduPilot Agent")
st.caption("A personalized AI learning and project coaching agent.")

with st.sidebar:
    st.header("User Profile")
    goal = st.text_input("Learning Goal", "Build an AI Agent project before June 10")
    level = st.selectbox("Current Level", ["Beginner", "Intermediate", "Advanced"])
    hours = st.slider("Available Study Hours Today", 1, 10, 4)

st.subheader("Today's Learning Plan")

if st.button("Generate Plan"):
    with st.spinner("EduPilot is generating your plan..."):
        try:
            result = run_graph(
                goal=goal,
                level=level,
                hours=hours,
            )

            st.success("生成完成！")

            tab1, tab2, tab3 = st.tabs(["📅 学习计划", "🧑‍🏫 导师讲解", "✅ 复盘验收"])

            with tab1:
                st.subheader("📅 今日学习计划")
                st.markdown(result["learning_plan"])

            with tab2:
                st.subheader("🧑‍🏫 导师讲解")
                st.markdown(result["tutor_explanation"])

            with tab3:
                st.subheader("✅ 复盘与验收")
                st.markdown(result["review"])

        except Exception as e:
            st.error("Failed to generate plan.")
            st.exception(e)

st.subheader("Ask EduPilot")

question = st.text_input("Ask a question about your AI learning project:")

if question:
    st.write("EduPilot:")
    st.write(f"You asked: {question}")
    st.write("Next we will connect this Q&A module with LangGraph and RAG.")