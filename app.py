import streamlit as st

from src.planner import generate_learning_plan

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
            plan = generate_learning_plan(goal, level, hours)
            st.markdown(plan)
        except Exception as e:
            st.error("Failed to generate plan.")
            st.exception(e)

st.subheader("Ask EduPilot")

question = st.text_input("Ask a question about your AI learning project:")

if question:
    st.write("EduPilot:")
    st.write(f"You asked: {question}")
    st.write("Next we will connect this Q&A module with LangGraph and RAG.")