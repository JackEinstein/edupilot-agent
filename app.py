import streamlit as st

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
    st.markdown(f"""
### Goal
{goal}

### Current Level
{level}

### Available Time
{hours} hours

### Suggested Plan
1. Review the core concept of AI Agent workflow.
2. Build the minimum Streamlit interface.
3. Implement a simple planning module.
4. Save today's progress and blockers.
5. Prepare for tomorrow's RAG module.
""")

st.subheader("Ask EduPilot")

question = st.text_input("Ask a question about your AI learning project:")

if question:
    st.write("EduPilot:")
    st.write(f"You asked: {question}")
    st.write("This is the minimum demo. Next we will connect it with LangGraph and RAG.")
