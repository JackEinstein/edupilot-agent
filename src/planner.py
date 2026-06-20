from langchain_core.messages import SystemMessage

from src.llm import get_llm
from src.prompts import render_prompt


def generate_learning_plan(goal, level, hours, history):
    """
    Generate a personalized daily learning plan.
    """

    llm = get_llm()

    prompt = render_prompt(
        "planner",
        learning_goal=f"""
        学习目标：{goal}
        当前水平：{level}
        可用时间：{hours} 小时
        """,
        short_term_history=history,
        long_term_memory="",
        rag_context="",
    )

    messages = [SystemMessage(content=prompt)]

    response = llm.invoke(messages)
    return response.content