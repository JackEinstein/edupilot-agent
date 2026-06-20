from langchain_core.messages import SystemMessage

from src.llm import get_llm
from src.prompts import render_prompt


def generate_tutor_explanation(goal, level, learning_plan, retrieved_context, history):
    """
    根据学习目标和学习计划，生成教学讲解内容。
    """

    llm = get_llm()

    prompt = render_prompt(
        "tutor",
        learning_goal=f"""
        学习目标：{goal}
        当前水平：{level}
        """,
        learning_plan=learning_plan,
        rag_context=retrieved_context,
        short_term_history=history,
        long_term_memory="",
    )

    messages = [SystemMessage(content=prompt)]

    response = llm.invoke(messages)
    return response.content