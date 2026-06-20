from langchain_core.messages import SystemMessage

from src.llm import get_llm
from src.prompts import render_prompt


def generate_review(goal, level, hours, learning_plan, tutor_explanation, history):
    """
    根据学习计划和导师讲解，生成复盘、自测和实践任务。
    """

    llm = get_llm()

    prompt = render_prompt(
        "reviewer",
        learning_goal=f"""
        学习目标：{goal}
        当前水平：{level}
        可用时间：{hours} 小时
        """,
        learning_plan=learning_plan,
        tutor_answer=tutor_explanation,
        quiz="",
        grade_feedback="",
        short_term_history=history,
        long_term_memory="",
    )

    messages = [SystemMessage(content=prompt)]

    response = llm.invoke(messages)
    return response.content