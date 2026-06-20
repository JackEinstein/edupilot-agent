from langchain_core.messages import SystemMessage

from src.llm import get_llm
from src.prompts import render_prompt


def generate_quiz(goal, level, learning_plan, tutor_explanation, retrieved_context):
    """
    根据学习计划和当前阶段，生成一套阶段测试题。
    """

    llm = get_llm()

    prompt = render_prompt(
        "quiz",
        learning_goal=f"""
        学习目标：{goal}
        当前水平：{level}
        """,
        learning_plan=learning_plan,
        tutor_answer=tutor_explanation,
        rag_context=retrieved_context,
    )

    messages = [SystemMessage(content=prompt)]

    response = llm.invoke(messages)
    return response.content


def grade_quiz(goal, level, tutor_explanation, quiz, student_answer):
    """
    根据 quiz 和学生作答情况，生成批改反馈。
    """

    llm = get_llm()

    prompt = render_prompt(
        "grade",
        quiz=quiz,
        student_answer=student_answer,
        tutor_answer=f"""
        学习目标：{goal}
        当前水平：{level}

        导师讲解：
        {tutor_explanation}
        """,
        rag_context="",
    )

    messages = [SystemMessage(content=prompt)]

    response = llm.invoke(messages)
    return response.content