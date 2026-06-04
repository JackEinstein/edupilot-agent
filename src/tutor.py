from langchain_core.messages import SystemMessage, HumanMessage
from src.llm import get_llm


def generate_tutor_explanation(goal, level, learning_plan):
    """
    根据学习目标和学习计划，生成教学讲解内容。
    """

    llm = get_llm()

    sys_message = SystemMessage(
        content=(
            "你是 EduPilot Agent 的一对一 AI 编程导师。"
            "你的任务不是重新制定计划，而是基于已有学习计划，"
            "解释今天任务中涉及的核心知识点，并帮助用户理解为什么要这样做。"
        )
    )

    hum_message = HumanMessage(
        content=f"""
        用户学习目标：
        {goal}
        
        用户当前水平：
        {level}
        
        今日学习计划：
        {learning_plan}
        
        请生成一份“导师讲解”，要求：
        1. 提炼今天最重要的 3-5 个核心知识点；
        2. 每个知识点用初学者能听懂的话解释；
        3. 说明这个知识点和 EduPilot Agent 项目的关系；
        4. 如果涉及 LangGraph、Agent、RAG、Streamlit，请尽量结合项目代码解释；
        5. 不要重新输出完整学习计划；
        6. 输出中文，语气像一位严格但耐心的工程导师。
        """
    )

    messages = [sys_message, hum_message]

    response = llm.invoke(messages)
    return response.content