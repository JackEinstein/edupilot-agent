from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import get_llm


def generate_learning_plan(goal, level, hours):
    """
    Generate a personalized daily learning plan.
    """

    llm = get_llm()

    sys_message = SystemMessage(
        content=(
        "你是 EduPilot Agent 的学习规划专家。"
        "你的任务是根据用户的学习目标、基础水平和时间要求，"
        "生成结构清晰、可执行的学习计划。"
        )
    )

    hum_message = HumanMessage(
        content=f"""
        用户学习目标：
        {goal}
    
        用户当前水平：
        {level}
        
        用户学习时间：
        {hours}
    
        要求：
        1. 按时间块安排，例如 0-30min、30-90min。
        2. 每个时间块必须有明确任务。
        3. 每个任务必须有具体产出。
        4. 不要写空泛建议。
        5. 项目目标是 6 月 10 日前完成一个可展示的 EduPilot Agent Demo。
        6. 计划要兼顾学习 LangGraph / RAG / Agent 和实际写代码。
        7. 最后给出“今日验收标准”。
        
        请用中文回答，结构清晰，语气像一位严格的一对一工程导师。
        """
    )

    messages = [sys_message, hum_message]

    response = llm.invoke(messages)
    return response.content