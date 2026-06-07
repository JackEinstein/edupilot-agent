from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import get_llm


def generate_learning_plan(goal, level, hours, history):
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
        历史对话：
        {history}
        
        用户学习目标：
        {goal}
    
        用户当前水平：
        {level}
        
        用户学习时间：
        {hours}
    
        要求：
        1. 如果用户当前输入是在追问历史对话，例如“我刚才说了什么”“我叫什么”“我正在学习什么”，必须优先根据【历史对话】直接回答，不要说用户没有告诉你。
        2. 如果历史对话中确实没有相关信息，再说明无法从历史中判断。
        3. 按时间块安排，例如 0-30min、30-90min。
        4. 每个时间块必须有明确任务。
        5. 每个任务必须有具体产出。
        6. 不要写空泛建议。
        7. 项目目标是 6 月 10 日前完成一个可展示的 EduPilot Agent Demo。
        8. 计划要兼顾学习 LangGraph / RAG / Agent 和实际写代码。
        9. 最后给出“今日验收标准”。
        
        请用中文回答，结构清晰，语气像一位严格的一对一工程导师。
        """
    )

    messages = [sys_message, hum_message]

    response = llm.invoke(messages)
    return response.content