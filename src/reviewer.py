from langchain_core.messages import SystemMessage, HumanMessage
from src.llm import get_llm


def generate_review(goal, level, hours, learning_plan, tutor_explanation, history):
    """
    根据学习计划和导师讲解，生成复盘、自测和实践任务。
    """

    llm = get_llm()

    sys_message = SystemMessage(
        content=(
            "你是 EduPilot Agent 的学习复盘教练。"
            "你的任务是帮助用户检查今天是否真正掌握内容，"
            "并给出可执行的自测问题、实践任务和验收标准。"
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
        
        用户可用学习时间：
        {hours}
        
        今日学习计划：
        {learning_plan}
        
        导师讲解：
        {tutor_explanation}
        
        请生成一份“学习复盘与验收清单”，要求：
        1. 自测问题和验收标准要结合历史对话中的学生薄弱点；
        2. 如果用户当前输入是在追问历史对话，先根据历史对话直接回答；
        3. 给出 5 个自测问题；
        4. 给出 3 个代码实践任务；
        5. 给出今日项目验收标准；
        6. 给出如果没完成，明天应该如何补救；
        7. 输出中文，要求具体、可执行，不要空泛鼓励。
        """
    )

    messages = [sys_message, hum_message]

    response = llm.invoke(messages)
    return response.content