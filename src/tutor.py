from langchain_core.messages import SystemMessage, HumanMessage
from src.llm import get_llm


def generate_tutor_explanation(goal, level, learning_plan, retrieved_context, history):
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
        历史对话：
        {history}
        
        用户学习目标：
        {goal}
        
        用户当前水平：
        {level}
        
        今日学习计划：
        {learning_plan}
        
        下面是从知识库检索到的参考资料：
        {retrieved_context}
        
        请生成一份“导师讲解”，要求：
        1. 如果历史对话中出现学生的姓名、学习主题、薄弱点，需要主动承接；
        2. 如果用户当前输入是在追问历史对话，先回答历史问题，再继续导师讲解；
        3. 优先基于参考资料进行讲解；
        4. 提炼今天最重要的 3-5 个核心知识点；
        5. 每个知识点用初学者能听懂的话解释；
        6. 说明这个知识点和 EduPilot Agent 项目的关系；
        7. 如果涉及 LangGraph、Agent、RAG、Streamlit，请尽量结合项目代码解释；
        8. 不要重新输出完整学习计划；
        9. 如果参考资料不足，可以补充你的通用知识，但要避免编造；
        10. 输出中文，语气像一位严格但耐心的工程导师。
        """
    )

    messages = [sys_message, hum_message]

    response = llm.invoke(messages)
    return response.content