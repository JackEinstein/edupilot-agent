import os

import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

# os.environ['DEEPSEEK_API_KEY'] = os.getenv('DEEPSEEK_API_KEY')
# os.environ['DEEPSEEK_API_BASE'] = os.getenv('DEEPSEEK_BASE_URL')

def get_llm():
    """Get a chat model from deepseek api key."""
    return ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        model='deepseek-chat',
        temperature=0.4
    )


def generate_learning_plan(goal, level, hours):
    """Generate a personalized daily learning plan."""
    llm = get_llm()

    prompt = f"""
    你是一个严格、专业、负责的 AI Agent 项目导师。

    请根据用户信息，为今天生成一份可执行的学习与开发计划。
    
    用户目标：{goal}
    当前水平：{level}
    今日可用时间：{hours} 小时
    
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

    response = llm.invoke(prompt)
    return response.content