from langchain_core.messages import AnyMessage


def format_langgraph_messages(message_history: list[AnyMessage], max_rounds=20) -> str:
    """
     将 LangGraph messages 历史消息格式化为 prompt 可读文本。
    """

    if not message_history:
        return "暂无历史对话。"

    recent_messages = message_history[-max_rounds:]
    lines = []

    for msg in recent_messages:
        role = '学生' if msg.type == 'human' else 'EduPilot'
        lines.append(f'{role}: {msg.content}')

    return '\n'.join(lines)


def format_qa_history(qa_history: list[dict], max_rounds=10) -> str:
    """
    将 qa 追问答疑历史转成 prompt 可读文本
    """

    if not qa_history:
        return '暂无历史问答。'

    recent_qa = qa_history[-max_rounds:]
    lines = []

    for i, item in enumerate(recent_qa, start=1):
        question = item['question']
        answer = item['answer']

        lines.append(f'学生第{i}轮追问：')
        lines.append(f"学生问题：{question}")
        lines.append(f"EduPilot 回答：{answer}")

    return '\n\n'.join(lines)


def format_react_agent_history(react_agent_history: list[dict], max_rounds=10) -> str:
    """
    将 ReAct Agent 对话历史转为 prompt 可读文本
    """

    if not react_agent_history:
        return '暂无 ReAct Agent 对话历史'

    recent_history = react_agent_history[-max_rounds:]
    lines = []

    for i, item in enumerate(recent_history, start=1):
        question = item['question']
        answer = item['final_answer']

        lines.append(f"第{i}轮 ReAct Agent 对话：")
        lines.append(f"学生提问：{question}")
        lines.append(f"Agent 回答：{answer}")

    return '\n\n'.join(lines)