def format_history(messages, max_messages=6):
    """
     将 LangGraph messages 历史消息格式化为 prompt 可读文本。
    """

    if not messages:
        return "暂无历史对话。"

    recent_messages = messages[-max_messages:]
    lines = []

    for msg in recent_messages:
        role = '学生' if msg.type == 'human' else 'EduPilot'
        lines.append(f'{role}: {msg.content}')

    return '\n'.join(lines)