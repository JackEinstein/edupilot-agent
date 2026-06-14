from src.memory_reflection import reflect_memory_candidate
from src.vector_memory import add_vector_memory, format_vector_memory, clear_vector_memory, apply_forgetting_policy


def add_reflective_memory(event_type: str, raw_event: str, source_event: str) -> dict:
    """
    先让 LLM 判断是否值得保存，再写入 Chroma 向量记忆库。
    """

    decision = reflect_memory_candidate(event_type=event_type, raw_event=raw_event)

    if not decision.get("should_save"):
        return {
            "saved": False,
            "action": "discard",
            "reason": decision.get("reason", "LLM 判断不需要保存"),
        }

    save_result = add_vector_memory(
        context=decision.get("summary", ""),
        memory_type=decision.get("memory_type", "general"),
        source_event=source_event,
    )

    return {
        **save_result,
        "action": "save" if save_result.get("saved") else "discard",
        "reason": decision.get("reason", ""),
    }


def retrieve_long_term_memory(query: str, k: int = 4) -> str:
    """
    根据当前 query 语义召回长期记忆，并格式化为 prompt 上下文。
    检索前先执行一次遗忘策略。
    """

    try:
        apply_forgetting_policy(
            max_idle_days=30,
            max_access_count=1,
        )
    except Exception:
        pass

    return format_vector_memory(query=query, k=k)


def clear_long_term_memory() -> None:
    clear_vector_memory()


def record_workflow_memory(goal: str, level: str, workflow_result: dict) -> dict:
    raw_event = f"""
【学习目标】
{goal}

【学生水平】
{level}

【学习计划】
{workflow_result.get('learning_plan', '')}

【导师讲解】
{workflow_result.get('tutor_explanation', '')}

【小测验】
{workflow_result.get('quiz', '')}

【复盘验收】
{workflow_result.get('review', '')}

【Workflow Reflection】
{workflow_result.get('workflow_reflection', '')}
"""
    return add_reflective_memory(
        event_type="workflow学习闭环",
        raw_event=raw_event,
        source_event="workflow",
    )


def record_qa_memory(goal: str, level: str, question: str, answer: str) -> dict:
    raw_event = f"""
【学习目标】
{goal}

【学生水平】
{level}

【学生追问】
{question}

【导师回答】
{answer}
"""
    return add_reflective_memory(
        event_type="follow-up QA追问答疑",
        raw_event=raw_event,
        source_event="qa",
    )


def record_react_memory(goal: str, level: str, question: str, answer: str) -> dict:
    raw_event = f"""
【学习目标】
{goal}

【学生水平】
{level}

【ReAct 问题】
{question}

【ReAct 最终回答】
{answer}
"""
    return add_reflective_memory(
        event_type="react agent交互",
        raw_event=raw_event,
        source_event="react_agent",
    )
