from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import get_llm
from src.vector_memory import _shorten, _sanitize


def reflect_memory_candidate(event_type: str, raw_event: str) -> dict:
    """
    判断一次学习事件是否值得保存为长期记忆。
    """

    clean_event = _shorten(_sanitize(raw_event))
    if not clean_event:
        return {
            "should_save": False,
            "memory_type": "noise",
            "summary": "",
            "reason": "事件内容为空",
        }

    system_message = SystemMessage(
        content="""
你是 EduPilot Agent 的长期记忆筛选器。
你的任务是判断一段学习事件是否值得保存为长期记忆。

长期记忆应该保存：
1. 学生长期学习目标、项目进度、下一步计划；
2. 学生稳定的薄弱点、掌握情况、学习偏好；
3. 对后续个性化辅导有帮助的信息。

不要保存：
1. 临时寒暄、无意义重复内容；
2. 过长的原始对话全文；
3. API Key、token、password 等敏感信息；
4. 只对当前一步有用、以后没有参考价值的细节。

你必须只输出 JSON，不要输出 Markdown，不要输出解释文字。
JSON 格式：
{
  "should_save": true 或 false,
  "memory_type": "project_progress / weakness / mastery / preference / next_step / general / noise",
  "summary": "如果值得保存，用 1-3 句话总结成长期记忆；如果不值得保存，留空",
  "reason": "简短说明原因"
}
"""
    )

    human_message = HumanMessage(
        content=f"""
【事件类型】
{event_type}

【学习事件】
{clean_event}

请判断是否写入长期记忆。
"""
    )

    try:
        response = get_llm().invoke([system_message, human_message])

    except Exception as exc:
        # LLM 失败时采用保守兜底：保存一段截断摘要，避免功能完全不可用。
        return {
            "should_save": True,
            "memory_type": "general",
            "summary": _shorten(clean_event, max_memory_char=800),
            "reason": f"LLM 反思失败，使用规则兜底：{exc}",
        }

    should_save = bool(response.get("should_save"))
    summary = str(response.get("summary") or "").strip()
    memory_type = str(response.get("memory_type") or "general").strip()
    reason = str(response.get("reason") or "").strip()

    if should_save and not summary:
        summary = _shorten(clean_event, max_memory_char=800)

    return {
        "should_save": should_save,
        "memory_type": memory_type,
        "summary": _sanitize(summary),
        "reason": reason,
    }
