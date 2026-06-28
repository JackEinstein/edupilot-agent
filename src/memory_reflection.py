import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import get_llm
from src.prompts import render_prompt
from src.vector_memory import _shorten, _sanitize


def _extract_json(text: str) -> dict:
    """
    从 LLM 输出中提取 JSON。

    兼容以下情况：
    1. 模型直接返回 JSON；
    2. 模型返回 ```json ... ```；
    3. 模型前后夹杂少量说明文字。
    """

    text = (text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # 兜底：尝试提取第一个 {...} JSON 片段
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"无法从 LLM 输出中解析 JSON：{text[:300]}")



def reflect_memory_candidate(event_type: str, raw_event: str) -> dict:
    """
    判断一次学习事件是否值得保存为长期记忆。
    """

    llm = get_llm()

    clean_event = _shorten(_sanitize(raw_event))
    if not clean_event:
        return {
            "should_save": False,
            "memory_type": "noise",
            "summary": "",
            "reason": "事件内容为空",
        }

    system_message = SystemMessage(
        content=render_prompt("memory_reflection_system")
    )

    human_message = HumanMessage(
        content=render_prompt(
            "memory_reflection_human",
            event_type=event_type,
            clean_event=clean_event,
        )
    )

    try:
        messages = [system_message, human_message]
        response = llm.invoke(messages)

    except Exception as exc:
        # LLM 失败时采用保守兜底：保存一段截断摘要，避免功能完全不可用。
        return {
            "should_save": True,
            "memory_type": "general",
            "summary": _shorten(clean_event, max_memory_char=800),
            "reason": f"LLM 反思失败，使用规则兜底：{exc}",
        }

    data = _extract_json(response.content)

    should_save = bool(data.get("should_save"))
    summary = str(data.get("summary") or "").strip()
    memory_type = str(data.get("memory_type") or "general").strip()
    reason = str(data.get("reason") or "").strip()

    if should_save and not summary:
        summary = _shorten(clean_event, max_memory_char=800)

    return {
        "should_save": should_save,
        "memory_type": memory_type,
        "summary": _sanitize(summary),
        "reason": reason,
    }
