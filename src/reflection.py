import re

from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import get_llm
from src.prompts import render_prompt


# =========================
# ReAct Agent Reflection
# =========================
def _format_trace(trace: list[dict]) -> str:
    """
    Format ReAct tool trace for reflection prompts.
    """

    if not trace:
        return "本轮没有工具调用"

    formatted_trace = []
    for i, item in enumerate(trace, start=1):
        step_type = item.get("type", 'unknown')
        name = item.get("name", 'unknown_tool')
        content = item.get("content", '')

        if len(content) >= 2000:
            content = content[:2000] + '\n...超过文本长度上限'

        formatted_trace.append(
            f"""
            'step': {i}
            '类型': {step_type}
            '工具': {name}
            '内容': {content}
            """
        )

    return "\n".join(formatted_trace)


def _split_reflection_output(text: str) -> tuple[str, str]:
    """
    Split a Reflection response into review notes and improved output.

    Some LLM responses may miss the requested markdown separator. In that case,
    keep the raw response as the usable output instead of failing the workflow.
    """

    text = text or ""

    if "## Improved Output" in text:
        reflection, improved = text.split("## Improved Output", maxsplit=1)
        reflection = reflection.replace("## Reflection", "").strip()
        return reflection, improved.strip()

    return (
        "模型未按指定格式输出，已使用原始输出作为改写结果。",
        text.strip(),
    )


def _should_improve(reflection: str) -> bool:
    """
    根据 Reflection 文本判断是否真的需要进入改写阶段。

    如果模型明确说“不需要改写”，或者评分较高，就直接保留 draft_answer，
    避免二次改写导致答案变短、信息丢失。
    """

    text = reflection or ""

    # 明确写了“不需要改写”
    if re.search(r"是否需要改写[\s\S]{0,100}不需要改写", text):
        return False

    # 如果能解析到评分，且评分 >= 85，也认为无需改写
    score_match = re.search(r"(\d{1,3})\s*分", text)
    if score_match:
        score = int(score_match.group(1))
        if score >= 85:
            return False

    return True


def reflect_answer(context: dict, question: str, draft_answer: str, trace: list[dict]) -> str:
    """
    Reflect on the ReAct Agent draft answer and identify improvement points.

    Reflection does not answer the student directly. It only checks whether the
    draft answer is grounded, complete, actionable, and suitable for the learner.
    """

    llm = get_llm()

    goal = context.get("goal") or ''
    hours = context.get("hours") or 4
    level = context.get("level") or ''
    learning_plan = context.get("learning_plan") or ''
    tutor_explanation = context.get("tutor_explanation") or ''
    retrieved_context = context.get("retrieved_context") or ''
    trace_text = _format_trace(trace)

    system_message = SystemMessage(
        content=render_prompt("react_reflection_system")
    )

    human_message = HumanMessage(
        content=render_prompt(
            "react_reflection_human",
            question=question,
            goal=goal,
            level=level,
            hours=hours,
            learning_plan=learning_plan,
            tutor_explanation=tutor_explanation,
            retrieved_context=retrieved_context,
            trace_text=trace_text,
            draft_answer=draft_answer,
        )
    )

    messages = [system_message, human_message]

    response = llm.invoke(messages)
    return response.content


def improve_answer(context: dict, question: str, draft_answer: str, reflection: str) -> str:
    """
    Improve the draft answer according to reflection feedback.
    """

    llm = get_llm()

    goal = context.get("goal") or ''
    level = context.get("level") or ''
    hours = context.get("hours") or 4

    system_message = SystemMessage(
        content=render_prompt("react_improve_system")
    )

    human_message = HumanMessage(
        content=render_prompt(
            "react_improve_human",
            question=question,
            goal=goal,
            level=level,
            hours=hours,
            draft_answer=draft_answer,
            reflection=reflection,
        )
    )

    messages = [system_message, human_message]

    response = llm.invoke(messages)
    return response.content


def run_reflection_loop(context: dict, question: str, draft_answer: str, trace: list[dict], enable_reflection: bool = True) -> dict:
    """
    Run one reflection-improvement pass after ReAct Agent generation.
    """

    if not enable_reflection:
        return {
            'draft_answer': draft_answer,
            'final_answer': draft_answer,
            'reflection': '',
            'used_reflection': False,
        }

    reflection = reflect_answer(
        context=context,
        question=question,
        draft_answer=draft_answer,
        trace=trace,
    )

    if not _should_improve(reflection):
        return {
            'draft_answer': draft_answer,
            'final_answer': draft_answer,
            'reflection': reflection,
            'used_reflection': True,
        }

    final_answer = improve_answer(
        context=context,
        question=question,
        draft_answer=draft_answer,
        reflection=reflection,
    )

    return {
        'draft_answer': draft_answer,
        'final_answer': final_answer,
        'reflection': reflection,
        'used_reflection': True,
    }


# =========================
# Workflow Reflection
# =========================

# 为不同节点生成对应的规则提示词
LIGHT_STAGE_RULES = {
    "tutor": """
你正在审查 Tutor Node 的导师讲解。
重点检查：
1. 是否承接学习计划，而不是重新制定计划；
2. 是否适合学生当前水平；
3. 是否讲清楚 3-5 个核心知识点；
4. 是否结合项目场景；
5. 是否能支撑后续 Quiz 出题。
""",
    "quiz": """
你正在审查 Quiz Node 的小测验。
重点检查：
1. 是否基于导师讲解出题；
2. 是否覆盖本轮核心知识点；
3. 难度是否适合学生当前水平；
4. 是否避免偏题、怪题、纯记忆题；
5. 题目是否清晰，学生知道怎么作答。
""",
    "reviewer": """
你正在审查 Reviewer Node 的复盘验收。
重点检查：
1. 是否总结本轮学习重点；
2. 是否给出可执行的自测问题、实践任务和验收标准；
3. 是否和学习计划、导师讲解、小测验保持一致；
4. 是否给出明天的补救建议；
5. 是否避免空泛鼓励。
""",
}


def _format_context(context: dict) -> str:
    """
    把上下文格式化成 prompt
    """

    if not context:
        return '暂无上下文'

    lines = []
    for key, value in context.items():

        if len(value) >= 2000:
            value = value[:2000]

        lines.append(f'【{key}】\n{str(value)}')

    return '\n\n'.join(lines)


def reflect_node_output(stage: str, context: dict, draft_output: str, enable_reflection: bool = True) -> dict:
    """
    轻量级节点 Reflection
    """

    if not enable_reflection:
        return {
            'improved_output': draft_output,
            'reflection': '',
            'used_reflection': False,
        }

    llm = get_llm()

    stage_rule = LIGHT_STAGE_RULES.get(stage, '请检查该节点输出是否清晰、准确、具体、可执行。')

    system_message = SystemMessage(
        content=render_prompt("workflow_node_reflection_system")
    )

    human_message = HumanMessage(
        content=render_prompt(
            "workflow_node_reflection_human",
            stage=stage,
            stage_rule=stage_rule,
            context_text=_format_context(context),
            draft_output=draft_output,
        )
    )

    messages = [system_message, human_message]

    response = llm.invoke(messages)

    reflection, improved_output = _split_reflection_output(response.content)

    return {
        'improved_output': improved_output,
        'reflection': reflection,
        'used_reflection': True,
    }


def reflect_workflow_output(context: dict, draft_answer: str, enable_reflection: bool = True) -> dict:
    """
    轻量级全局 Workflow Reflection。

    作用：
    检查 learning_plan、tutor_explanation、quiz、review 是否前后一致，
    然后生成最终展示给用户的学习方案。
    """

    if not enable_reflection:
        return {
            "final_answer": draft_answer,
            "reflection": "",
            "used_reflection": False,
        }

    llm = get_llm()

    system_message = SystemMessage(
        content=render_prompt("workflow_global_reflection_system")
    )

    human_message = HumanMessage(
        content=render_prompt(
            "workflow_global_reflection_human",
            context_text=_format_context(context),
            draft_answer=draft_answer,
        )
    )

    messages = [system_message, human_message]

    response = llm.invoke(messages)

    reflection, final_answer = _split_reflection_output(response.content)

    return {
        'final_answer': final_answer,
        'reflection': reflection,
        'used_reflection': True,
    }
