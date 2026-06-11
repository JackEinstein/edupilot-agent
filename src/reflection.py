from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import get_llm


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
        content=(
            "你是 EduPilot Agent 的 Reflection Reviewer。"
            "你的任务不是重新回答学生问题，而是严格审查 Agent 的草稿回答，"
            "找出是否存在遗漏、空泛、没有结合工具结果、没有结合学生水平、"
            "或者可能编造的问题。"
        )
    )

    human_message = HumanMessage(
        content=f"""
        【学生问题】
        {question}
        
        【学习目标】
        {goal}
        
        【学生当前水平】
        {level}
        
        【可用学习时间】
        {hours}
        
        【已有学习计划】
        {learning_plan}
        
        【已有导师讲解】
        {tutor_explanation}
        
        【RAG 检索资料】
        {retrieved_context}
        
        【本轮 React Agent 调用工具轨迹】
        {trace_text}
        
        【Agent 草稿回答】
        {draft_answer}
        
        请对草稿回答做 Reflection 审查，必须按下面格式输出：

        ## Reflection 评分
        给出 0-100 分，并说明扣分原因。

        ## 主要问题
        用 3-5 条指出草稿回答的具体问题。如果草稿回答已经较好，也要指出可继续优化的地方。

        ## 改写建议
        给出下一版回答应该如何改进，要求具体、可执行。

        ## 是否需要改写
        只能回答：需要改写 / 不需要改写。
        """
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
        content=(
            "你是 EduPilot Agent 的最终回答改写器。"
            "你需要根据 Reflection 审查意见改写答案。"
            "最终输出只给学生看的正式回答，不要输出审查过程，"
            "不要暴露模型的私密推理过程。"
        )
    )

    human_message = HumanMessage(
        content=f"""
        【学生问题】
        {question}

        【学生学习目标】
        {goal}

        【学生当前水平】
        {level}

        【今日可用时间】
        {hours}

        【Agent 草稿回答】
        {draft_answer}

        【Reflection 审查意见】
        {reflection}

        请输出改写后的最终答案。要求：
        1. 用中文回答；
        2. 像老师带学生做项目一样具体、清楚、可执行；
        3. 保留草稿中正确、有用的内容；
        4. 修正 Reflection 指出的问题；
        5. 不要说“根据 Reflection”；
        6. 不要输出草稿、评分或审查过程。
        """
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
        content=(
            "你是 EduPilot Workflow 的轻量级节点审查器。"
            "你的任务是检查某个节点的草稿输出，并在不改变节点职责的前提下改写它。"
            "你不是在生成完整学习方案，只负责当前节点的局部质量控制。"
        )
    )

    human_message = HumanMessage(
        content=f"""
        当前节点：
        {stage}

        审查标准：
        {stage_rule}

        最小必要上下文：
        {_format_context(context)}

        当前节点草稿：
        {draft_output}

        请严格按照下面格式输出，不要增加其他一级标题：
        
        ## Reflection
        - 主要问题：用 1-3 条指出草稿可改进处；如果已经较好，也指出可以更贴合项目或更精炼的地方。
        - 改写原则：说明你将如何小幅改写。
        
        ## Improved Output
        这里输出改写后的正式节点内容。
        
        要求：
        1. 只输出当前节点应该交给后续节点使用的内容；
        2. 不要在 Improved Output 里提到 Reflection、评分、自检过程；
        3. 尽量保留草稿中的正确内容；
        4. 只做必要修正，不要大幅扩写；
        5. 不要引入上下文之外的新事实；
        6. 输出中文 Markdown。
        """
    )

    messages = [system_message, human_message]

    response = llm.invoke(messages)

    reflection, improved_output = response.content.split('## Improved Output', maxsplit=1)
    reflection = reflection.replace('## Reflection', '').strip()
    improved_output = improved_output.strip()

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
        content=(
            "你是 EduPilot Workflow 的全局审查器。"
            "你的任务是检查完整学习闭环是否一致、具体、适合学生水平，"
            "然后生成最终展示给学生的版本。"
        )
    )

    human_message = HumanMessage(
        content=f"""
        最小必要上下文：
        {_format_context(context)}
    
        Workflow 草稿总输出：
        {draft_answer}
    
        请严格按照下面格式输出，不要增加其他一级标题：
    
        ## Reflection
        - 一致性问题：检查计划、讲解、小测、复盘是否前后一致；
        - 可执行性问题：检查是否有明确任务、产出和验收标准；
        - 改写原则：说明你将如何小幅改写最终版本。
    
        ## Improved Output
        这里输出最终给学生看的完整学习方案。
    
        要求：
        1. 保留原来的学习计划、导师讲解、小测验、复盘四个部分；
        2. 修正明显重复、脱节、空泛的地方；
        3. 不要大幅扩写；
        4. 不要编造新资料；
        5. 输出中文 Markdown。
        """
    )

    messages = [system_message, human_message]

    response = llm.invoke(messages)

    reflection, final_answer = response.content.split('## Improved Output', maxsplit=1)
    reflection = reflection.replace('## Reflection', '').strip()
    final_answer = final_answer.strip()

    return {
        'final_answer': final_answer,
        'reflection': reflection,
        'used_reflection': True,
    }
