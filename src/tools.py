from langchain.tools import tool

from src.history import format_qa_history, format_react_agent_history
from src.planner import generate_learning_plan
from src.qa import answer_followup_question
from src.quiz import generate_quiz, grade_quiz
from src.retriever import format_retrieved_chunks
from src.reviewer import generate_review
from src.tutor import generate_tutor_explanation


def build_tools(context: dict):
    """
    构建 ReAct Agent 可以调用的工具箱
    """

    goal = context.get("goal") or "Build an AI Agent project in 10 days"
    level = context.get("level") or 'beginner'
    hours = context.get("hours") or 4
    retrieved_context = context.get("retrieved_context") or ''
    learning_plan = context.get("learning_plan") or ''
    tutor_explanation = context.get("tutor_explanation") or ''
    quiz = context.get("quiz") or ''
    review = context.get("review") or ''
    qa_history = context.get("qa_history") or []
    react_history = context.get("react_agent_history") or []

    qa_history_text = format_qa_history(qa_history)
    react_history_text = format_react_agent_history(react_history)
    history_text = f"""
        【qa问答历史】
        {qa_history_text}
        
        【ReAct Agent 问答历史】
        {react_history_text}
        """


    def _get_retrieved_context(query='', k=3) -> str:
        if retrieved_context and not query:
            return retrieved_context

        return format_retrieved_chunks(query=query, k=k)

    def _format_current_context():
        return f"""
            【学习目标】
            {goal}
            
            【当前基础】
            {level}

            【今日可用时间】
            {hours} 小时

            【学习计划】
            {learning_plan}

            【导师讲解】
            {tutor_explanation}

            【小测验】
            {quiz}

            【复盘验收】
            {review}

            【RAG 检索资料】
            {retrieved_context}
            
            【历史对话】
            {history_text}
            """

    def _get_learning_plan(extra_goal=''):
        if learning_plan and not extra_goal:           # 如果已有生成的计划，直接返回
            return learning_plan

        tool_goal = goal

        if extra_goal:
            tool_goal = f"""
            {goal}
            
            学生补充目标：
            {extra_goal}

            如果已经存在学习计划，请在学习计划基础上调整，而不是直接推翻。
            旧学习计划：
            {learning_plan}"""

        return generate_learning_plan(
            goal=tool_goal,
            level=level,
            hours=hours,
            history=history_text,
        )

    def _get_tutor_explanation(topic=''):
        if tutor_explanation and not topic:
            return tutor_explanation

        tool_goal = goal
        tool_plan = _get_learning_plan(extra_goal=topic)
        tool_context = _get_retrieved_context(query=topic or goal)

        if topic:
            tool_goal = f"{goal}\n学生当前关注主题：{topic}"

        return generate_tutor_explanation(
            goal=tool_goal,
            level=level,
            learning_plan=tool_plan,
            retrieved_context=tool_context,
            history=history_text,
        )


    @tool
    def rag_tool(query: str, k: int = 3) -> str:
        """
        Search EduPilot local RAG knowledge base.

        Use this tool when the student asks about LangGraph, RAG, Agent,
        memory, tool calling, code concepts, or uploaded learning materials.
        """

        return _get_retrieved_context(query=query, k=k)

    @tool
    def get_current_context_tool() -> str:
        """
        Read current EduPilot session context.

        Use this tool when the student asks to summarize today's learning,
        compare workflow and agent mode, or answer based on current session.
        """

        return _format_current_context()

    @tool
    def plan_tool(extra_goal: str = '') -> str:
        """
        Generate a learning plan with EduPilot planner module.

        Use this tool when the student asks for a plan, schedule, roadmap,
        implementation steps, or what to do next.
        """

        return _get_learning_plan(extra_goal=extra_goal)

    @tool
    def tutor_tool(topic: str = '') -> str:
        """
        Generate tutor explanation with EduPilot tutor module.

        Use this tool when the student asks for teaching, explanation,
        concept clarification, or a guided walkthrough.
        """

        return _get_tutor_explanation(topic=topic)

    @tool
    def quiz_tool(topic: str = '') -> str:
        """
        Generate a new quiz with EduPilot quiz module.

        Use this tool when the student asks for a quiz, practice questions,
        self-check questions, or a new assessment. This tool can work even
        before Workflow Mode runs by generating missing plan/explanation first.
        """

        tool_goal = goal
        if topic:
            tool_goal = f'{goal}\n学生指定测验主题：{topic}'

        tool_plan = _get_learning_plan(extra_goal=topic)
        tool_tutorial = _get_tutor_explanation(topic=topic)
        tool_context = _get_retrieved_context(query=topic or goal)

        return generate_quiz(
            goal=tool_goal,
            level=level,
            learning_plan=tool_plan,
            tutor_explanation=tool_tutorial,
            retrieved_context=tool_context,
        )

    @tool
    def grade_quiz_answer_tool(student_answer: str, quiz_text: str) -> str:
        """
        Grade the student's quiz answer with EduPilot grading module.

        Use this tool when the student provides an answer and asks for grading,
        correction, feedback, score, or reference answers. If the current session
        has no quiz, the student can provide quiz_text together with the answer.
        """

        if not student_answer:
            return "请先提供学生答案"

        if not quiz_text:
            return "当前没有可调用的小测试，请先调用 quiz_tool 生成题目"

        tool_tutorial = _get_tutor_explanation(topic='批改学生测验答案')
        active_quiz = quiz_text or quiz

        return grade_quiz(
            goal=goal,
            level=level,
            tutor_explanation=tool_tutorial,
            quiz=active_quiz,
            student_answer=student_answer,
        )

    @tool
    def qa_tool(question: str) -> str:
        """
        Answer a follow-up question with EduPilot QA module.

        Use this tool when the student asks a conceptual, coding, or project question
        that should be answered based on current session context and RAG retrieval.
        """

        if not question:
            return "请先输入问题"

        tool_plan = _get_learning_plan(extra_goal=question)
        tool_tutorial = _get_tutor_explanation(topic=question)
        tool_context = _get_retrieved_context(query=question)

        return answer_followup_question(
            goal=goal,
            level=level,
            learning_plan=tool_plan,
            tutor_explanation=tool_tutorial,
            retrieved_context=tool_context,
            qa_history=qa_history,
            question=question,
        )

    @tool
    def review_tool(focus: str = '') -> str:
        """
        Generate review checklist with EduPilot reviewer module.

        Use this tool when the student asks for review, acceptance criteria,
        self-check questions, summary, or today's completion standard.
        """

        tool_plan = _get_learning_plan(extra_goal=focus)
        tool_tutorial = _get_tutor_explanation(topic=focus or '学习复盘')

        return generate_review(
            goal=goal,
            level=level,
            hours=hours,
            learning_plan=tool_plan,
            tutor_explanation=tool_tutorial,
            history=history_text,
        )


    return [
        rag_tool,
        get_current_context_tool,
        plan_tool,
        tutor_tool,
        quiz_tool,
        grade_quiz_answer_tool,
        qa_tool,
        review_tool,
    ]