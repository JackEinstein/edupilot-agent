from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from src.long_term_memory import record_workflow_memory, retrieve_long_term_memory
from src.reflection import reflect_node_output, reflect_workflow_output
from src.quiz import generate_quiz
from src.short_term_memory import format_langgraph_messages
from src.retriever import format_retrieved_chunks, DEFAULT_RETRIEVAL_MODE
from src.reviewer import generate_review
from src.planner import generate_learning_plan
from src.tutor import generate_tutor_explanation


class EduPilotState(TypedDict, total=False):
    goal: str
    level: str
    hours: int
    enable_reflection: bool

    rag_top_k: int
    rag_fetch_k: int
    retrieval_mode: str

    retrieved_context: str

    learning_plan: str

    tutor_explanation: str
    tutor_reflection: str

    quiz: str
    quiz_reflection: str

    review: str
    review_reflection: str

    workflow_draft_answer: str
    workflow_reflection: str
    final_answer: str

    messages: Annotated[list[AnyMessage], add_messages]

    long_term_memory: str


def _build_light_reflection_context(state: EduPilotState, stage: str, history: str = "") -> dict:
    """
    为不同节点构造最小必要上下文。

    原则：
    - 不传整个 state；
    - 只传判断当前节点质量所需的信息；
    - 减少 token 成本和上下文噪声。
    """

    base_context = {
        "学习目标": state.get("goal", ""),
        "学生水平": state.get("level", ""),
    }

    if stage == "tutor":
        return {
            **base_context,
            "今日学习计划": state.get("learning_plan", ""),
            "RAG 检索资料": state.get("retrieved_context", ""),
            "历史对话": history,
        }

    if stage == "quiz":
        return {
            **base_context,
            "今日学习计划": state.get("learning_plan", ""),
            "导师讲解": state.get("tutor_explanation", ""),
        }

    if stage == "reviewer":
        return {
            **base_context,
            "今日学习计划": state.get("learning_plan", ""),
            "导师讲解": state.get("tutor_explanation", ""),
            "本轮小测验": state.get("quiz", ""),
            "历史对话": history,
        }

    if stage == "global":
        return {
            **base_context,
            "学习时间": f'{state.get("hours", "")} 小时',
            "今日学习计划": state.get("learning_plan", ""),
            "导师讲解": state.get("tutor_explanation", ""),
            "本轮小测验": state.get("quiz", ""),
            "复盘验收": state.get("review", ""),
        }

    return base_context


def retriever_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that retrieve knowledge from vectorstore
    """

    retrieved_knowledge = format_retrieved_chunks(
        query=state['goal'],
        k=state.get('rag_top_k', 3),
        fetch_k=state.get('rag_fetch_k', None),
        retrieval_mode=state.get('retrieval_mode', DEFAULT_RETRIEVAL_MODE),
    )

    retrieved_memory = retrieve_long_term_memory(
        query=state['goal'],
        k=4
    )

    retrieved_context = f"""
        【从知识库检索的文本】
        {retrieved_knowledge}
        
        【从长期记忆检索的文本】
        {retrieved_memory}
        """

    return {
        'retrieved_context': retrieved_context,
        'long_term_memory': retrieved_memory,
    }


def planner_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that generate today's learning plan
    """

    short_term_memory = format_langgraph_messages(state.get('messages', [])[:-1])
    long_term_memory = state.get('long_term_memory', '')
    history = f"""
        【短期记忆】
        {short_term_memory}
        
        【长期记忆】
        {long_term_memory}
        """

    plan = generate_learning_plan(
        goal=state['goal'],
        level=state['level'],
        hours=state['hours'],
        history=history,
    )

    return {
        'learning_plan': plan,
    }


def tutor_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that explain today's knowledge
    """

    short_term_memory = format_langgraph_messages(state.get('messages', [])[:-1])
    long_term_memory = state.get('long_term_memory', '')
    history = f"""
        【短期记忆】
        {short_term_memory}

        【长期记忆】
        {long_term_memory}
        """

    explanation = generate_tutor_explanation(
        goal=state['goal'],
        level=state['level'],
        learning_plan=state['learning_plan'],
        retrieved_context=state['retrieved_context'],
        history=history,
    )

    tutor_reflection = reflect_node_output(
        stage='tutor',
        context=_build_light_reflection_context(
            state=state,
            stage='tutor',
            history=history,
        ),
        draft_output=explanation,
        enable_reflection=state.get('enable_reflection', True),
    )

    return {
        'tutor_explanation': tutor_reflection['improved_output'],
        'tutor_reflection': tutor_reflection['reflection'],
    }


def quiz_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that generate today's quiz and correct student's answer
    """

    short_term_memory = format_langgraph_messages(state.get('messages', [])[:-1])
    long_term_memory = state.get('long_term_memory', '')
    history = f"""
        【短期记忆】
        {short_term_memory}

        【长期记忆】
        {long_term_memory}
        """

    quiz = generate_quiz(
        goal=state['goal'],
        level=state['level'],
        learning_plan=state['learning_plan'],
        tutor_explanation=state['tutor_explanation'],
        retrieved_context=state['retrieved_context'],
    )

    quiz_reflection = reflect_node_output(
        stage='quiz',
        context=_build_light_reflection_context(
            state=state,
            stage='quiz',
            history=history,
        ),
        draft_output=quiz,
        enable_reflection=state.get('enable_reflection', True),
    )

    return {
        'quiz': quiz_reflection['improved_output'],
        'quiz_reflection': quiz_reflection['reflection'],
    }


def reviewer_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that review knowledge in the past
    """

    short_term_memory = format_langgraph_messages(state.get('messages', [])[:-1])
    long_term_memory = state.get('long_term_memory', '')
    history = f"""
        【短期记忆】
        {short_term_memory}

        【长期记忆】
        {long_term_memory}
        """

    review = generate_review(
        goal=state['goal'],
        level=state['level'],
        hours=state['hours'],
        learning_plan=state['learning_plan'],
        tutor_explanation=state['tutor_explanation'],
        history=history,
    )

    review_reflection = reflect_node_output(
        stage='reviewer',
        context=_build_light_reflection_context(
            state=state,
            stage='reviewer',
            history=history,
        ),
        draft_output=review,
        enable_reflection=state.get('enable_reflection', True),
    )

    workflow_draft_answer = f"""
# EduPilot Workflow 学习方案

## 【今日计划】

{state['learning_plan']}

## 【导师讲解】

{state['tutor_explanation']}

## 【小测验收】

{state['quiz']}

## 【复盘与总结】

{review_reflection['improved_output']}
"""

    return {
        'review': review_reflection['improved_output'],
        'review_reflection': review_reflection['reflection'],
        'workflow_draft_answer': workflow_draft_answer,
    }


def reflection_node(state: EduPilotState) -> EduPilotState:
    """
    Final lightweight global Reflection node.

    It checks the whole workflow output and writes final AIMessage
    into LangGraph memory.
    """

    short_term_memory = format_langgraph_messages(state.get('messages', [])[:-1])
    long_term_memory = state.get('long_term_memory', '')
    history = f"""
        【短期记忆】
        {short_term_memory}

        【长期记忆】
        {long_term_memory}
        """

    final_reflection = reflect_workflow_output(
        context=_build_light_reflection_context(
            state=state,
            stage='global',
            history=history,
        ),
        draft_answer=state['workflow_draft_answer'],
        enable_reflection=state.get('enable_reflection', True),
    )

    return {
        'final_answer': final_reflection['final_answer'],
        'workflow_reflection': final_reflection['reflection'],
        'messages': [
            AIMessage(content=final_reflection['final_answer'])
        ]
    }


checkpointer = InMemorySaver()


def build_graph():
    """
    Build LangGraph Workflow
    """

    graph = StateGraph(EduPilotState)

    graph.add_node('retriever', retriever_node)
    graph.add_node('planner', planner_node)
    graph.add_node('tutor', tutor_node)
    graph.add_node('quiz', quiz_node)
    graph.add_node('reviewer', reviewer_node)
    graph.add_node('reflection', reflection_node)

    graph.add_edge(START, 'retriever')
    graph.add_edge('retriever', 'planner')
    graph.add_edge('planner', 'tutor')
    graph.add_edge('tutor', 'quiz')
    graph.add_edge('quiz', 'reviewer')
    graph.add_edge('reviewer', 'reflection')
    graph.add_edge('reflection', END)

    return graph.compile(checkpointer=checkpointer)


edupilot_graph = build_graph()


def run_graph(
    goal,
    level,
    hours,
    thread_id='default',
    enable_reflection=True,
    rag_top_k=3,
    rag_fetch_k=8,
    retrieval_mode=DEFAULT_RETRIEVAL_MODE,
):
    """
    Run LangGraph Workflow
    """

    user_message = f"""
    学习目标：{goal}
    当前水平：{level}
    今天可用学习时间：{hours}
    """

    initial_state = {
        'goal': goal,
        'level': level,
        'hours': hours,
        'enable_reflection': enable_reflection,

        'rag_top_k': rag_top_k,
        'rag_fetch_k': rag_fetch_k,
        'retrieval_mode': retrieval_mode,

        'retrieved_context': '',

        'learning_plan': '',

        'tutor_explanation': '',
        'tutor_reflection': '',

        'quiz': '',
        'quiz_reflection': '',

        'review': '',
        'review_reflection': '',

        'workflow_draft': '',
        'workflow_reflection': '',
        'final_answer': '',

        'messages': [HumanMessage(content=user_message)],
    }

    config = {
        'configurable': {
            'thread_id': thread_id,
        }
    }

    result = edupilot_graph.invoke(initial_state, config)

    # 保存长期记忆
    try:
        result['memory_result'] = record_workflow_memory(
            goal=goal,
            level=level,
            workflow_result=result,
        )

    except Exception as exc:
        result['memory_result'] = {
            'saved': False,
            'action': 'error',
            'reason': str(exc),
        }

    return result
