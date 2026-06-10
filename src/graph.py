from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from src.quiz import generate_quiz
from src.history import format_langgraph_messages
from src.retriever import format_retrieved_chunks
from src.reviewer import generate_review
from src.planner import generate_learning_plan
from src.tutor import generate_tutor_explanation


class EduPilotState(TypedDict):
    goal: str
    level: str
    hours: int
    retrieved_context: str
    learning_plan: str
    tutor_explanation: str
    quiz: str
    review: str
    messages: Annotated[list[AnyMessage], add_messages]


def retriever_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that retrieve knowledge from vectorstore
    """

    retrieve = format_retrieved_chunks(
        query=state['goal'],
        k=3
    )

    return {
        'retrieved_context': retrieve,
    }


def planner_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that generate today's learning plan
    """

    history = format_langgraph_messages(state.get('messages', [])[:-1])

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

    history = format_langgraph_messages(state.get('messages', [])[:-1])

    explanation = generate_tutor_explanation(
        goal=state['goal'],
        level=state['level'],
        learning_plan=state['learning_plan'],
        retrieved_context=state['retrieved_context'],
        history=history,
    )

    return {
        'tutor_explanation': explanation,
    }


def quiz_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that generate today's quiz and correct student's answer
    """

    # history = format_history(state.get('messages', [])[:-1])

    quiz = generate_quiz(
        goal=state['goal'],
        level=state['level'],
        learning_plan=state['learning_plan'],
        tutor_explanation=state['tutor_explanation'],
        retrieved_context=state['retrieved_context'],
    )

    return {
        'quiz': quiz,
    }


def reviewer_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that review knowledge in the past
    """

    history = format_langgraph_messages(state.get('messages', [])[:-1])

    review = generate_review(
        goal=state['goal'],
        level=state['level'],
        hours=state['hours'],
        learning_plan=state['learning_plan'],
        tutor_explanation=state['tutor_explanation'],
        history=history,
    )

    return {
        'review': review,
        'messages': [
            AIMessage(
                content=f"""
                本轮复盘：{review}
                本轮小测：{state['quiz']}
                """
                )
        ],
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

    graph.add_edge(START, 'retriever')
    graph.add_edge('retriever', 'planner')
    graph.add_edge('planner', 'tutor')
    graph.add_edge('tutor', 'quiz')
    graph.add_edge('quiz', 'reviewer')
    graph.add_edge('reviewer', END)

    return graph.compile(checkpointer=checkpointer)


edupilot_graph = build_graph()


def run_graph(goal, level, hours, thread_id='default'):
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
        'retrieved_context': '',
        'learning_plan': '',
        'tutor_explanation': '',
        'quiz': '',
        'review': '',
        'messages': [HumanMessage(content=user_message)],
    }

    config = {
        'configurable': {
            'thread_id': thread_id,
        }
    }

    result = edupilot_graph.invoke(initial_state, config)
    return result
