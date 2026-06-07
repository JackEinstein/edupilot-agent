from typing import TypedDict

from langgraph.graph import StateGraph, END, START

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
    review: str


def retriever_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that retrieve knowledge from vectorstore
    """

    retrieve = format_retrieved_chunks(
        query=state['goal'],
        k=3
    )

    return {
        **state,
        'retrieved_context': retrieve,
    }


def planner_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that generate today's learning plan
    """

    plan = generate_learning_plan(
        goal=state['goal'],
        level=state['level'],
        hours=state['hours'],
    )

    return {
        **state,
        'learning_plan': plan,
    }


def tutor_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that explain today's knowledge
    """

    explanation = generate_tutor_explanation(
        goal=state['goal'],
        level=state['level'],
        learning_plan=state['learning_plan'],
        retrieved_context=state['retrieved_context'],
    )

    return {
        **state,
        'tutor_explanation': explanation,
    }


def reviewer_node(state: EduPilotState) -> EduPilotState:
    """
    A node in LangGraph that review knowledge in the past
    """

    review = generate_review(
        goal=state['goal'],
        level=state['level'],
        hours=state['hours'],
        learning_plan=state['learning_plan'],
        tutor_explanation=state['tutor_explanation'],
    )

    return {
        **state,
        'review': review,
    }


def build_graph():
    """
    Build LangGraph Workflow
    """

    graph = StateGraph(EduPilotState)

    graph.add_node('retriever', retriever_node)
    graph.add_node('planner', planner_node)
    graph.add_node('tutor', tutor_node)
    graph.add_node('reviewer', reviewer_node)

    graph.add_edge(START, 'retriever')
    graph.add_edge('retriever', 'planner')
    graph.add_edge('planner', 'tutor')
    graph.add_edge('tutor', 'reviewer')
    graph.add_edge('reviewer', END)

    return graph.compile()


edupilot_graph = build_graph()


def run_graph(goal, level, hours):
    """
    Run LangGraph Workflow
    """

    initial_state = {
        'goal': goal,
        'level': level,
        'hours': hours,
        'retrieved_context': '',
        'learning_plan': '',
        'tutor_explanation': '',
        'review': '',
    }

    result = edupilot_graph.invoke(initial_state)
    return result
