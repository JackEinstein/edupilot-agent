from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.planner import generate_learning_plan


class EduPilotState(TypedDict):
    goal: str
    level: str
    hours: int
    plan: str


def planner_node(state: EduPilotState) -> EduPilotState:
    """A node in LangGraph that generate today's learning plan"""
    plan = generate_learning_plan(
        goal=state['goal'],
        level=state['level'],
        hours=state['hours'],
    )

    return {
        **state,
        'plan': plan,
    }


def build_graph():
    """Build LangGraph Workflow"""
    graph = StateGraph(EduPilotState)

    graph.add_node('planner', planner_node)
    graph.set_entry_point('planner')
    graph.add_edge('planner', END)

    return graph.compile()


edupilot_graph = build_graph()


def run_graph(goal, level, hours):
    """Run LangGraph Workflow"""
    result = edupilot_graph.invoke({
        'goal': goal,
        'level': level,
        'hours': hours,
        'plan': '',
    })

    return result['plan']
