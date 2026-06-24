from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.reflection import run_reflection_loop
from src.llm import get_llm
from src.tools import build_tools
from src.prompts import render_prompt
from src.skills import format_skills_for_agent


react_agent_checkpointer = InMemorySaver()


def build_system_prompt(user_input: str = "", short_term_memory: str = ""):
    """
    Build system prompt for EduPilot ReAct-style Tool Calling Agent.
    """

    return render_prompt(
        "react_system",
        skill_context=format_skills_for_agent(user_input),
        short_term_memory=short_term_memory or "暂无 Redis 短期会话记忆。",
    )


def extract_tool_trace(messages: list[AnyMessage]) -> list[dict]:
    """
    Extract tool call trace for Streamlit display.
    """

    trace = []

    for msg in messages:

        if msg.type == 'ai':
            tool_calls = getattr(msg, 'tool_calls', None) or []
            for call in tool_calls:
                trace.append(
                    {
                        'type': 'tool_call',
                        'name': call.get('name', 'unknown_tool'),
                        'content': str(call.get('args', {})),
                    }
                )

        if msg.type == 'tool':
            trace.append(
                {
                    'type': 'tool_result',
                    'name': getattr(msg, 'name', 'unknown_tool'),
                    'content': str(getattr(msg, 'content', ''))[:1500],
                }
            )

    return trace


def run_react_agent(context: dict, question: str, thread_id: str = 'default', enable_reflection: bool = True) -> dict:
    """
    Run EduPilot ReAct-style Tool Calling Agent.
    """

    agent = create_agent(
        model=get_llm(),
        tools=build_tools(context),
        system_prompt=build_system_prompt(
            user_input=question,
            short_term_memory=context.get("short_term_memory", ""),
        ),
        checkpointer=react_agent_checkpointer,
    )

    config = {
        'configurable': {
            'thread_id': f'react-agent-{thread_id}',
        }
    }

    result = agent.invoke(
        {
            'messages': [
                {
                    'role': 'user',
                    'content': question
                }
            ]
        },
        config=config,
    )

    messages = result.get('messages', [])
    draft_answer = str(getattr(messages[-1], 'content', '')) if messages else ''

    trace = extract_tool_trace(messages)

    reflection_result = run_reflection_loop(
        context=context,
        question=question,
        draft_answer=draft_answer,
        trace=trace,
        enable_reflection=enable_reflection,
    )

    return {
        'draft_answer': reflection_result['draft_answer'],
        'final_answer': reflection_result['final_answer'],
        'reflection': reflection_result['reflection'],
        'used_reflection': reflection_result['used_reflection'],
        'trace': trace
    }


