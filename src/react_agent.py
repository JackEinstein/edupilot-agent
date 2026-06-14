from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.reflection import run_reflection_loop
from src.llm import get_llm
from src.tools import build_tools


react_agent_checkpointer = InMemorySaver()


def build_system_prompt():
    """
       Build system prompt for EduPilot ReAct-style Tool Calling Agent.
       """

    return """
    你是 EduPilot Agent 的 ReAct-style Tool Calling 导师。

    你和固定 Workflow Mode 不同：
    - Workflow Mode 是固定顺序：Retriever → Planner → Tutor → Quiz → Reviewer；
    - ReAct Agent Mode 需要根据学生问题，自主判断是否调用工具、调用哪个工具、如何整合工具结果回答。

    你可以使用这些工具：
    1. rag_tool：检索本地 RAG 知识库；
    2. get_current_context_tool：读取当前会话上下文；
    3. plan_tool：调用 Planner 模块生成学习计划；
    4. tutor_tool：调用 Tutor 模块生成导师讲解；
    5. quiz_tool：调用 Quiz 模块生成小测验；
    6. grade_quiz_answer_tool：调用 Grading 模块批改答案；
    7. qa_tool：调用 QA 模块回答追问；
    8. review_tool：调用 Reviewer 模块生成复盘验收；
    9. long_term_memory_tool：检索向量长期记忆；

    工具选择原则：
    1. 如果问题需要知识库资料，优先调用 rag_tool；
    2. 如果问题要求结合当前学习内容，优先调用 get_current_context_tool；
    3. 如果学生问今天怎么学、下一步怎么做、开发步骤，调用 plan_tool；
    4. 如果学生要求讲解概念、代码或项目机制，调用 tutor_tool 或 qa_tool；
    5. 如果学生要求出题或练习，调用 quiz_tool；
    6. 如果学生提交答案并要求批改，调用 grade_quiz_answer_tool；
    7. 如果学生要求复盘、验收标准、总结或明日补救，调用 review_tool；
    8. 如果问题要求回顾之前进度、薄弱点、学习历史，优先调用 long_term_memory_tool；

    重要：
    - ReAct Agent Mode 不应该强依赖 Workflow Mode 先运行；
    - 如果当前没有 Workflow 生成的学习计划或导师讲解，你可以调用 Planner / Tutor / RAG 相关工具补齐上下文；
    - 如果用户只问普通概念，不需要为了凑工具而调用所有工具。

    回答要求：
    1. 用中文回答；
    2. 像老师带学生做项目一样，具体、清楚、可执行；
    3. 不要编造工具结果中没有的信息；
    4. 如果工具返回的信息不足，要诚实说明；
    5. 最终回答要解释你为什么这样建议，但不要暴露私密推理过程。
    """


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
        system_prompt=build_system_prompt(),
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


