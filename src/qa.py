from langchain_core.messages import SystemMessage

from src.long_term_memory import record_qa_memory
from src.short_term_memory import format_qa_history
from src.llm import get_llm
from src.retriever import format_retrieved_chunks, DEFAULT_RETRIEVAL_MODE
from src.prompts import render_prompt


def answer_followup_question(
    goal,
    level,
    learning_plan,
    tutor_explanation,
    retrieved_context,
    qa_history,
    question,
    rag_top_k=3,
    rag_fetch_k=8,
    retrieval_mode=DEFAULT_RETRIEVAL_MODE,
):
    """
    根据本轮学习内容、RAG知识库、问答历史回答学生的问题
    """

    llm = get_llm()

    qa_history_text = format_qa_history(qa_history)

    followup_context = format_retrieved_chunks(
        query=question,
        k=rag_top_k,
        fetch_k=rag_fetch_k,
        retrieval_mode=retrieval_mode,
    )

    prompt = render_prompt(
        "qa",
        question=question,
        learning_goal=f"""
        学习目标：{goal}
        当前水平：{level}
        """,
        learning_plan=learning_plan,
        tutor_answer=tutor_explanation,
        rag_context=f"""
        【今天原始 RAG 检索资料】
        {retrieved_context}

        【针对当前追问重新检索到的资料】
        {followup_context}
        """,
        short_term_history=qa_history_text,
        long_term_memory="",
    )

    messages = [SystemMessage(content=prompt)]

    answer = llm.invoke(messages)

    # 把 qa 对话保存进长期记忆
    try:
        record_qa_memory(
            goal=goal,
            level=level,
            question=question,
            answer=answer.content,
        )

    except Exception:
        pass

    return answer.content
