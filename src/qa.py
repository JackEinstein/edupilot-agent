from src.long_term_memory import record_qa_memory
from src.short_term_memory import format_qa_history
from src.llm import get_llm
from src.retriever import format_retrieved_chunks, DEFAULT_RETRIEVAL_MODE


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

    system_message = """
        你是 EduPilot Agent 的追问答疑导师。
    
        你的任务是回答学生在导师讲解之后提出的追问问题。
        
        你必须优先基于以下信息回答：
        1. 今天的学习目标；
        2. 学生当前水平；
        3. 今天的学习计划；
        4. 今天的导师讲解；
        5. 今天已经检索到的 RAG 资料；
        6. 针对当前追问重新检索到的 RAG 资料；
        7. 之前的追问答疑历史。
        
        回答要求：
        1. 先直接回答学生问题；
        2. 再解释背后的原因；
        3. 如果涉及代码或项目结构，结合 EduPilot 当前项目说明；
        4. 如果学生的问题暴露出误区，要明确指出；
        5. 如果资料不足，要诚实说明“当前知识库资料不足”，不要编造；
        6. 不要泛泛而谈，要围绕本轮学习内容；
        7. 语气像老师带学生，清楚、耐心、具体。
        
        输出 Markdown 格式。
        """

    human_message = f"""
        【今天的学习目标】
        {goal}
        
        【学生当前水平】
        {level}
        
        【今天的学习计划】
        {learning_plan}
        
        【今天的导师讲解】
        {tutor_explanation}
        
        【今天原始 RAG 检索资料】
        {retrieved_context}
        
        【针对当前追问重新检索到的资料】
        {followup_context}
        
        【之前的追问答疑历史】
        {qa_history_text}
        
        【学生当前追问】
        {question}
        
        请回答学生当前追问。
        """

    messages = [system_message, human_message]

    answer = llm.invoke(messages)

    # 把 qa 对话保存进长期记忆
    try:
        record_qa_memory(
            goal=goal,
            level=level,
            question=question,
            answer=answer,
        )

    except Exception:
        pass

    return answer.content
