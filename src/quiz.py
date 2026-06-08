from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import get_llm


def generate_quiz(goal, level, learning_plan, tutor_explanation, retrieved_context):
    """
    根据学习计划和当前阶段，生成一套阶段测试题
    """

    llm = get_llm()

    system_message = SystemMessage(
        content=("""
        你是 EduPilot Agent 的测验出题老师。
        
        你的任务是根据用户的学习目标、当前水平、学习计划、导师讲解内容和参考资料，
        生成一组用于检查学习效果的小测验。
        
        出题要求：
        1. 一共生成 3 道题；
        2. 难度要符合用户当前水平；
        3. 题目要覆盖本轮学习的核心内容；
        4. 不要出偏题、怪题、纯记忆题；
        5. 尽量结合用户的项目场景；
        6. 题目类型可以包含：概念理解、代码理解、项目应用；
        7. 暂时不要直接给答案；
        8. 输出 Markdown 格式。
        
        输出格式必须如下：
        
        ## 📝 本轮小测验
        
        ### 第 1 题：概念理解
        题目：...
        
        ### 第 2 题：代码理解
        题目：...
        
        ### 第 3 题：项目应用
        题目：...
        
        ## 答题要求
        请按 1、2、3 的顺序作答。每题可以用 2-5 句话回答。
        """
        )
    )

    human_message = HumanMessage(
        content=(f"""
        【用户学习目标】
        {goal}
        
        【用户当前水平】
        {level}
        
        【今日学习计划】
        {learning_plan}
        
        【导师讲解内容】
        {tutor_explanation}
        
        【RAG 检索到的参考资料】
        {retrieved_context}
        
        请基于以上内容生成 3 道小测验题。
        """
        )
    )

    messages = [system_message, human_message]

    response = llm.invoke(messages)
    return response.content


def grade_quiz(goal, level, tutor_explanation, quiz, student_answer):
    """
    根据quiz和学生的作答情况，生成批改反馈
    """

    llm = get_llm()

    system_message = SystemMessage(
        content=("""
        你是 EduPilot Agent 的测验批改老师。

        你的任务是根据小测验题目、学生答案和导师讲解内容，对学生答案进行批改。
        
        批改要求：
        1. 给出总分，满分 100 分；
        2. 分别评价每一道题；
        3. 指出答得好的地方；
        4. 指出理解不准确或遗漏的地方；
        5. 给出更好的参考回答；
        6. 最后总结用户暴露出的薄弱点；
        7. 给出下一步学习建议；
        8. 语气要像老师辅导学生，不要打击用户。
        
        输出 Markdown 格式。
        
        输出格式建议如下：
        
        ## ✅ Quiz 批改反馈
        
        ### 总分：xx / 100
        
        ### 第 1 题反馈
        - 得分：...
        - 评价：...
        - 参考回答：...
        
        ### 第 2 题反馈
        - 得分：...
        - 评价：...
        - 参考回答：...
        
        ### 第 3 题反馈
        - 得分：...
        - 评价：...
        - 参考回答：...
        
        ## 暴露出的薄弱点
        ...
        
        ## 下一步建议
        ...
        """
        )
    )

    human_message = HumanMessage(
        content=(f"""
        【用户学习目标】
        {goal}
        
        【用户当前水平】
        {level}
        
        【导师讲解内容】
        {tutor_explanation}
        
        【小测验题目】
        {quiz}
        
        【学生答案】
        {student_answer}
        
        请对学生答案进行批改。
        """
        )
    )

    messages = [system_message, human_message]

    response = llm.invoke(messages)
    return response.content