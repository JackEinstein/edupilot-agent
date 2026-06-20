from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class SkillSpec:
    """
    Skill 是 EduPilot 的能力单元。
    它不是 LangChain Tool，而是更高层的能力描述：
    - 什么时候触发；
    - 用哪个 Prompt；
    - 关联哪些工具；
    - 能力边界是什么。
    """
    name: str
    display_name: str
    description: str
    trigger_keywords: List[str]
    prompt_name: str
    related_tools: List[str]
    demo_query: str


SKILL_REGISTRY: Dict[str, SkillSpec] = {
    "learning_planner": SkillSpec(
        name="learning_planner",
        display_name="个性化学习规划 Skill",
        description="根据学习目标、历史记录、长期记忆和 RAG 资料生成学习计划。",
        trigger_keywords=["计划", "学习路线", "怎么学", "安排", "目标", "planner"],
        prompt_name="planner",
        related_tools=["plan_learning"],
        demo_query="我想学习 LangGraph 的短期记忆机制，帮我安排今天的学习计划。",
    ),

    "rag_tutor": SkillSpec(
        name="rag_tutor",
        display_name="RAG 导师讲解 Skill",
        description="基于本地知识库检索结果进行导师式讲解。",
        trigger_keywords=["讲解", "解释", "原理", "是什么", "为什么", "tutor", "RAG"],
        prompt_name="tutor",
        related_tools=["rag_search", "tutor_explain"],
        demo_query="请结合知识库资料讲解一下 LangGraph 的 StateGraph。",
    ),

    "quiz_generation": SkillSpec(
        name="quiz_generation",
        display_name="小测验生成 Skill",
        description="根据学习内容自动生成选择题和简答题。",
        trigger_keywords=["测验", "小测", "题目", "出题", "quiz"],
        prompt_name="quiz",
        related_tools=["generate_quiz"],
        demo_query="根据刚才的导师讲解，帮我生成一组小测验。",
    ),

    "quiz_grading": SkillSpec(
        name="quiz_grading",
        display_name="智能批改 Skill",
        description="根据学生答案、测验题和学习资料进行逐题批改。",
        trigger_keywords=["批改", "评分", "答案", "哪里错", "grade"],
        prompt_name="grade",
        related_tools=["grade_quiz"],
        demo_query="这是我的答案：1A 2C，简答题我认为 RAG 是先检索再生成，请帮我批改。",
    ),

    "followup_qa": SkillSpec(
        name="followup_qa",
        display_name="追问答疑 Skill",
        description="基于当前学习上下文继续回答学生追问。",
        trigger_keywords=["追问", "不懂", "为什么", "能不能再解释", "qa", "问题"],
        prompt_name="qa",
        related_tools=["answer_followup_question"],
        demo_query="我还是不懂 thread_id 为什么能实现短期记忆。",
    ),

    "reflection_review": SkillSpec(
        name="reflection_review",
        display_name="Reflection 自检 Skill",
        description="对各模块输出进行自检、纠错和结构优化。",
        trigger_keywords=["反思", "检查", "优化回答", "自检", "reflection"],
        prompt_name="reflection",
        related_tools=["reflect_answer"],
        demo_query="请检查刚才的导师讲解有没有遗漏关键点。",
    ),

    "memory_personalization": SkillSpec(
        name="memory_personalization",
        display_name="长期记忆个性化 Skill",
        description="召回和写入用户长期学习画像，用于个性化计划和讲解。",
        trigger_keywords=["记忆", "长期记忆", "我的偏好", "上次", "memory"],
        prompt_name="planner",
        related_tools=["retrieve_long_term_memory", "save_long_term_memory"],
        demo_query="你还记得我之前学习 RAG 时哪里最容易卡住吗？",
    ),

    "retrieval_debug": SkillSpec(
        name="retrieval_debug",
        display_name="RAG 检索调试 Skill",
        description="展示 rough、light_rerank、model_rerank 三种检索模式的召回与排序结果。",
        trigger_keywords=["检索", "召回", "rerank", "排序", "debug", "调试"],
        prompt_name="tutor",
        related_tools=["rag_search"],
        demo_query="用三种 RAG 检索模式对 LangGraph memory 做一次对比。",
    ),
}


def list_skills() -> list[dict]:
    """
    给 Streamlit 展示所有 Skill。
    """
    return [
        {
            "name": skill.name,
            "display_name": skill.display_name,
            "description": skill.description,
            "trigger_keywords": skill.trigger_keywords,
            "prompt_name": skill.prompt_name,
            "related_tools": skill.related_tools,
            "demo_query": skill.demo_query,
        }
        for skill in SKILL_REGISTRY.values()
    ]


def detect_skills(user_input: str) -> list[SkillSpec]:
    """
    简单关键词匹配版 Skill 识别。
    Day 10 先做轻量可解释版本，后续可以升级为 LLM Router。
    """
    text = (user_input or "").lower()
    matched = []

    for skill in SKILL_REGISTRY.values():
        for keyword in skill.trigger_keywords:
            if keyword.lower() in text:
                matched.append(skill)
                break

    if matched:
        return matched

    # 兜底：默认进入 RAG 导师或学习规划
    if len(text) <= 20:
        return [SKILL_REGISTRY["rag_tutor"]]

    return [SKILL_REGISTRY["learning_planner"]]


def format_skills_for_agent(user_input: str = "") -> str:
    """
    给 ReAct system prompt 注入 Skill 描述。
    """
    if user_input:
        skills = detect_skills(user_input)
    else:
        skills = list(SKILL_REGISTRY.values())

    lines = []
    for skill in skills:
        lines.append(
            f"- {skill.display_name}：{skill.description}；"
            f"关联 Prompt：{skill.prompt_name}；"
            f"关联工具：{', '.join(skill.related_tools)}"
        )

    return "\n".join(lines)


def format_skills_for_display(user_input: str) -> str:
    """
    给 Streamlit 展示当前命中的 Skill。
    """
    skills = detect_skills(user_input)

    blocks = []
    for skill in skills:
        block = f"""
### {skill.display_name}

**能力说明**：{skill.description}

**触发关键词**：{", ".join(skill.trigger_keywords)}

**关联 Prompt**：`{skill.prompt_name}`

**关联工具**：{", ".join(f"`{tool}`" for tool in skill.related_tools)}

**示例请求**：{skill.demo_query}
""".strip()
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)