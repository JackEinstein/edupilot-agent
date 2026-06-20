# EduPilot Agent

EduPilot Agent 是一个面向个性化学习场景的 AI 学习助手项目，支持从学习目标输入、知识库检索、学习计划生成、导师讲解、小测验、智能批改、追问答疑，到 ReAct Tool Calling、Reflection 自检和 Chroma 向量长期记忆的完整闭环。

当前版本重点新增 **Prompt Registry 与 Skill Registry**：  
- 通过 `src/prompts.py` 统一管理 Planner、Tutor、Quiz、Grading、QA、Reviewer、Reflection、ReAct System Prompt 等模块提示词；
- 通过 `src/skills.py` 抽象 EduPilot 的高层学习能力，并在 Streamlit 前端展示用户输入命中的 Skill、关联 Prompt 和相关 Tool；
- ReAct 模式会将当前用户请求命中的 Skill 信息注入 system prompt，辅助模型选择工具；
- Workflow 模式保持原有固定 LangGraph 主链路，Skill 不写入 `graph.py`，只作为前端展示和可解释性增强。

---

## 1. 项目定位

本项目主要用于展示一个教育方向 AI Agent 的端到端工程能力，适合作为 AI Agent / RAG / LangGraph / Tool Calling / Memory / Reflection / Prompt Engineering 方向的简历项目 Demo。

用户输入学习目标后，系统可以自动完成：

1. 从本地知识库中检索相关学习资料；
2. 结合学习目标生成今日学习计划；
3. 根据计划和资料生成导师式讲解；
4. 自动生成本轮小测验；
5. 支持学生作答并进行智能批改；
6. 支持基于本轮学习内容继续追问；
7. 支持 ReAct Agent 自主选择工具；
8. 支持 Workflow Reflection 和 ReAct Reflection；
9. 支持基于 Chroma 的向量数据库长期记忆；
10. 支持长期记忆的轻量级遗忘机制；
11. 支持 RAG 三种检索模式：`rough`、`light_rerank`、`model_rerank`；
12. 支持 Prompt Registry 与 Skill Registry 展示。

---

## 2. 当前版本核心功能

### 2.1 Workflow Mode

Workflow Mode 是固定工作流模式，基于 LangGraph 串联多个学习节点，适合稳定完成一轮完整学习闭环。

```text
User Input
    ↓
Retriever: RAG Retrieval / Rerank + Long-term Memory
    ↓
Planner
    ↓
Tutor + Lightweight Reflection
    ↓
Quiz + Lightweight Reflection
    ↓
Reviewer + Lightweight Reflection
    ↓
Workflow Reflection
    ↓
Final Answer
```

Workflow Mode 当前保持固定链路，不根据 Skill 做条件分支，避免破坏已经稳定运行的主流程。

---

### 2.2 ReAct Agent Mode

ReAct Agent Mode 面向开放式问题处理，支持 Agent 根据用户问题自主选择工具。

当前可用工具包括：

- `rag_tool`：检索本地知识库；
- `get_current_context_tool`：读取当前学习上下文；
- `plan_tool`：生成或调整学习计划；
- `tutor_tool`：生成导师式讲解；
- `quiz_tool`：生成小测验；
- `grade_quiz_answer_tool`：批改学生答案；
- `qa_tool`：基于当前上下文进行追问答疑；
- `review_tool`：生成复盘与验收清单；
- `long_term_memory_tool`：检索长期记忆。

ReAct 模式新增 Skill 上下文注入：系统会根据用户输入匹配可能相关的 Skill，并将其注入 ReAct system prompt，用于辅助模型判断应该优先调用哪些工具。

---

### 2.3 RAG 本地知识库

系统支持上传 `.md`、`.txt` 等文本资料到本地知识库，并使用 Chroma 向量数据库进行检索。

RAG 检索结果会注入到 Planner、Tutor、Quiz、QA 等模块中，使生成内容尽量基于用户上传的学习资料。

---

### 2.4 RAG 召回优化与三种检索模式

当前支持三种检索模式：

#### 1. `rough`

基础向量召回模式，只使用 Chroma 的相似度检索结果。  
适合快速调试、低成本检索和对比 rerank 效果。

#### 2. `light_rerank`

轻量级 rerank 模式。  
先进行较大范围粗召回，再根据关键词重合度、标题/正文匹配、片段长度等规则进行轻量重排。  
适合本地 Demo 场景，速度快、依赖少、可解释性强。

#### 3. `model_rerank`

模型重排序预留模式。  
用于后续接入 bge-reranker / cross-encoder 等重排模型。  
当前版本保留接口与降级逻辑，便于未来扩展。

---

### 2.5 RAG / Rerank 调试面板

Streamlit 页面提供 RAG 调试面板，可查看：

- 当前检索模式；
- `top_k` 与 `fetch_k` 参数；
- 粗召回结果；
- 轻量 rerank 后的排序结果；
- 不同检索模式下的输出差异。

---

### 2.6 Reflection 自检机制

系统支持两类 Reflection：

1. **Workflow Reflection**  
   对 Tutor、Quiz、Reviewer 等节点输出进行轻量级自检，并在最终阶段进行全局自检。

2. **ReAct Reflection**  
   ReAct Agent 生成草稿回答后，会进行一次最终回答自检与改写，减少遗漏和过度简略问题。

---

### 2.7 向量数据库长期记忆

系统使用 Chroma 管理长期记忆，记录用户在学习过程中的问题、偏好、薄弱点和阶段性进展。

长期记忆可在后续学习计划、导师讲解和 QA 追问中被召回，用于实现个性化学习。

---

### 2.8 Memory Reflection 与遗忘机制

系统支持 Memory Reflection，对学习事件进行筛选，判断是否值得写入长期记忆。

同时提供轻量级遗忘机制：

- 根据最近访问时间；
- 根据访问次数；
- 根据用户与 scope；
- 过滤低价值或长期未使用记忆。

---

### 2.9 Prompt Registry

当前版本新增 `src/prompts.py`，将各模块提示词从业务代码中抽离出来，统一注册、渲染和展示。

已注册 Prompt 包括：

- `planner`：学习计划生成；
- `tutor`：导师式讲解；
- `quiz`：小测验生成；
- `grade`：测验批改；
- `qa`：追问答疑；
- `reviewer`：学习复盘；
- `reflection`：回答自检；
- `react_system`：ReAct Agent 系统提示词。

每个 Prompt 使用 `PromptSpec` 描述：

```python
@dataclass(frozen=True)
class PromptSpec:
    name: str
    version: str
    description: str
    variables: List[str]
    template: str
```

当前版本采用单模板渲染方式：

```python
prompt = render_prompt("planner", ...)
messages = [SystemMessage(content=prompt)]
```

该方式改动小、稳定性高，适合当前阶段快速完成 Prompt 模块化。后续可进一步升级为 `system_template + human_template` 的双模板结构，实现更规范的 SystemMessage / HumanMessage 分层。

---

### 2.10 Skill Registry

当前版本新增 `src/skills.py`，将 EduPilot 的高层能力抽象为 Skill。

已注册 Skill 包括：

- 个性化学习规划 Skill；
- RAG 导师讲解 Skill；
- 小测验生成 Skill；
- 智能批改 Skill；
- 追问答疑 Skill；
- Reflection 自检 Skill；
- 长期记忆个性化 Skill；
- RAG 检索调试 Skill。

每个 Skill 使用 `SkillSpec` 描述：

```python
@dataclass(frozen=True)
class SkillSpec:
    name: str
    display_name: str
    description: str
    trigger_keywords: List[str]
    prompt_name: str
    related_tools: List[str]
    demo_query: str
```

当前 Skill Router 是轻量级规则路由，主要根据关键词匹配用户输入，并返回可能命中的 Skill。  
它目前主要用于：

1. Streamlit 前端展示；
2. 说明用户意图对应的 Prompt 与 Tool；
3. 给 ReAct system prompt 注入当前可能相关的 Skill 信息；
4. 增强项目可解释性和面试展示效果。

需要说明的是，当前 Skill Router 不是生产级智能意图分类器。关键词未命中时会进入默认兜底逻辑，准确性有限。后续可以升级为规则路由 + LLM Router + 置信度判断的混合路由系统。

---

## 3. 技术栈

### 前端与交互

- Streamlit
- Python
- Markdown 文件上传与调试面板展示

### LLM 与 Agent

- LangChain
- LangGraph
- ReAct Tool Calling
- Prompt Registry
- Skill Registry
- Reflection

### RAG 与向量数据库

- Chroma
- 本地知识库
- 向量召回
- 轻量 rerank
- 模型 rerank 预留接口

### Memory

- LangGraph `InMemorySaver` 短期记忆
- Chroma 向量长期记忆
- Memory Reflection
- 遗忘机制

---

## 4. 项目结构

```text
edupilot-agent/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── knowledge/
├── storage/
│   ├── chroma/
│   └── long_term_memory/
└── src/
    ├── graph.py
    ├── planner.py
    ├── tutor.py
    ├── quiz.py
    ├── qa.py
    ├── reviewer.py
    ├── retriever.py
    ├── react_agent.py
    ├── tools.py
    ├── prompts.py
    ├── skills.py
    ├── reflection.py
    ├── long_term_memory.py
    ├── memory_reflection.py
    ├── short_term_memory.py
    ├── vector_memory.py
    └── llm.py
```

---

## 5. 环境配置

建议使用 Python 3.10。

```bash
conda create -n edupilot python=3.10
conda activate edupilot
pip install -r requirements.txt
```

配置 `.env`：

```bash
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

---

## 6. 运行方式

```bash
streamlit run app.py
```

---

## 7. 功能测试建议

### 7.1 基础启动测试

```bash
python -m compileall app.py src
streamlit run app.py
```

检查页面是否正常启动，Sidebar 是否显示：

- 学习参数设置；
- 知识库上传；
- RAG 检索参数；
- Prompt / Skill 展示；
- 长期记忆面板。

---

### 7.2 Prompt Registry 测试

在 Sidebar 打开 Prompt Registry，检查：

- `planner`
- `tutor`
- `quiz`
- `grade`
- `qa`
- `reviewer`
- `reflection`
- `react_system`

是否都能展示版本、说明、变量和完整模板。

---

### 7.3 Skill Registry 测试

在 Skill Registry 输入：

```text
我想学习 LangGraph 的短期记忆机制，帮我安排今天的学习计划。
```

预期命中：

```text
个性化学习规划 Skill
```

输入：

```text
请结合知识库讲解 RAG 的召回和 rerank 有什么区别。
```

预期命中：

```text
RAG 导师讲解 Skill
RAG 检索调试 Skill
```

---

### 7.4 Workflow 主流程测试

输入学习目标后运行 Workflow Mode，检查是否完整生成：

- 学习计划；
- 导师讲解；
- 小测验；
- 复盘；
- Reflection 自检结果。

Workflow Mode 不受 Skill 路由控制，仍保持固定学习闭环。

---

### 7.5 ReAct Agent 测试

测试问题：

```text
你现在有哪些 Skill？请列出来，并说明每个 Skill 适合什么时候用。
```

观察：

- ReAct 是否能正常回答；
- 是否显示工具调用轨迹；
- 是否展示本轮命中的 Skill；
- Reflection 是否正常输出。

---

### 7.6 RAG 检索模式测试

分别测试：

- `rough`
- `light_rerank`
- `model_rerank`

观察三种模式下召回片段和排序结果是否正常展示。

---

## 8. 当前版本亮点

1. 实现 Workflow + ReAct 双模式；
2. 使用 LangGraph 构建稳定学习闭环；
3. 支持本地知识库 RAG 检索；
4. 支持三种 RAG 检索模式与调试面板；
5. 支持 ReAct Tool Calling；
6. 支持节点级与全局 Reflection；
7. 支持 Chroma 向量长期记忆；
8. 支持 Memory Reflection 与轻量遗忘机制；
9. 新增 Prompt Registry，实现提示词集中管理和版本化展示；
10. 新增 Skill Registry，实现能力抽象、Prompt / Tool 绑定和前端可解释展示；
11. ReAct system prompt 可注入当前命中的 Skill 信息，辅助工具选择。

---

## 9. 简历描述参考

```text
EduPilot Agent：面向个性化学习场景的 AI Agent 项目。基于 Streamlit + LangChain + LangGraph + Chroma 实现学习目标输入、RAG 知识库检索、学习计划生成、导师讲解、小测验、智能批改、追问答疑、ReAct Tool Calling、Reflection 自检和长期记忆的完整闭环。
```

```text
设计并实现 Prompt Registry，将 Planner、Tutor、Quiz、Grading、QA、Reviewer、Reflection、ReAct System Prompt 等提示词统一注册和版本化管理，提升 Agent 可维护性与调试效率。
```

```text
构建 Skill Registry 能力层，将学习规划、RAG 导师讲解、测验生成、智能批改、追问答疑、长期记忆和 RAG 调试抽象为高层 Skill，并绑定触发规则、Prompt 模板与 Tool 调用链路，用于前端展示和 ReAct system prompt 注入。
```

```text
实现 RAG 三种检索模式 rough / light_rerank / model_rerank，并在 Streamlit 中提供检索调试面板，用于观察粗召回、轻量重排和模型重排预留模式下的检索差异。
```

---

## 10. 后续计划

1. 将 Prompt Registry 从单模板升级为 `system_template + human_template` 双模板结构；
2. 将关键词 Skill Router 升级为规则优先 + LLM 意图分类的混合路由；
3. 为 Skill Router 增加置信度、fallback 和测试集评估；
4. 为 Prompt 版本增加效果评估和变更记录；
5. 为 ReAct 工具调用增加更细粒度的工具权限控制；
6. 优化长期记忆质量，增加记忆去重与重要性评分；
7. 增强 RAG rerank，可接入本地或云端 cross-encoder reranker；
8. 增加 Demo 截图和项目演示视频说明。
