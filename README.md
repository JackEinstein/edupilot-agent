# EduPilot Agent

EduPilot Agent 是一个面向个性化学习的 AI 学习助手项目，基于 LangGraph、LangChain、RAG 和 ReAct Tool Calling 构建。项目支持根据学生的学习目标、当前基础和可用时间，自动生成学习计划、导师讲解、小测验、测验批改、追问答疑和复盘验收，并进一步支持 Workflow Mode 与 ReAct Agent Mode 两种运行模式。

本项目用于展示一个教育场景下的完整 Agent 闭环：从资料检索、计划生成、内容讲解、测验评估到动态工具调用答疑，形成较完整的个性化学习流程。

---

## 1. 项目目标

EduPilot Agent 的目标是构建一个能够辅助学生完成每日学习任务的智能导师系统。

核心目标包括：

1. 根据学习目标生成结构化学习计划；
2. 结合本地知识库进行 RAG 检索；
3. 基于学习计划和检索资料生成导师式讲解；
4. 自动生成小测验并支持答案批改；
5. 支持学生对当前学习内容继续追问答疑；
6. 通过 Reviewer 模块生成复盘问题与验收标准；
7. 将 RAG、Planner、Tutor、Quiz、Grading、QA、Reviewer 等核心能力封装为 tools；
8. 基于 ReAct-style Tool Calling Agent 实现动态工具选择；
9. 支持固定 Workflow 与动态 ReAct Agent 双模式运行。

---

## 2. 当前功能

### 2.1 Workflow Mode

Workflow Mode 是基于 LangGraph 的固定学习工作流。

执行流程如下：

```text
User Input
    ↓
Retriever
    ↓
Planner
    ↓
Tutor
    ↓
Quiz
    ↓
Reviewer
```

该模式适合一键生成完整学习闭环，输出内容包括：

* 今日学习计划；
* RAG 检索资料；
* 导师讲解；
* 本轮小测验；
* Quiz 批改反馈；
* Follow-up QA 追问答疑；
* 复盘问题与验收标准；
* 原始 LangGraph State 数据。

---

### 2.2 ReAct Agent Mode

ReAct Agent Mode 是基于 LangChain `create_agent` 构建的动态工具调用模式。

执行流程如下：

```text
User Question
    ↓
LLM decides whether to use tools
    ↓
Tool Call
    ↓
Tool Result
    ↓
Final Answer
```

该模式会让模型根据用户问题自主判断是否调用工具，以及调用哪个工具。当前已封装的工具包括：

* `rag_tool`：封装本地 RAG 知识库检索；
* `plan_tool`：封装 Planner 学习计划生成；
* `tutor_tool`：封装 Tutor 导师讲解生成；
* `quiz_tool`：封装 Quiz 小测验生成；
* `grade_quiz_answer_tool`：封装 Quiz 答案批改；
* `qa_tool`：封装 Follow-up QA 追问答疑；
* `review_tool`：封装 Reviewer 复盘验收；
* `get_current_context_tool`：读取当前学习会话上下文。

ReAct Agent Mode 可以在未运行 Workflow Mode 的情况下独立调用工具，也可以在 Workflow Mode 已生成学习上下文后，复用已有学习计划、讲解、小测验、复盘和 RAG 资料，使回答更贴合当前学习内容。

---

## 3. 项目结构

```text
edupilot-agent/
├── app.py                  # Streamlit 前端入口，支持 Workflow / ReAct 双模式
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明文档
├── data/
│   └── knowledge/          # 本地知识库资料目录
├── src/
│   ├── graph.py            # LangGraph 固定工作流编排
│   ├── llm.py              # 大模型初始化
│   ├── retriever.py        # RAG 文档加载、切分、向量库构建和检索
│   ├── planner.py          # 学习计划生成模块
│   ├── tutor.py            # 导师讲解生成模块
│   ├── quiz.py             # 小测验生成与答案批改模块
│   ├── qa.py               # Follow-up QA 追问答疑模块
│   ├── reviewer.py         # 复盘问题与验收标准生成模块
│   ├── history.py          # QA / ReAct 历史记录格式化模块
│   ├── tools.py            # ReAct Agent 工具封装层
│   └── react_agent.py      # ReAct-style Tool Calling Agent 构建与运行
```

---

## 4. 技术栈

* Python
* Streamlit
* LangChain
* LangGraph
* Chroma
* RAG
* ReAct-style Tool Calling
* DeepSeek / OpenAI compatible LLM API

---

## 5. 核心模块说明

### 5.1 `src/graph.py`

`graph.py` 负责固定 Workflow 的编排，使用 LangGraph 串联多个节点。

主要节点包括：

* Retriever Node；
* Planner Node；
* Tutor Node；
* Quiz Node；
* Reviewer Node。

该模式强调流程稳定性，适合生成完整学习方案。

---

### 5.2 `src/tools.py`

`tools.py` 是 ReAct Agent 的工具封装层。

它不会重新实现业务逻辑，而是把已有模块包装成 LangChain tools：

```text
retriever.py  → rag_tool
planner.py    → plan_tool
tutor.py      → tutor_tool
quiz.py       → quiz_tool / grade_quiz_answer_tool
qa.py         → qa_tool
reviewer.py   → review_tool
```

这种设计避免了业务逻辑重复，使 Workflow Mode 和 ReAct Agent Mode 可以复用同一套底层能力。

---

### 5.3 `src/react_agent.py`

`react_agent.py` 负责构建和运行 ReAct-style Tool Calling Agent。

主要功能包括：

1. 初始化 LLM；
2. 加载 `tools.py` 中定义的工具；
3. 使用 `create_agent` 构建 Tool Calling Agent；
4. 使用 `thread_id` 和 checkpointer 管理短期会话；
5. 提取工具调用轨迹；
6. 返回最终回答和工具调用 trace。

---

### 5.4 `src/history.py`

`history.py` 负责历史记录格式化。

当前主要用于：

* 格式化 Follow-up QA 历史；
* 格式化 ReAct Agent 历史；
* 将历史记录整理为 LLM 更容易理解的文本。

---

## 6. ReAct Tool Calling 设计

本项目中的 ReAct Agent 体现为：

```text
Reason：模型根据用户问题判断是否需要工具；
Act：模型发起 tool call；
Observe：工具返回执行结果；
Answer：模型基于工具结果生成最终回答。
```

一次典型调用过程如下：

```text
用户：请帮我生成一套 Tool Calling 小测验。

AIMessage：
模型决定调用 quiz_tool，参数为 {"topic": "Tool Calling"}。

ToolMessage：
quiz_tool 返回生成的小测验。

AIMessage：
模型根据工具结果整理最终回答并返回给用户。
```

前端会展示工具调用轨迹，包括：

* 调用了哪个工具；
* 工具参数是什么；
* 工具返回了什么结果。

这增强了 Agent 的可解释性，也便于调试和项目展示。

---

## 7. 安装与运行

### 7.1 创建环境

```bash
conda create -n edupilot python=3.10
conda activate edupilot
```

### 7.2 安装依赖

```bash
pip install -r requirements.txt
```

### 7.3 配置环境变量

根据 `src/llm.py` 中使用的模型服务配置 API Key。

如果使用 DeepSeek，可参考：

```bash
export DEEPSEEK_API_KEY="your_api_key"
```

Windows PowerShell 可使用：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
```

如果你使用的是其他 OpenAI-compatible API，请根据 `src/llm.py` 中的配置修改对应环境变量。

### 7.4 启动项目

```bash
streamlit run app.py
```

---

## 8. 使用方式

### 8.1 Workflow Mode

1. 在侧边栏选择学生基础和今日学习时间；
2. 选择 `Workflow Mode`；
3. 输入学习目标；
4. 点击“生成学习方案”；
5. 查看学习计划、导师讲解、小测验、复盘验收和 RAG 检索资料；
6. 在“追问答疑”中继续提问；
7. 在“小测验”中提交答案并获取批改反馈。

---

### 8.2 ReAct Agent Mode

1. 在侧边栏选择 `ReAct Agent Mode`；
2. 输入想让 Agent 处理的问题；
3. Agent 会根据问题自动选择是否调用工具；
4. 展开“查看本轮工具调用过程”，查看工具调用 trace。

示例问题：

```text
老师，我今天想学习 ReAct Tool Calling，请你直接给我一份学习计划和小测验。
```

```text
老师，请解释 Workflow Mode 和 ReAct Agent Mode 的区别。
```

```text
老师，请检索知识库并解释 LangGraph 的 checkpointer 和 thread_id 是什么关系。
```

```text
请批改我的答案：Tool Calling 是模型根据问题调用外部函数，ReAct 是 Reason + Act。
```

---

## 9. 当前版本亮点

1. 实现了从学习目标到学习计划、导师讲解、小测验、批改和复盘的完整学习闭环；
2. 使用 LangGraph 构建固定 Workflow，保证学习流程稳定输出；
3. 使用 Chroma 构建本地 RAG 知识库，支持上传和重建资料；
4. 新增 Follow-up QA 模块，支持学生基于当前学习内容继续追问；
5. 新增 ReAct Agent Mode，将多个业务模块封装为 tools；
6. 支持模型根据用户问题动态选择工具；
7. 支持展示工具调用 trace，提高 Agent 可解释性；
8. 支持 Workflow Mode 与 ReAct Agent Mode 双模式运行；
9. 通过 `thread_id` 管理不同学习会话，支持短期会话隔离。

---

## 10. 与普通聊天机器人的区别

普通聊天机器人通常是：

```text
用户问题 → LLM 直接回答
```

EduPilot Agent 的 Workflow Mode 是：

```text
用户目标 → 固定学习工作流 → 学习计划 / 讲解 / 测验 / 复盘
```

EduPilot Agent 的 ReAct Agent Mode 是：

```text
用户问题 → 模型判断是否需要工具 → 调用工具 → 工具返回结果 → 模型生成最终回答
```

因此，本项目不仅是一个问答应用，而是一个具有工作流编排、RAG 检索、工具调用和学习闭环能力的教育场景 Agent Demo。

---

## 11. 后续优化方向

后续可以继续扩展：

1. 引入长期记忆模块，保存学生画像、学习偏好和历史薄弱点；
2. 增加 Reflection 模块，让 Agent 对回答和学习效果进行自我反思；
3. 增加学习进度追踪和错题记录；
4. 将 Chroma 本地向量库替换为可持久化服务；
5. 支持更多文件类型，如 PDF、DOCX；
6. 增加用户登录和多用户学习空间；
7. 优化工具选择策略，减少不必要的 tool call；
8. 增加 LangSmith 或日志模块，用于观察 Agent 运行过程。

