# EduPilot Agent

EduPilot Agent 是一个面向个性化学习场景的教育智能体 Demo。项目基于 **LangGraph + LangChain + RAG + Tool Calling + Reflection** 构建，目标是模拟一个 AI 学习导师，帮助用户围绕某个学习目标完成“资料检索、学习规划、导师讲解、小测验、复盘验收、追问答疑、自检改写”的完整学习闭环。

本项目不仅支持固定的 LangGraph Workflow 学习流程，也支持 ReAct Agent Mode，让模型根据用户问题自主决定是否调用工具，并在最终回答后通过 Reflection 机制进行自检和改写。

---

## 1. 项目目标

EduPilot Agent 的核心目标是构建一个可运行、可展示、可扩展的教育类 Agent 项目，用于体现以下能力：

1. 基于用户目标生成个性化学习计划；
2. 从本地知识库中检索相关学习资料；
3. 结合 RAG 资料进行导师式讲解；
4. 自动生成小测验；
5. 支持学生作答并生成批改反馈；
6. 支持学生围绕本轮学习内容继续追问；
7. 支持 ReAct Tool Calling Agent 动态调用工具；
8. 支持 Workflow Mode 与 ReAct Agent Mode 双运行模式；
9. 支持 Reflection 自检改写，提高 Agent 输出质量；
10. 支持 thread_id 会话隔离，实现短期记忆管理。

---

## 2. 当前功能

### 2.1 Workflow Mode：固定学习闭环

Workflow Mode 使用 LangGraph 构建固定多节点工作流，适合完整学习任务。

当前流程如下：

```text
User Input
    ↓
Retriever Node
    ↓
Planner Node
    ↓
Tutor Node + Node-level Reflection
    ↓
Quiz Node + Node-level Reflection
    ↓
Reviewer Node + Node-level Reflection
    ↓
Workflow Reflection Node
    ↓
Final Answer
```

各节点职责如下：

| 节点                  | 功能                     |
| ------------------- | ---------------------- |
| Retriever           | 从本地 Chroma 向量库检索相关学习资料 |
| Planner             | 根据学习目标、基础水平和可用时间生成学习计划 |
| Tutor               | 基于学习计划和资料进行导师式讲解       |
| Quiz                | 根据讲解内容生成小测验            |
| Reviewer            | 生成复盘内容和验收标准            |
| Workflow Reflection | 检查完整学习闭环的一致性和可执行性      |

---

### 2.2 ReAct Agent Mode：动态工具调用

ReAct Agent Mode 让模型根据用户问题自主判断是否需要调用工具。

流程如下：

```text
User Question
    ↓
LLM decides whether to use tools
    ↓
Tool Call / Tool Result
    ↓
Draft Answer
    ↓
Reflection Review
    ↓
Final Answer
```

当前可用 tools 包括：

| Tool                     | 功能              |
| ------------------------ | --------------- |
| rag_tool                 | 调用 RAG 检索知识库资料  |
| plan_tool                | 生成学习计划          |
| tutor_tool               | 生成导师讲解          |
| quiz_tool                | 生成小测验           |
| grade_quiz_answer_tool   | 批改学生答案          |
| qa_tool                  | 基于当前学习上下文进行追问答疑 |
| review_tool              | 生成复盘和验收内容       |
| get_current_context_tool | 读取当前学习会话上下文     |

相比 Workflow Mode，ReAct Agent Mode 更适合自由问答、临时任务和动态工具调用场景。

---

### 2.3 RAG 本地知识库

项目支持上传 `.md` 和 `.txt` 学习资料，并将其保存到本地知识库目录。

支持功能：

1. 上传 Markdown / txt 学习资料；
2. 保存到 `data/knowledge/`；
3. 使用文本切分器切分文档；
4. 构建 Chroma 向量数据库；
5. 根据用户问题检索相关 chunk；
6. 将检索结果提供给 Planner、Tutor、QA 和 ReAct Agent 使用。

RAG 的作用是让 EduPilot Agent 不只依赖大模型自身知识，而是能够结合用户上传的学习资料生成更贴合项目的回答。

---

### 2.4 Reflection 自检改写

Day 7 新增 Reflection 机制，使 Agent 能够对自身输出进行质量审查和改写。

当前 Reflection 分为两类：

#### 1. Workflow Reflection

Workflow Mode 中采用轻量级节点 Reflection + 全局 Reflection。

节点级 Reflection 目前覆盖：

| 节点       | Reflection 检查重点          |
| -------- | ------------------------ |
| Tutor    | 是否承接学习计划、适合学生水平、讲清核心知识点  |
| Quiz     | 是否基于导师讲解出题、难度是否合适、是否覆盖重点 |
| Reviewer | 是否总结本轮学习重点、是否给出可执行验收标准   |

全局 Workflow Reflection 检查：

1. 学习计划、导师讲解、小测验、复盘是否前后一致；
2. 是否符合用户目标、基础水平和学习时长；
3. 是否存在重复、空泛、脱节或遗漏；
4. 是否有明确学习任务、输出结果和验收标准；
5. 最终输出是否适合展示给用户。

#### 2. ReAct Reflection

ReAct Agent Mode 中，Reflection 不作为普通 tool，而是作为最终回答后的 critic-rewrite loop。

也就是说：

```text
ReAct Agent 先动态调用工具并生成草稿
    ↓
Reflection 检查草稿和工具调用结果
    ↓
Rewrite 生成最终回答
```

这种设计避免了让模型自己决定是否调用 Reflection tool，从而保证每轮回答都经过稳定的质量控制。

---

### 2.5 会话记忆

项目通过 `thread_id` 管理不同学习会话，用于隔离 Workflow / ReAct Agent 的短期记忆。

每次开启新学习会话时，会重置：

1. 当前 thread_id；
2. 最近一次 Workflow 结果；
3. Quiz 批改反馈；
4. Follow-up QA 历史；
5. ReAct Agent 对话历史。

---

## 3. 技术栈

| 模块           | 技术                             |
| ------------ | ------------------------------ |
| 前端           | Streamlit                      |
| Agent 工作流    | LangGraph                      |
| LLM 调用       | LangChain Chat Model           |
| RAG          | Chroma 向量库                     |
| 文档处理         | Markdown / txt 文档读取与切分         |
| Tool Calling | LangChain tools / create_agent |
| Reflection   | Critic-Rewrite Prompt          |
| 会话管理         | thread_id + session_state      |
| 版本管理         | Git                            |

---

## 4. 项目结构

```text
edupilot-agent/
├── app.py                      # Streamlit 前端入口
├── README.md                   # 项目说明文档
├── requirements.txt            # 项目依赖
├── data/
│   ├── knowledge/              # 本地知识库原始资料
│   └── chroma/                 # Chroma 向量数据库
└── src/
    ├── graph.py                # LangGraph Workflow 主流程
    ├── react_agent.py          # ReAct Tool Calling Agent
    ├── tools.py                # ReAct Agent 可调用工具
    ├── retriever.py            # RAG 检索和向量库构建
    ├── planner.py              # 学习计划生成模块
    ├── tutor.py                # 导师讲解模块
    ├── quiz.py                 # 小测验生成与批改模块
    ├── qa.py                   # 追问答疑模块
    ├── reviewer.py             # 复盘验收模块
    ├── reflection.py           # Reflection 自检改写模块
    ├── history.py              # 历史消息格式化
    └── llm.py                  # LLM 初始化与配置
```

---

## 5. 安装与运行

### 5.1 克隆项目

```bash
git clone <your-repo-url>
cd edupilot-agent
```

### 5.2 创建虚拟环境

```bash
conda create -n edupilot python=3.10
conda activate edupilot
```

或者使用 venv：

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```bash
.venv\Scripts\activate
```

### 5.3 安装依赖

```bash
pip install -r requirements.txt
```

### 5.4 配置环境变量

在项目根目录创建 `.env` 文件，并填写模型 API Key。

示例：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

具体变量名以 `src/llm.py` 中的配置为准。

### 5.5 启动项目

```bash
streamlit run app.py
```

启动后，在浏览器中打开 Streamlit 页面即可使用。

---

## 6. 使用方式

### 6.1 Workflow Mode

适合完整学习闭环任务。

使用步骤：

1. 在侧边栏选择基础水平；
2. 设置今日可学习时间；
3. 选择 `Workflow Mode`；
4. 输入学习目标；
5. 点击“生成学习方案”。

示例输入：

```text
今天是 EduPilot Agent Day 7。Day 6 已完成 ReAct Tool Calling Agent 双模式，Git 已提交。今天计划学习轻量级 Workflow Reflection。
```

输出内容包括：

1. 今日学习计划；
2. 导师讲解；
3. 追问答疑入口；
4. 本轮小测验；
5. 复盘与验收；
6. Reflection 自检过程；
7. RAG 检索资料。

---

### 6.2 ReAct Agent Mode

适合自由问答和动态工具调用任务。

使用步骤：

1. 在侧边栏选择 `ReAct Agent Mode`；
2. 输入问题；
3. Agent 会自动判断是否调用工具；
4. 展示最终回答、工具调用过程和 Reflection 自检意见。

示例输入：

```text
Day 6 已完成 ReAct Tool Calling Agent 双模式，今天 Day 7 要做 Reflection。请告诉我今天应该怎么验收。
```

页面会展示：

1. ReAct 最终回答；
2. ReAct 草稿回答；
3. Reflection 审查意见；
4. 工具调用 trace。

---

## 7. Reflection 设计说明

本项目没有将 Reflection 简单封装成普通 tool，而是将其设计为 Agent 内部质量控制流程。

原因如下：

1. tool calling 是由模型自主决定是否调用的，可能出现漏调；
2. Reflection 是每轮回答后的质量控制，不应该依赖模型自己决定是否执行；
3. Workflow Mode 中，Reflection 更适合作为节点级检查和全局检查；
4. ReAct Mode 中，Reflection 更适合作为最终回答后的 critic-rewrite loop。

因此，本项目采用以下设计：

```text
Workflow Mode:
Node Output Draft
    ↓
Node-level Reflection
    ↓
Improved Node Output
    ↓
Global Workflow Reflection
    ↓
Final Answer

ReAct Agent Mode:
Tool Calling Draft Answer
    ↓
Reflection Review
    ↓
Final Answer
```

---

## 8. 开发进度

### Day 1 - 项目初始化

* 搭建基础项目结构；
* 引入 Streamlit 前端；
* 构建最初的 LangGraph Workflow；
* 实现基础 Planner / Tutor 节点。

### Day 2 - RAG 知识库

* 支持上传 Markdown / txt 学习资料；
* 构建本地 Chroma 向量库；
* 实现 Retriever 节点；
* 将 RAG 资料接入学习计划和导师讲解。

### Day 3 - 闭环完善

* 完善 Planner、Tutor、Reviewer；
* 优化前端展示；
* 补充 README 和项目说明。

### Day 4 - 记忆管理

* 引入 thread_id；
* 支持学习会话隔离；
* 保存 QA 历史和 Workflow 结果。

### Day 5 - Quiz 与 QA

* 新增 Quiz 生成；
* 新增 Quiz 批改；
* 新增 Follow-up QA；
* 完善学习闭环。

### Day 6 - ReAct Tool Calling Agent

* 将 RAG / Planner / Tutor / Quiz / Grading / QA / Reviewer 封装成 tools；
* 新增 ReAct Agent Mode；
* 支持模型自主决定是否调用工具；
* 前端展示工具调用轨迹。

### Day 7 - Reflection 自检改写

* 新增轻量级 Workflow Reflection；
* 支持 Tutor / Quiz / Reviewer 节点级自检；
* 新增 Global Workflow Reflection；
* 新增 ReAct Agent 最终回答后的 Reflection 改写；
* 前端展示草稿、自检意见、最终输出和工具调用过程；
* 修复 Reflection 字段和返回值不一致问题。

---

## 9. 测试命令

语法检查：

```bash
python -m compileall app.py src
```

启动前端：

```bash
streamlit run app.py
```

Git 状态检查：

```bash
git status
```

---

## 10. 当前项目亮点

1. **完整教育 Agent 闭环**
   覆盖学习计划、资料检索、导师讲解、小测验、批改、复盘和追问答疑。

2. **LangGraph 多节点工作流**
   使用节点拆分复杂任务，使流程清晰、状态可控、易于调试。

3. **RAG 本地知识库**
   支持上传学习资料并构建向量库，使回答能够结合用户资料。

4. **ReAct Tool Calling Agent**
   模型可以自主决定是否调用工具，而不是固定执行流程。

5. **Reflection 自检改写**
   Workflow 和 ReAct 两种模式都支持自检与改写，提高输出质量。

6. **双模式架构**
   Workflow Mode 适合完整学习闭环，ReAct Agent Mode 适合自由问答和动态任务。

7. **前端可视化展示**
   Streamlit 页面支持展示学习结果、工具调用轨迹、Reflection 草稿和自检意见。

---

## 11. 后续优化方向

后续可以继续优化：

1. 增加长期记忆 Store；
2. 支持用户画像和学习偏好保存；
3. 增加更多学习工具，例如错题本、知识点追踪、学习进度统计；
4. 引入更细粒度的 Agent 评估指标；
5. 增加项目架构图和 Demo 截图；
6. 优化 Streamlit 页面布局；
7. 将 Reflection 结果结构化为 JSON，便于前端展示和统计；
8. 将 ReAct Agent 重构为更细粒度的自定义 LangGraph Agent。

---


