# EduPilot Agent

EduPilot Agent 是一个面向个性化学习场景的 AI 学习助手项目，支持从学习目标输入、知识库检索、学习计划生成、导师讲解、小测验、智能批改、追问答疑，到 ReAct Tool Calling、Reflection 自检和向量数据库长期记忆的完整闭环。

本项目主要用于展示一个教育方向 AI Agent 的端到端工程能力，适合作为 AI Agent / RAG / LangGraph / Tool Calling / Memory 方向的简历项目 Demo。

---

## 1. 项目简介

EduPilot Agent 的目标是帮助学生围绕一个学习目标完成一轮结构化学习流程。

用户输入学习目标后，系统可以自动完成：

1. 从本地知识库中检索相关学习资料；
2. 结合学习目标生成今日学习计划；
3. 根据计划和资料生成导师式讲解；
4. 自动生成小测验；
5. 支持学生作答并进行批改反馈；
6. 支持基于本轮学习内容继续追问；
7. 支持 ReAct Agent 自主选择工具；
8. 支持 Workflow Reflection 和 ReAct Reflection；
9. 支持基于 Chroma 的向量数据库长期记忆；
10. 支持长期记忆的轻量级遗忘机制。

---

## 2. 当前版本核心功能

### 2.1 Workflow Mode

Workflow Mode 是固定工作流模式，基于 LangGraph 串联多个节点：

```text
User Input
    ↓
Retriever: RAG + Long-term Memory
    ↓
Planner
    ↓
Tutor + Lightweight Reflection
    ↓
Quiz + Lightweight Reflection
    ↓
Reviewer + Lightweight Reflection
    ↓
Global Workflow Reflection
    ↓
Memory Reflection → Chroma Long-term Memory
    ↓
Final Answer
```

主要输出包括：

* 学习计划；
* 导师讲解；
* 小测验；
* 复盘验收；
* 节点级 Reflection；
* 全局 Workflow Reflection；
* 长期记忆写入结果；
* RAG 与长期记忆召回上下文。

---

### 2.2 ReAct Agent Mode

ReAct Agent Mode 是动态工具调用模式。模型会根据用户问题自动判断是否需要调用工具，并结合工具结果生成最终回答。

当前支持的工具包括：

* `rag_tool`：检索本地知识库；
* `plan_tool`：生成学习计划；
* `tutor_tool`：生成导师讲解；
* `quiz_tool`：生成小测验；
* `grade_quiz_answer_tool`：批改学生答案；
* `qa_tool`：基于当前学习内容追问答疑；
* `review_tool`：生成复盘验收；
* `get_current_context_tool`：读取当前会话上下文；
* `long_term_memory_tool`：检索 Chroma 向量长期记忆。

ReAct Agent 生成草稿回答后，会进一步通过 Reflection 模块进行自检与改写。

---

### 2.3 RAG 本地知识库

项目支持上传 `.md` 和 `.txt` 学习资料，并构建本地 Chroma 向量知识库。

知识库流程：

```text
上传学习资料
    ↓
保存到 data/knowledge/
    ↓
文本切分
    ↓
Embedding
    ↓
写入 data/chroma_db/
    ↓
根据用户目标进行语义检索
```

当前 RAG 知识库主要用于辅助 Planner、Tutor、Quiz、QA 和 ReAct Agent 回答。

---

### 2.4 Reflection 自检机制

当前版本包含两类 Reflection：

#### 1. Workflow Reflection

Workflow Mode 中包含节点级 Reflection 和全局 Reflection：

* Tutor Reflection：检查导师讲解是否清晰、准确、贴合目标；
* Quiz Reflection：检查小测题是否围绕本轮学习内容；
* Reviewer Reflection：检查复盘反馈是否完整；
* Global Workflow Reflection：对本轮 Workflow 总输出进行整体自检。

#### 2. ReAct Reflection

ReAct Agent 生成初稿回答后，会通过 Reflection 模块进行最终审查，尽量减少回答过短、遗漏重点或工具结果利用不足的问题。

---

### 2.5 向量数据库长期记忆

当前版本实现了基于 Chroma 的简化版长期记忆系统。

长期记忆不是保存所有聊天记录，而是保存经过 Memory Reflection 判断后具有长期价值的信息，例如：

* 当前项目进度；
* 学生长期薄弱点；
* 已掌握内容；
* 下一步学习任务；
* 重要的 ReAct Agent 交互结果；
* Workflow 学习闭环总结。

长期记忆保存路径：

```text
data/memory_chroma_db/
```

RAG 知识库和长期记忆库是分开的：

```text
data/chroma_db/           # RAG 知识库
data/memory_chroma_db/    # 长期记忆库
```

这样可以避免外部学习资料和个性化学习画像混在一起。

---

### 2.6 Memory Reflection

长期记忆写入前，会先经过 Memory Reflection 判断。

系统不会直接把完整对话写入长期记忆，而是先让 LLM 判断：

```text
这段学习事件是否具有长期保存价值？
如果值得保存，应该总结成什么长期记忆？
如果不值得保存，应该丢弃。
```

简化版 Memory Reflection 输出包括：

* `should_save`：是否值得保存；
* `memory_type`：记忆类型；
* `summary`：长期记忆摘要；
* `reason`：保存或丢弃原因。

---

### 2.7 长期记忆遗忘机制

当前版本实现了轻量级遗忘机制。

遗忘机制不是直接删除记忆，而是将低频、长期未访问的记忆软归档为 `archived`。

核心字段包括：

* `status`：记忆状态，默认为 `active`；
* `created_at`：创建时间；
* `updated_at`：更新时间；
* `last_accessed_at`：最近访问时间；
* `access_count`：被检索命中的次数。

默认检索只召回 `active` 记忆。

遗忘规则示例：

```text
如果一条长期记忆超过 30 天未被访问，
并且访问次数较低，
则将其状态改为 archived。
```

这样可以避免长期记忆库无限膨胀，同时避免误删重要记忆。

---

## 3. 技术栈

### 前端与交互

* Streamlit

### LLM 与 Agent

* LangChain
* LangGraph
* DeepSeek API
* ReAct Tool Calling

### RAG 与向量数据库

* Chroma
* langchain-chroma
* Sentence Transformers / HuggingFace Embeddings
* Markdown / TXT 本地知识库

### 记忆机制

* LangGraph short-term memory / `thread_id`
* Chroma vector long-term memory
* Memory Reflection
* 轻量级 forgetting policy

### 开发语言

* Python 3.10+

---

## 4. 项目结构

```text
edupilot-agent/
├── app.py                         # Streamlit 主界面
├── requirements.txt               # 项目依赖
├── README.md                      # 项目说明
├── .env.example                   # 环境变量示例
├── .gitignore                     # Git 忽略配置
│
├── data/
│   ├── knowledge/                 # 用户上传的本地知识资料
│   ├── chroma_db/                 # RAG 知识库向量数据库
│   └── memory_chroma_db/          # 长期记忆向量数据库，本地生成，不提交 Git
│
└── src/
    ├── llm.py                     # LLM 初始化
    ├── graph.py                   # LangGraph Workflow 主流程
    ├── planner.py                 # 学习计划生成
    ├── tutor.py                   # 导师讲解生成
    ├── quiz.py                    # 小测生成与批改
    ├── reviewer.py                # 复盘验收
    ├── qa.py                      # Follow-up QA 追问答疑
    ├── retriever.py               # RAG 知识库构建与检索
    ├── history.py                 # 短期历史消息格式化
    ├── reflection.py              # Workflow / ReAct Reflection
    ├── tools.py                   # ReAct Agent 工具集合
    ├── react_agent.py             # ReAct Agent 调用入口
    ├── long_term_memory.py        # 长期记忆统一入口
    ├── memory_reflection.py       # Memory Reflection 判断是否写入长期记忆
    └── vector_memory.py           # Chroma 向量长期记忆存取与遗忘机制
```

---

## 5. 安装与运行

### 5.1 克隆项目

```bash
git clone <your-repo-url>
cd edupilot-agent
```

### 5.2 创建环境

推荐使用 Conda：

```bash
conda create -n edupilot python=3.10
conda activate edupilot
```

### 5.3 安装依赖

```bash
pip install -r requirements.txt
```

### 5.4 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

然后在 `.env` 中配置你的模型 API Key，例如：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

实际字段名以 `src/llm.py` 中读取方式为准。

### 5.5 启动项目

```bash
streamlit run app.py
```

---

## 6. 使用方式

### 6.1 Workflow Mode

1. 在侧边栏选择学生基础和学习时长；
2. 选择 `Workflow Mode`；
3. 输入学习目标；
4. 点击“生成学习方案”；
5. 查看学习计划、导师讲解、小测验、复盘、Reflection 和长期记忆结果。

示例学习目标：

```text
EduPilot Agent Day 8：实现简化版 Chroma 向量长期记忆，理解长期记忆、语义检索、Memory Reflection 和遗忘机制。
```

---

### 6.2 ReAct Agent Mode

1. 切换到 `ReAct Agent Mode`；
2. 输入自由问题；
3. Agent 会自动判断是否调用工具；
4. 展开工具调用过程可以查看 tool call 与 tool result；
5. 展开 Reflection 可以查看最终回答的自检过程。

示例问题：

```text
老师，请根据我的长期记忆，总结我目前 EduPilot Agent 项目的进度和下一步任务。
```

---

### 6.3 上传知识库资料

侧边栏支持上传 `.md` 和 `.txt` 文件。

上传后点击：

```text
保存上传资料
```

然后点击：

```text
重新构建知识库
```

系统会将资料切分、向量化，并写入本地 Chroma 知识库。

---

### 6.4 长期记忆管理

侧边栏提供长期记忆管理功能：

* 查看有效记忆数量；
* 查看已归档记忆数量；
* 查看总记忆数量；
* 手动执行一次遗忘检查；
* 确认后清空长期记忆。

长期记忆数据保存在本地：

```text
data/memory_chroma_db/
```

该目录不建议提交到 Git。

---

## 7. 短期记忆与长期记忆的区别

EduPilot 同时使用短期记忆和长期记忆。

### 短期记忆

短期记忆主要用于当前会话上下文，例如：

* 当前用户刚刚问了什么；
* 当前 Workflow 生成了什么学习计划；
* 当前小测题和批改反馈；
* 当前 ReAct 工具调用轨迹。

当前版本主要通过 `thread_id`、`messages` 和 `st.session_state` 管理短期上下文。

### 长期记忆

长期记忆用于跨会话保存学习画像，例如：

* 之前学到了哪里；
* 哪些知识点薄弱；
* 项目做到哪一步；
* 下次应该继续做什么。

当前版本使用 Chroma 向量数据库保存长期记忆，并通过语义检索召回。

两者不是替代关系，而是互补关系：

```text
短期记忆负责当前上下文连续性；
长期记忆负责跨会话个性化。
```

---

## 8. 当前版本测试方式

### 8.1 语法检查

```bash
python -m compileall app.py src
```

### 8.2 启动测试

```bash
streamlit run app.py
```

### 8.3 Workflow 长期记忆写入测试

1. 清空长期记忆；
2. 在 Workflow Mode 输入明确学习目标；
3. 点击生成学习方案；
4. 查看长期记忆数量是否增加；
5. 查看“长期记忆”Tab 中的写入结果。

### 8.4 ReAct 长期记忆召回测试

在 ReAct Agent Mode 输入：

```text
老师，请根据我的长期记忆，总结我目前 EduPilot Agent 项目的进度和下一步任务。
```

检查：

* 是否调用 `long_term_memory_tool`；
* 回答是否结合了历史项目进度；
* 工具调用轨迹是否正常展示。

### 8.5 遗忘机制测试

默认遗忘策略较保守，通常不会立刻归档当天新建记忆。

如需测试，可临时将遗忘参数改小，例如：

```python
apply_forgetting_policy(
    max_idle_days=0,
    max_access_count=10,
)
```

点击侧边栏“执行一次遗忘检查”，观察归档数量是否变化。

测试完成后建议恢复默认参数：

```python
apply_forgetting_policy(
    max_idle_days=30,
    max_access_count=1,
)
```

---

## 9. Git 注意事项

以下内容不建议提交：

```text
.env
data/chroma_db/
data/memory_chroma_db/
__pycache__/
.DS_Store
```

建议 `.gitignore` 中包含：

```gitignore
.env
__pycache__/
*.pyc
.DS_Store

data/chroma_db/
data/memory_chroma_db/
```

提交代码前建议运行：

```bash
python -m compileall app.py src
git status
```

---

## 10. 当前项目亮点

当前版本已经覆盖一个教育类 AI Agent Demo 的主要能力：

1. 基于 LangGraph 的多节点 Workflow 编排；
2. 基于 Chroma 的本地 RAG 知识库；
3. Planner / Tutor / Quiz / Reviewer 学习闭环；
4. Follow-up QA 追问答疑；
5. ReAct Tool Calling Agent；
6. 工具调用轨迹可视化；
7. Workflow Reflection 和 ReAct Reflection；
8. 基于 Chroma 的向量长期记忆；
9. Memory Reflection 判断是否写入长期记忆；
10. 基于访问时间和访问次数的轻量级遗忘机制；
11. Streamlit 可交互 Demo 页面。

---

## 11. 后续优化方向

后续可以继续扩展：

1. RAG 召回优化：增加多 query 检索、上下文压缩和 rerank；
2. Prompt 工程展示：将 Planner / Tutor / Quiz / Memory Prompt 模块化；
3. Skill 展示：将学习规划、错题分析、复盘总结抽象为可复用 skill；
4. 长期记忆合并：支持相似记忆合并和旧记忆更新；
5. 记忆可信度：加入 importance / confidence 评分；
6. FastAPI 封装：将 Agent 能力包装成 RESTful API；
7. 前后端分离：使用 React / Vue 作为前端，FastAPI 作为后端；
8. 评测体系：增加 Harness / LangSmith 风格的测试用例与效果评估；
9. 多用户隔离：增加用户登录和用户级长期记忆隔离。

---

## 12. 简历描述示例

可以在简历中描述为：

```text
EduPilot Agent：基于 LangGraph + RAG + ReAct Tool Calling 的个性化学习助手。项目使用 Streamlit 构建交互界面，基于 LangGraph 编排 Retriever、Planner、Tutor、Quiz、Reviewer 和 Reflection 多节点学习闭环；使用 Chroma 构建本地知识库与向量长期记忆库，支持学习资料检索、跨会话学习画像召回、Memory Reflection 写入判断和轻量级遗忘机制；同时封装 RAG、Planner、Tutor、Quiz、Grading、QA、Reviewer、Long-term Memory 等工具，实现 ReAct Agent 动态工具调用与工具轨迹展示。
```

---

## 13. 项目状态

当前版本已完成：

```text
Day 1：项目初始化、Streamlit MVP、LLM 接入、LangGraph 单节点
Day 2：Planner → Tutor → Reviewer 基础 Workflow
Day 3：RAG 本地知识库
Day 4：短期记忆与会话管理
Day 5：Quiz、Grading、Follow-up QA
Day 6：ReAct Tool Calling Agent
Day 7：Workflow Reflection 与 ReAct Reflection
Day 8：Chroma 向量长期记忆、Memory Reflection 与轻量级遗忘机制
```

当前项目已具备一个教育方向 AI Agent 的完整 Demo 闭环。



