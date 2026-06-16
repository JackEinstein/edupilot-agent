# EduPilot Agent

EduPilot Agent 是一个面向个性化学习场景的 AI 学习助手项目，支持从学习目标输入、知识库检索、学习计划生成、导师讲解、小测验、智能批改、追问答疑，到 ReAct Tool Calling、Reflection 自检和 Chroma 向量长期记忆的完整闭环。

当前版本重点新增了 **RAG 召回优化与三种检索模式**，用于提升知识库检索质量，并在 Streamlit 页面中提供 RAG / Rerank 调试面板，方便观察不同检索策略下的召回结果和重排效果。

---

## 1. 项目定位

本项目主要用于展示一个教育方向 AI Agent 的端到端工程能力，适合作为 AI Agent / RAG / LangGraph / Tool Calling / Memory 方向的简历项目 Demo。

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
11. 支持 RAG 三种检索模式：`rough`、`light_rerank`、`model_rerank`。

---

## 2. 当前版本核心功能

### 2.1 Workflow Mode

Workflow Mode 是固定工作流模式，基于 LangGraph 串联多个学习节点，适合稳定地完成一轮完整学习闭环。

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
Global Workflow Reflection
    ↓
Memory Reflection → Chroma Long-term Memory
    ↓
Final Answer
```

主要输出包括：

* 今日学习计划；
* 导师讲解；
* 小测验；
* 测验批改反馈；
* 复盘验收建议；
* 节点级 Reflection；
* 全局 Workflow Reflection；
* RAG 检索资料；
* 长期记忆召回与写入结果。

---

### 2.2 ReAct Agent Mode

ReAct Agent Mode 是动态工具调用模式。模型会根据用户问题自主判断是否需要调用工具，并结合工具结果生成最终回答。

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

ReAct Agent 生成草稿回答后，可以通过 Reflection 模块进行自检与改写，从而减少回答过短、遗漏重点或工具结果利用不足的问题。

---

### 2.3 RAG 本地知识库

项目支持上传 `.md` 和 `.txt` 学习资料，并构建本地 Chroma 向量知识库。

知识库构建流程：

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
根据用户问题进行语义检索
```

当前 RAG 知识库主要用于辅助：

* Planner 生成学习计划；
* Tutor 生成导师讲解；
* Quiz 生成小测题；
* Follow-up QA 追问答疑；
* ReAct Agent 工具调用回答。

---

### 2.4 RAG 召回优化与三种检索模式

当前版本新增了 RAG 检索模式选择，用于对比不同召回策略的效果。

#### 1. `rough`

基础向量召回模式。

该模式直接使用 Chroma 进行相似度检索，返回与 query 最相似的 top_k 个知识片段。

#### 2. `light_rerank`

轻量级重排模式。

该模式会先扩大粗召回候选数量，然后基于轻量规则对候选片段重新排序：

```text
query
    ↓
Chroma 粗召回 fetch_k 个候选片段
    ↓
计算向量相似度分数
    ↓
计算关键词匹配分数
    ↓
结合来源多样性等轻量规则进行重排
    ↓
返回最终 top_k 个片段
```

这是当前 Day 9 的重点功能。

#### 3. `model_rerank`

模型重排模式。

该模式预留给 CrossEncoder / bge-reranker 等模型重排器使用。如果本地暂未安装或加载 reranker 模型，则可以自动降级到 `light_rerank`，避免主流程报错。

---

### 2.5 RAG / Rerank 调试面板

Streamlit 页面中提供了独立的 RAG 调试面板，用于测试三种检索模式。

调试面板支持：

* 输入独立检索 query；
* 调整 `retrieval_mode`；
* 调整最终返回数量 `top_k`；
* 调整粗召回候选数量 `fetch_k`；
* 查看每个 chunk 的来源、chunk 编号、检索分数和 rerank 分数。

该调试面板不会写入短期记忆或长期记忆，只用于观察检索效果。

---

### 2.6 Reflection 自检机制

当前版本包含两类 Reflection。

Workflow Mode 中包含节点级 Reflection 和全局 Reflection：

* Tutor Reflection：检查导师讲解是否清晰、准确、贴合目标；
* Quiz Reflection：检查小测题是否围绕本轮学习内容；
* Reviewer Reflection：检查复盘反馈是否完整；
* Global Workflow Reflection：对本轮 Workflow 总输出进行整体自检。

ReAct Agent 生成初稿回答后，可以通过 Reflection 模块进行最终审查与改写，尽量减少回答过短、遗漏重点或工具结果利用不足的问题。

---

### 2.7 向量数据库长期记忆

当前版本实现了基于 Chroma 的长期记忆系统。

长期记忆不是保存所有聊天记录，而是保存经过 Memory Reflection 判断后具有长期价值的信息，例如：

* 当前项目进度；
* 学生长期薄弱点；
* 已掌握内容；
* 下一步学习任务；
* 重要的 ReAct Agent 交互结果；
* Workflow 学习闭环总结。

RAG 知识库和长期记忆库是分开的：

```text
data/chroma_db/           # RAG 知识库
data/memory_chroma_db/    # 长期记忆库
```

---

### 2.8 Memory Reflection 与遗忘机制

长期记忆写入前，会先经过 Memory Reflection 判断。

系统不会直接把完整对话写入长期记忆，而是先判断：

```text
这段学习事件是否具有长期保存价值？
如果值得保存，应该总结成什么长期记忆？
如果不值得保存，应该丢弃。
```

长期记忆还支持轻量级遗忘机制。遗忘机制不会直接删除记忆，而是将低频、长期未访问的记忆软归档为 `archived`。

核心字段包括：

* `status`：记忆状态，默认为 `active`；
* `created_at`：创建时间；
* `updated_at`：更新时间；
* `last_accessed_at`：最近访问时间；
* `access_count`：被检索命中的次数。

默认检索只召回 `active` 记忆。

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
* HuggingFace Embeddings
* `sentence-transformers/all-MiniLM-L6-v2`
* RecursiveCharacterTextSplitter

### Memory

* LangGraph Checkpointer / 短期记忆
* Chroma 向量长期记忆
* Memory Reflection
* Forgetting Policy

---

## 4. 项目结构

```text
edupilot-agent/
├── app.py                         # Streamlit 前端入口
├── requirements.txt               # Python 依赖
├── .env.example                   # 环境变量示例
├── data/
│   ├── knowledge/                 # 用户上传的 RAG 学习资料
│   ├── chroma_db/                 # RAG 知识库向量数据库
│   └── memory_chroma_db/          # 长期记忆向量数据库
└── src/
    ├── graph.py                   # LangGraph Workflow 主流程
    ├── llm.py                     # LLM 初始化
    ├── planner.py                 # 学习计划生成
    ├── tutor.py                   # 导师讲解生成
    ├── quiz.py                    # 小测生成与批改
    ├── reviewer.py                # 复盘验收
    ├── qa.py                      # Follow-up QA 追问答疑
    ├── retriever.py               # RAG 知识库构建、检索与 rerank
    ├── tools.py                   # ReAct Agent 工具定义
    ├── react_agent.py             # ReAct Agent 运行逻辑
    ├── reflection.py              # Workflow / ReAct Reflection
    ├── short_term_memory.py       # 短期记忆格式化
    ├── long_term_memory.py        # 长期记忆读写入口
    ├── vector_memory.py           # Chroma 向量长期记忆实现
    └── memory_reflection.py       # Memory Reflection 判断逻辑
```

---

## 5. 环境配置

推荐使用 Python 3.10。

```bash
conda create -n edupilot python=3.10
conda activate edupilot
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

然后在 `.env` 中填写 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

---

## 6. 运行方式

在项目根目录运行：

```bash
streamlit run app.py
```

打开页面后，可以完成以下操作：

1. 上传 `.md` 或 `.txt` 学习资料；
2. 点击重建知识库；
3. 选择运行模式：`Workflow Mode` 或 `ReAct Agent Mode`；
4. 设置 RAG 检索模式：`rough`、`light_rerank`、`model_rerank`；
5. 设置 `top_k` 和 `fetch_k`；
6. 输入学习目标并生成学习方案；
7. 使用 RAG / Rerank 调试面板观察检索效果。

---

## 7. 功能测试建议

### 7.1 基础启动测试

```bash
python -m compileall app.py src
streamlit run app.py
```

预期：页面正常打开，无语法错误。

### 7.2 RAG 知识库测试

上传 Markdown 或 TXT 文件后，点击重建知识库。

推荐测试问题：

```text
LangGraph 的 thread_id 有什么作用？
RAG 和普通大模型问答有什么区别？
```

预期：能够检索出知识库片段。

### 7.3 三种检索模式测试

分别选择：

```text
rough
light_rerank
model_rerank
```

使用同一个 query 测试召回结果。

重点观察：

* `rough` 是否能完成基础向量召回；
* `light_rerank` 是否能返回 rerank 相关分数；
* `model_rerank` 在未接入模型时是否能正常降级，不影响主流程。

### 7.4 Workflow 主流程测试

输入学习目标：

```text
我想学习 LangGraph 的短期记忆和 thread_id
```

推荐参数：

```text
学习水平：Beginner
学习时长：2 小时
retrieval_mode：light_rerank
top_k：4
fetch_k：10
```

预期输出：

* 今日学习计划；
* 导师讲解；
* 小测验；
* 复盘建议；
* RAG 检索资料；
* Reflection 结果；
* 长期记忆写入结果。

---

## 8. 当前版本亮点

* 基于 LangGraph 构建多节点学习工作流；
* 支持 Workflow Mode 和 ReAct Agent Mode 双模式；
* 支持本地 Markdown / TXT 知识库上传与 Chroma 向量检索；
* 支持三种 RAG 检索模式，便于对比粗召回、轻量 rerank 和模型 rerank；
* 支持 RAG / Rerank 可视化调试面板，增强项目可解释性；
* 支持 Quiz 自动生成、学生作答与智能批改；
* 支持 Follow-up QA，围绕本轮学习内容继续答疑；
* 支持 Workflow Reflection 和 ReAct Reflection；
* 支持 Chroma 向量长期记忆、Memory Reflection 和轻量遗忘机制。

---

## 9. 简历描述参考

```text
EduPilot Agent：基于 LangGraph、LangChain、Chroma 和 Streamlit 构建的个性化学习 Agent。项目实现了学习目标输入、RAG 知识库检索、学习计划生成、导师讲解、小测验生成与批改、追问答疑、ReAct Tool Calling、Reflection 自检以及 Chroma 向量长期记忆闭环。进一步优化 RAG 检索链路，设计 rough、light_rerank、model_rerank 三种检索模式，并通过 top_k / fetch_k 参数与 RAG 调试面板展示召回和重排效果，提升系统回答的依据性、可解释性与工程展示能力。
```

---

## 10. 后续计划

* 进一步完善 `model_rerank`，正式接入 bge-reranker 或 CrossEncoder；
* 将 Prompt 模板模块化，统一 Planner / Tutor / Quiz / QA 输出风格；
* 增加 Agent 能力展示与更清晰的工具调用说明；
* 增加 FastAPI 服务化接口，支持外部系统调用；
* 增加评测集与自动化评估脚本，对比不同检索模式下的回答质量。
