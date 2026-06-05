# EduPilot Agent

## 项目简介

EduPilot Agent 是一个基于 LangGraph、RAG 和大语言模型构建的教育智能体（Educational Agent）。

项目旨在模拟一位智能学习导师，围绕用户提出的学习主题，自动完成：

- 学习资料检索
- 学习路径规划
- 导师式知识讲解
- 学习复盘与验收

与传统聊天机器人不同，EduPilot Agent 并非直接生成答案，而是通过多节点工作流组织学习过程，帮助用户完成从知识获取到知识掌握的完整学习闭环。

---

## 项目目标

构建一个具备以下能力的教育 Agent：

- 基于本地知识库进行 RAG 检索
- 自动制定学习计划
- 导师式知识讲解
- 学习效果验收
- 支持用户上传个性化学习资料
- 支持知识库动态更新

---

## 系统架构

```text
User
 │
 ▼
Streamlit UI
 │
 ▼
LangGraph Workflow
 │
 ├── Retriever
 │       │
 │       ▼
 │   Chroma Vector DB
 │
 ├── Planner
 │
 ├── Tutor
 │
 └── Reviewer
 │
 ▼
Learning Result

```

---

## 工作流程

### 1. 用户输入学习问题

例如：

```text
我想学习 LangGraph 的多节点工作流

```

### 2. Retriever 检索知识库

从本地知识库中检索最相关的学习资料：

```text
data/knowledge/

```

### 3. Planner 制定学习计划

根据用户目标与检索资料生成学习路径。

例如：

```text
第一部分：LangGraph 基础概念
第二部分：StateGraph
第三部分：节点与边
第四部分：Memory

```

### 4. Tutor 导师讲解

基于检索资料进行结构化讲解。

### 5. Reviewer 复盘验收

生成复习建议和掌握情况检查问题。

---

## 技术栈

### LLM

- DeepSeek Chat

### Agent Framework

- LangGraph

### RAG

- LangChain
- Chroma
- HuggingFace Embeddings

### Frontend

- Streamlit

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2

```

### Knowledge Base

```text
Markdown (.md)
Text (.txt)

```

---

## 项目结构

```text
edupilot-agent/
│
├── app.py
│
├── data/
│   ├── knowledge/
│   └── chroma_db/
│
├── src/
│   ├── graph.py
│   ├── planner.py
│   ├── tutor.py
│   └── retriever.py
│
├── requirements.txt
│
└── README.md

```

---

## RAG 知识库功能

### 上传学习资料

支持上传：

```text
.md
.txt

```

文件将自动保存到：

```text
data/knowledge/

```

### 重建知识库

上传资料后可点击：

```text
重新构建知识库

```

系统会：

```text
读取文档
↓
文本切分
↓
Embedding 向量化
↓
写入 Chroma

```

### 检索展示

支持展示：

- 来源文件
- Chunk 内容
- 检索距离分数

方便观察 RAG 检索效果。

---

## 运行项目

### 1. 克隆项目

```bash
git clone <your_repo_url>
cd edupilot-agent

```

### 2. 创建环境

```bash
conda create -n edupilot python=3.10
conda activate edupilot

```

### 3. 安装依赖

```bash
pip install -r requirements.txt

```

### 4. 配置环境变量

创建：

```text
.env

```

配置：

```env
API_KEY=your_api_key
BASE_URL=your_api_base

```

### 5. 启动项目

```bash
streamlit run app.py

```

---

## 当前功能

### 已完成

- Streamlit 前端
- DeepSeek 接入
- LangGraph 工作流
- Planner 节点
- Tutor 节点
- Reviewer 节点
- Chroma 向量数据库
- Markdown / TXT 知识库
- 文件上传
- 知识库重建
- RAG 检索展示

### 开发中

- Quiz Node
- 学习测验
- LangGraph Checkpoint Memory
- 学习画像
- 长期记忆
- 多知识库管理
- Rerank 检索优化

---

## 项目亮点

### 1. 多节点 Agent 工作流

基于 LangGraph 实现：

```text
Retriever
→ Planner
→ Tutor
→ Reviewer

```

将学习任务拆解为多个协作节点。

### 2. 本地 RAG 知识库

支持用户上传资料并动态构建向量数据库。

### 3. 教育场景闭环

实现：

```text
资料检索
→ 学习规划
→ 导师讲解
→ 学习验收

```

而非单轮问答。

### 4. 可扩展架构

后续可扩展：

- Quiz Agent
- Memory Agent
- Learning Profile
- Multi-Agent Collaboration

---

## 后续规划

### Phase 1

- Quiz Node
- 学习测验

### Phase 2

- LangGraph Checkpointer
- 短期记忆

### Phase 3

- 长期学习画像
- 个性化推荐

### Phase 4

- 多 Agent 协作学习系统

---

## License

MIT License