# EduPilot Agent

EduPilot Agent 是一个面向个性化学习场景的 AI Agent 项目，目标是构建一个能够完成学习规划、资料检索、导师讲解、测验生成、批改反馈、追问答疑、Reflection 自检、长期记忆与短期会话记忆的学习型智能体。

项目以 **LangChain / LangGraph** 为核心，结合 **RAG、ReAct Tool Calling、Prompt Registry、Skill Registry、Reflection、长期记忆、FastAPI、Redis** 等模块，逐步形成一个可演示、可扩展、可服务化的 AI Agent 闭环系统。

---

## 1. 项目定位

EduPilot Agent 面向“个性化学习助手”场景，主要解决以下问题：

* 学生不知道如何拆解学习目标；
* 学习资料分散，缺少针对性讲解；
* 普通 Chatbot 缺乏工作流、工具调用和记忆能力；
* 学习过程缺少复盘、自测和长期跟踪；
* 本地 Demo 难以扩展为可调用的后端服务。

因此，本项目尝试构建一个具备以下能力的学习型 Agent：

```text
学习目标输入
  ↓
学习计划生成
  ↓
知识库资料检索
  ↓
导师式讲解
  ↓
测验生成与批改
  ↓
学生追问答疑
  ↓
Reflection 自检优化
  ↓
长期记忆沉淀
  ↓
FastAPI 服务化接口
  ↓
Redis 短期会话记忆
```

---

## 2. 当前核心功能

### 2.1 固定 Workflow 主链路

项目保留一个稳定的 LangGraph Workflow 主链路，用于完成完整学习流程：

```text
Planner → Tutor → Quiz / Reviewer → Reflection
```

主要能力包括：

* 根据学习目标生成今日学习计划；
* 从本地知识库检索相关资料；
* 根据计划和资料生成导师讲解；
* 自动生成小测验；
* 支持学生作答并进行智能批改；
* 支持学习复盘和验收。

---

### 2.2 RAG 本地知识库检索

项目支持本地知识库上传、切分、向量化和检索，使用 Chroma 作为向量数据库。

RAG 模块支持：

* 本地 Markdown / PDF 等资料构建知识库；
* 基于用户问题检索相关资料；
* 将检索结果注入 Tutor / QA / ReAct Agent；
* 支持不同检索参数配置。

当前检索参数包括：

```python
rag_top_k
rag_fetch_k
retrieval_mode
```

其中：

* `rag_fetch_k`：初始召回候选文档数量；
* `rag_top_k`：最终送入模型的文档数量；
* `retrieval_mode`：检索模式，例如向量召回、轻量 rerank 等。

---

### 2.3 ReAct Tool Calling Agent

项目实现了 ReAct 风格的工具调用 Agent，使系统不再只是单轮问答，而是可以根据问题自主选择工具。

ReAct Agent 可调用的能力包括：

* RAG 检索工具；
* 学习计划工具；
* 导师讲解工具；
* 测验生成工具；
* 批改工具；
* 复盘工具；
* 当前上下文工具；
* 长期记忆工具。

ReAct Agent 的典型流程为：

```text
用户问题
  ↓
Agent 判断是否需要工具
  ↓
调用相关工具
  ↓
读取工具结果
  ↓
继续推理
  ↓
生成最终回答
```

---

### 2.4 Reflection 自检机制

项目支持 Reflection 模块，对 Agent 初稿进行审查和优化。

Reflection 输出包括：

* 草稿回答质量评分；
* 存在的问题；
* 改写建议；
* 是否需要修改；
* 最终修订回答。

在 API 返回中，Reflection 相关字段包括：

```json
{
  "draft_answer": "...",
  "reflection": "...",
  "used_reflection": true,
  "final_answer": "..."
}
```

---

### 2.5 长期记忆 Long-term Memory

项目实现了长期记忆模块，用于保存对后续个性化辅导有价值的信息。

长期记忆主要记录：

* 学生长期学习目标；
* 学习偏好；
* 项目进度；
* 薄弱点；
* 已掌握内容；
* 后续建议。

长期记忆与 Redis 短期记忆不同：

| 维度   | Redis 短期记忆 | 长期记忆          |
| ---- | ---------- | ------------- |
| 作用   | 保存最近 N 轮会话 | 保存长期有价值的信息    |
| 范围   | 当前 session | 跨 session     |
| 存储   | Redis      | Chroma / 向量库  |
| 生命周期 | 临时，可设置过期   | 长期保留          |
| 典型内容 | 最近问答上下文    | 学习偏好、项目进度、薄弱点 |

---

### 2.6 Prompt Registry

项目将核心 Prompt 统一注册到 Prompt Registry 中，便于集中管理和展示。

每个 Prompt 包含：

* `name`
* `version`
* `description`
* `variables`
* `template`

这样可以避免 Prompt 散落在多个文件中，提高可维护性。

---

### 2.7 Skill Registry

项目实现 Skill Registry，用于描述 Agent 当前具备的能力模块。

Skill Registry 的作用包括：

* 展示 Agent 有哪些能力；
* 为 ReAct Agent 提供可用能力说明；
* 根据用户问题检测可能命中的 Skill；
* 在前端展示当前问题匹配到的能力模块。

示例 Skill：

```text
rag_tutor
planner
quiz_generate
quiz_grade
followup_qa
memory_personalization
reflection_review
```

---

## 3. Day 11 新增功能：FastAPI + Redis 短期会话记忆

Day 11 的目标是将原有 EduPilot Agent 从本地 Streamlit Demo 扩展为可调用的后端服务，并引入 Redis 实现 API 场景下的短期会话记忆。

本次新增：

```text
FastAPI 服务层
  +
Redis 短期会话记忆
  +
/health 健康检查接口
  +
/react/chat ReAct 对话接口
```

重要原则：

```text
新增 FastAPI 和 Redis
不破坏原 Streamlit 前端
不改动 LangGraph 固定 Workflow 主链路
```

---

## 4. Day 11 文件改动说明

### 4.1 新增 `api_server.py`

FastAPI 服务入口，提供两个接口：

```text
GET  /health
POST /react/chat
```

其中：

* `/health`：检查 FastAPI 服务和 Redis 连接状态；
* `/react/chat`：接收客户端问题，调用 ReAct Agent，读写 Redis 记忆，并返回结构化结果。

---

### 4.2 新增 `src/redis_memory.py`

Redis 短期记忆模块，核心职责：

* 根据 `session_id` 生成 Redis key；
* 读取某个 session 最近 N 轮对话；
* 写入当前轮用户问题和 Agent 回答；
* 使用 Redis List 保存对话历史；
* 使用 `LTRIM` 控制只保留最近 N 轮；
* 支持 Redis 健康检查。

Redis key 示例：

```text
edupilot:session:day11-test:react
```

---

### 4.3 修改 `src/react_agent.py`

让 ReAct Agent 支持接收 FastAPI 注入的 Redis 短期记忆。

核心思路：

```text
FastAPI 从 Redis 读取历史
  ↓
格式化为 short_term_memory
  ↓
放入 context
  ↓
run_react_agent() 接收 context
  ↓
build_system_prompt() 注入 system prompt
  ↓
Agent 回答时可以看到最近对话
```

---

### 4.4 修改 `src/prompts.py`

在 ReAct system prompt 中新增 Redis 短期会话记忆区域：

```text
【Redis 短期会话记忆】
{short_term_memory}
```

这样 Redis 记忆不是写死在 Agent 逻辑中，而是作为 Prompt Registry 的变量注入。

---

### 4.5 修改 `requirements.txt`

新增 FastAPI 和 Redis 相关依赖：

```text
fastapi
uvicorn
redis
pydantic
```

其中：

* `fastapi`：后端 API 框架；
* `uvicorn`：FastAPI 运行服务器；
* `redis`：Python Redis 客户端；
* `pydantic`：请求和响应数据模型校验。

---

### 4.6 修改 `.env.example`

新增 Redis 配置：

```env
REDIS_URL=redis://localhost:6379/0
EDUPILOT_REDIS_MAX_ROUNDS=10
EDUPILOT_REDIS_TTL_SECONDS=0
```

字段说明：

| 字段                           | 作用                   |
| ---------------------------- | -------------------- |
| `REDIS_URL`                  | Redis 连接地址           |
| `EDUPILOT_REDIS_MAX_ROUNDS`  | 默认保留最近几轮对话           |
| `EDUPILOT_REDIS_TTL_SECONDS` | Redis 记忆过期时间，0 表示不过期 |

---

## 5. FastAPI 接口说明

### 5.1 健康检查接口

```http
GET /health
```

示例请求：

```bash
curl http://127.0.0.1:8000/health
```

示例返回：

```json
{
  "status": "ok",
  "service": "edupilot-api",
  "redis": {
    "ok": true,
    "url": "redis://localhost:6379/0",
    "max_rounds": 10
  }
}
```

如果 Redis 未连接成功，`status` 可能返回：

```json
{
  "status": "degraded"
}
```

---

### 5.2 ReAct Chat 接口

```http
POST /react/chat
```

示例请求：

```bash
curl -X POST "http://127.0.0.1:8000/react/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "day11-test",
    "question": "老师，Redis 短期记忆是什么？",
    "enable_reflection": true,
    "max_memory_rounds": 10
  }'
```

示例返回字段：

```json
{
  "session_id": "day11-test",
  "final_answer": "...",
  "draft_answer": "...",
  "reflection": "...",
  "used_reflection": true,
  "trace": [],
  "matched_skills": [],
  "redis_memory": {
    "saved": true,
    "key": "edupilot:session:day11-test:react",
    "max_rounds": 10,
    "history_rounds": 1
  },
  "long_term_memory_result": {}
}
```

---

## 6. 请求体模型：ReactChatRequest

`ReactChatRequest` 用于定义 `/react/chat` 接口接收的请求数据。

主要字段：

| 字段                  | 说明                     |
| ------------------- | ---------------------- |
| `session_id`        | 客户端会话 ID，用于区分 Redis 记忆 |
| `question`          | 用户当前问题，必填              |
| `goal`              | 当前学习目标                 |
| `level`             | 学生当前水平                 |
| `hours`             | 今日可学习时间                |
| `enable_reflection` | 是否启用 Reflection        |
| `rag_top_k`         | RAG 最终使用资料数量           |
| `rag_fetch_k`       | RAG 初始召回资料数量           |
| `retrieval_mode`    | RAG 检索模式               |
| `max_memory_rounds` | Redis 最多保留最近几轮对话       |

其中 `session_id` 由客户端或前端生成，同一个会话中保持不变。

如果客户端不传 `session_id`，默认使用：

```text
default
```

但真实项目中不建议多个用户共用默认值。

---

## 7. 响应体模型：ReactChatResponse

`ReactChatResponse` 用于规范 `/react/chat` 接口返回结果。

主要字段：

| 字段                        | 说明                 |
| ------------------------- | ------------------ |
| `session_id`              | 当前会话 ID            |
| `final_answer`            | 最终回答               |
| `draft_answer`            | Reflection 前的草稿回答  |
| `reflection`              | Reflection 自检内容    |
| `used_reflection`         | 是否启用并使用 Reflection |
| `trace`                   | ReAct 工具调用轨迹       |
| `matched_skills`          | 当前问题命中的 Skill      |
| `redis_memory`            | Redis 记忆写入状态       |
| `long_term_memory_result` | 长期记忆写入结果           |

---

## 8. Redis 短期记忆机制

Redis 在本项目中用于 API 场景下的短期会话记忆。

### 8.1 写入逻辑

每次 `/react/chat` 调用完成后，系统会将本轮对话写入 Redis：

```json
{
  "question": "...",
  "final_answer": "...",
  "draft_answer": "...",
  "trace": [],
  "matched_skills": [],
  "created_at": "..."
}
```

### 8.2 读取逻辑

下一次同一个 `session_id` 请求到来时，系统会先从 Redis 中读取最近 N 轮历史，再格式化为 `short_term_memory` 注入 ReAct Agent。

### 8.3 记忆裁剪

Redis 使用 List 保存历史，并通过 `LTRIM` 保留最近 N 轮：

```text
RPUSH 新对话
LTRIM 只保留最近 max_memory_rounds 轮
```

这样可以避免会话历史无限增长。

---

## 9. session_id、thread_id、session_state 的区别

| 名称                 | 所属层级              | 作用              |
| ------------------ | ----------------- | --------------- |
| `session_id`       | FastAPI / Redis   | 区分不同客户端会话       |
| `thread_id`        | LangGraph / ReAct | 区分 Agent 内部运行线程 |
| `st.session_state` | Streamlit         | 保存页面状态          |

当前 Day 11 设计中：

```text
session_id 用于 Redis 短期记忆
thread_id 可用于 Agent 单次运行隔离
st.session_state 用于 Streamlit 页面状态
```

后续如果将 Streamlit 接入 FastAPI，可以由 Streamlit 在页面会话开始时生成 `session_id`，并在每次请求 `/react/chat` 时传给后端。

---

## 10. 本地运行方式

### 10.1 安装依赖

```bash
pip install -r requirements.txt
```

至少需要包含：

```text
fastapi
uvicorn
redis
pydantic
```

---

### 10.2 启动 Redis

如果使用 conda 安装 Redis Server：

```bash
conda activate edupilot
conda install -c conda-forge redis-server
redis-server
```

新开终端测试：

```bash
redis-cli ping
```

如果返回：

```text
PONG
```

说明 Redis 启动成功。

---

### 10.3 启动 FastAPI

在项目根目录执行：

```bash
uvicorn api_server:app --reload --port 8000
```

启动成功后，会看到：

```text
Uvicorn running on http://127.0.0.1:8000
```

其中：

* `127.0.0.1` 表示本机地址；
* `8000` 是 FastAPI 服务端口；
* `--reload` 表示开发模式下代码修改后自动重启。

---

### 10.4 测试健康检查

```bash
curl http://127.0.0.1:8000/health
```

期望返回：

```json
{
  "status": "ok",
  "service": "edupilot-api",
  "redis": {
    "ok": true
  }
}
```

---

### 10.5 测试 ReAct Chat 第一轮

```bash
curl -X POST "http://127.0.0.1:8000/react/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "day11-test",
    "question": "老师，Redis 短期记忆是什么？",
    "enable_reflection": true,
    "max_memory_rounds": 10
  }' | python -m json.tool
```

重点检查：

```json
"redis_memory": {
  "saved": true
}
```

---

### 10.6 测试同一 session_id 的连续记忆

继续使用相同的 `session_id`：

```bash
curl -X POST "http://127.0.0.1:8000/react/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "day11-test",
    "question": "刚才我问你的问题是什么？",
    "enable_reflection": true,
    "max_memory_rounds": 10
  }' | python -m json.tool
```

如果 Agent 能回答上一轮问题，例如：

```text
你刚才问的是：“老师，Redis 短期记忆是什么？”
```

说明 Redis 短期记忆读取成功。

---

### 10.7 查看 Redis 中保存的历史

```bash
redis-cli keys "*day11-test*"
```

期望看到：

```text
edupilot:session:day11-test:react
```

查看具体内容：

```bash
redis-cli LRANGE edupilot:session:day11-test:react 0 -1
```

---

## 11. 当前测试结果

Day 11 本地测试已完成以下验证：

```text
[√] redis-cli ping 返回 PONG
[√] /health 返回 status=ok
[√] FastAPI 成功连接 Redis
[√] /react/chat 可以返回 final_answer
[√] ReAct trace 正常返回
[√] matched_skills 正常返回
[√] Reflection 正常运行
[√] 第一轮对话成功写入 Redis
[√] 第二轮同 session_id 成功读取上一轮问题
[√] Redis history_rounds 从 1 增加到 2
[√] 长期记忆模块 record_react_memory 被触发
```

---

## 12. Streamlit 与 FastAPI 的关系

当前项目同时保留两个入口：

```text
Streamlit：本地可视化 Demo 入口
FastAPI：后端服务化接口入口
```

目前 Streamlit 仍可以直接调用原有 Workflow / ReAct Agent。

后续可以进一步改造为：

```text
用户在 Streamlit 页面输入问题
  ↓
Streamlit 使用 requests.post()
  ↓
请求 FastAPI /react/chat
  ↓
FastAPI 调用 ReAct Agent + Redis
  ↓
返回 final_answer
  ↓
Streamlit 展示回答
```

这样可以实现前端展示层和后端 Agent 服务层解耦。

---

## 13. 当前项目结构示意

```text
edupilot-agent/
├── app.py                         # Streamlit 前端入口
├── api_server.py                  # FastAPI 后端入口
├── requirements.txt
├── .env.example
├── data/
│   └── knowledge/                 # 本地知识库资料
└── src/
    ├── graph.py                   # LangGraph 固定 Workflow
    ├── planner.py                 # 学习计划模块
    ├── tutor.py                   # 导师讲解模块
    ├── quiz.py                    # 测验生成与批改
    ├── qa.py                      # 学生追问答疑
    ├── reviewer.py                # 复盘验收
    ├── retriever.py               # RAG 检索模块
    ├── react_agent.py             # ReAct Tool Calling Agent
    ├── tools.py                   # ReAct 工具集合
    ├── prompts.py                 # Prompt Registry
    ├── skills.py                  # Skill Registry
    ├── redis_memory.py            # Redis 短期会话记忆
    ├── long_term_memory.py        # 长期记忆
    ├── vector_memory.py           # 向量记忆底层封装
    ├── memory_reflection.py       # 记忆反思与写入判断
    └── short_term_memory.py       # 历史对话格式化工具
```

---

## 14. 面试表述

可以这样介绍 Day 11 的工作：

> Day 11 我为 EduPilot Agent 增加了 FastAPI 后端服务层，将原本主要依赖 Streamlit 的本地 Demo 扩展为可被外部系统调用的 API 服务。新增 `/health` 接口用于检查服务和 Redis 状态，新增 `/react/chat` 接口用于接收用户问题、调用 ReAct Agent、执行 Reflection，并返回结构化结果。
>
> 同时，我引入 Redis 作为短期会话记忆，根据客户端传入的 `session_id` 保存最近 N 轮对话。每次请求到来时，后端会先从 Redis 读取该 session 的历史记录，格式化后注入 ReAct Agent 的 system prompt，再生成回答；回答完成后再写回 Redis。这样实现了 API 场景下的多轮上下文记忆。
>
> 在架构上，我保留原有 Streamlit 和 LangGraph Workflow 主链路不动，只新增服务层和 Redis 记忆模块，从而实现最小侵入式扩展。当前项目已经具备 Workflow、RAG、ReAct Tool Calling、Reflection、Prompt Registry、Skill Registry、长期记忆和 Redis 短期记忆等核心 Agent 能力。

---

## 15. 后续优化方向

后续可以继续扩展：

* 将 Streamlit 前端改造为调用 FastAPI；
* 在前端增加 API 调试面板；
* 将 `trace`、`reflection`、`redis_memory` 放入折叠调试区；
* 为 `/react/chat` 增加异常降级逻辑；
* 增加 `/memory/clear` 接口，支持清空指定 session 的 Redis 记忆；
* 增加 `/sessions` 接口，查看当前 Redis 中的 session；
* 将 Redis TTL 配置用于自动过期短期记忆；
* 将 FastAPI 部署到云端，实现远程访问；
* 为接口增加用户鉴权和访问日志。
