# LangGraph 学习笔记

LangGraph 是一个用于构建有状态 AI Agent 工作流的框架。

它的核心概念包括 State、Node、Edge 和 Graph。

State 表示工作流运行过程中的共享状态，可以保存用户输入、中间结果、模型输出等信息。

Node 表示工作流中的一个处理步骤，通常是一个 Python 函数。每个节点接收 State，并返回更新后的 State。

Edge 表示节点之间的连接关系，用来控制执行顺序。

普通 Edge 表示固定流程，例如 START -> planner -> tutor -> END。

Conditional Edge 表示条件分支，可以根据 State 的内容决定下一步执行哪个节点。

在 EduPilot Agent 项目中，Planner 节点负责生成学习计划，Tutor 节点负责讲解知识点，Reviewer 节点负责生成复盘和练习任务。

未来接入 RAG 后，Retriever 节点会先从知识库中检索相关资料，再把检索结果交给 Tutor 节点生成更可靠的讲解。