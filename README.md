# EduPilot Agent

EduPilot is a personalized AI learning and project coaching agent built with LangGraph, RAG, memory and Streamlit.

## Features

- Personalized learning planning
- RAG-based Q&A over course documents
- Quiz generation and grading
- Learning progress tracking
- Weakness analysis and daily review

## Tech Stack

- LangGraph
- LangChain
- Streamlit
- Chroma / FAISS
- SQLite / JSON memory
- OpenAI / DeepSeek / Qwen API

## Project Goal

Build a closed-loop AI learning agent for personalized study planning, tutoring, practice, grading and progress review.

## System Architecture

用户输入
  │
  ▼
Streamlit 前端
  │
  ▼
LangGraph Supervisor
  ├── Profile Node：读取学习者画像
  ├── Planner Node：生成今日学习计划
  ├── Retriever Node：检索课程/官方文档
  ├── Tutor Node：答疑与引导式讲解
  ├── Quiz Node：生成练习题
  ├── Grader Node：批改答案并打分
  ├── Reflection Node：总结薄弱点
  └── Memory Node：写入 SQLite / JSON / Checkpoint
  │
  ▼
RAG Knowledge Base + User Memory + Progress Dashboard
