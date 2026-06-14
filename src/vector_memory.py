import datetime
import re
import shutil
from pathlib import Path
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.retriever import get_embedding_model

PROJECT_DIR = Path(__file__).resolve().parent.parent
MEMORY_CHROMA_DIR = PROJECT_DIR / 'data' / "memory"

SENSITIVE_PATTERNS = [
    r"api[_-]?key\s*[:=]\s*[^\s]+",
    r"secret\s*[:=]\s*[^\s]+",
    r"password\s*[:=]\s*[^\s]+",
    r"token\s*[:=]\s*[^\s]+",
]


def _get_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_time(time_text: str):
    """
    把字符串时间转成 datetime 对象。
    """

    return datetime.datetime.strptime(time_text, "%Y-%m-%d %H:%M:%S")


def _shorten(text, max_memory_char = 2000) -> str:
    """
    如果上下文文本量太大，裁剪记忆文本为2000个字符串
    """

    text= str(text or '').strip()

    if len(text) > max_memory_char:
        return text[:max_memory_char] + '...(内容过长，已截断)'

    return text


def _sanitize(text):
    """
    检测api、key、password等敏感信息并替换
    """

    text = str(text or '').strip()

    for pattern in SENSITIVE_PATTERNS:
        text = re.sub(pattern, "[已脱敏]", text, flags=re.IGNORECASE)

    return text


def get_memory_vectorstore():
    """
    获取长期记忆向量数据库
    """

    MEMORY_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name='edupilot_memory',
        embedding_function=get_embedding_model(),
        persist_directory=str(MEMORY_CHROMA_DIR),
    )


def add_vector_memory(
        context: str,
        memory_type: str = 'general',
        source_event: str = 'manual',
        user_id: str = 'local_user',
        scope: str = 'edupilot_agent',
) -> dict:
    """
    把对话的上下文保存到向量数据库中作为长期记忆
    """

    clean_context = _shorten(_sanitize(context))

    memory_id = str(uuid4())
    now = _get_now()

    metadata = {
        # 基本记忆信息
        "memory_id": memory_id,
        "memory_type": memory_type,
        "source_event": source_event,

        # 隔离不同用户和项目
        "user_id": user_id,
        "scope": scope,

        # 遗忘机制，记录生命周期
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "last_accessed_at": now,
        "access_count": 0,
    }

    document = Document(
        page_content=clean_context,
        metadata=metadata,
    )

    vectorstore = get_memory_vectorstore()
    vectorstore.add_documents([document], ids=[memory_id])

    return {
        'saved': True,
        'memory_id': memory_id,
        "content": clean_context,
        "memory_type": memory_type,
        "source_event": source_event,
        "created_at": now,
    }


def get_vector_memory_by_id(memory_id: str) -> dict | None:
    """
    根据 memory_id 读取一条长期记忆。
    """

    if not memory_id or not MEMORY_CHROMA_DIR.exists():
        return None

    vectorstore = get_memory_vectorstore()

    try:
        data = vectorstore.get(ids=[memory_id])
    except Exception:
        return None

    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    if not ids or not documents:
        return None

    return {
        "memory_id": ids[0],
        "content": documents[0],
        "metadata": metadatas[0] or {},
    }


def update_vector_memory(memory_id: str, content: str | None = None, metadata_updates: dict | None = None) -> bool:
    """
    更新一条长期记忆。
    先删除旧记录，再用同一个 memory_id 重新写入。
    这样可以兼容多数 Chroma / LangChain 版本。
    """

    old_memory = get_vector_memory_by_id(memory_id)
    if not old_memory:
        return False

    old_content = old_memory["content"]
    old_metadata = old_memory["metadata"]

    new_content = content if content is not None else old_content
    new_metadata = {
        **old_metadata,
        **(metadata_updates or {}),
        "updated_at": _get_now(),
    }

    vectorstore = get_memory_vectorstore()

    try:
        vectorstore.delete(ids=[memory_id])
        document = Document(
            page_content=new_content,
            metadata=new_metadata,
        )
        vectorstore.add_documents([document], ids=[memory_id])
        return True

    except Exception:
        return False


def update_memory_access(memory_id: str) -> bool:
    """
    某条长期记忆被语义检索命中后，更新访问次数和最近访问时间。
    """

    old_memory = get_vector_memory_by_id(memory_id)
    if not old_memory:
        return False

    metadata = old_memory["metadata"]
    old_count = int(metadata.get("access_count") or 0)

    return update_vector_memory(
        memory_id=memory_id,
        metadata_updates={
            "access_count": old_count + 1,
            "last_accessed_at": _get_now(),
        },
    )


def archive_vector_memory(memory_id: str, reason: str = "长期未访问，自动归档") -> bool:
    """
    软遗忘：不物理删除记忆，只把 status 改成 archived。
    默认检索只查 active，所以 archived 记忆不会被正常召回。
    """

    return update_vector_memory(
        memory_id=memory_id,
        metadata_updates={
            "status": "archived",
            "archived_at": _get_now(),
            "archive_reason": reason,
        },
    )


def apply_forgetting_policy(
        max_idle_days: int = 30,
        max_access_count: int = 1,
        user_id: str = "local_user",
        scope: str = "edupilot_agent",
) -> dict:
    """
    长期记忆遗忘机制。

    规则：
    1. 只处理当前 user_id + scope 的 active 记忆；
    2. 如果一条记忆超过 max_idle_days 天没有被访问；
    3. 并且 access_count <= max_access_count；
    4. 就把它软归档为 archived。
    """

    if not MEMORY_CHROMA_DIR.exists():
        return {
            "checked": 0,
            "archived": 0,
            "reason": "长期记忆库不存在",
        }

    vectorstore = get_memory_vectorstore()

    try:
        data = vectorstore.get()
    except Exception as exc:
        return {
            "checked": 0,
            "archived": 0,
            "reason": f"读取长期记忆失败：{exc}",
        }

    ids = data.get("ids") or []
    metadatas = data.get("metadatas") or []

    now = datetime.datetime.now()
    checked = 0
    archived = 0

    for memory_id, metadata in zip(ids, metadatas):
        metadata = metadata or {}

        if metadata.get("user_id") != user_id:
            continue

        if metadata.get("scope") != scope:
            continue

        if metadata.get("status", "active") != "active":
            continue

        checked += 1

        last_accessed_at = metadata.get("last_accessed_at") or metadata.get("created_at")
        last_time = _parse_time(last_accessed_at)

        if not last_time:
            continue

        idle_days = (now - last_time).days
        access_count = int(metadata.get("access_count") or 0)

        if idle_days >= max_idle_days and access_count <= max_access_count:
            ok = archive_vector_memory(
                memory_id=memory_id,
                reason=f"超过 {max_idle_days} 天未访问，且访问次数 <= {max_access_count}",
            )
            if ok:
                archived += 1

    return {
        "checked": checked,
        "archived": archived,
        "max_idle_days": max_idle_days,
        "max_access_count": max_access_count,
    }


def search_vector_memory(
        query: str,
        k: int = 4,
        user_id: str = 'local_user',
        scope: str = 'edupilot_agent',
):
    """
    根据 query 语义检索长期记忆。
    """

    query = _sanitize(query)

    vectorstore = get_memory_vectorstore()

    # 简化隔离：只召回当前 user_id + 当前项目 scope 的记忆。
    memory_filter = {
        "$and": [
            {"user_id": user_id},
            {"scope": scope},
            {"status": "active"},
        ]
    }

    try:
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=memory_filter,
        )
    except Exception:
        # 某些 Chroma 版本对空库 filter 检索报错，这里兜底返回空列表。
        return []

    memories = []
    for document, score in results:
        metadata = document.metadata or {}
        memory_id = metadata.get("memory_id", "")

        if memory_id:
            update_memory_access(memory_id)

        memories.append(
            {
                "memory_id": metadata.get("memory_id", ""),
                "memory_type": metadata.get("memory_type", "general"),
                "source_event": metadata.get("source_event", "manual"),
                "created_at": metadata.get("created_at", ""),
                "last_accessed_at": metadata.get("last_accessed_at", ""),
                "access_count": metadata.get("access_count", 0),
                "content": document.page_content,
                "score": score,
            }
        )

    return memories


def format_vector_memory(
        query: str,
        k: int = 4,
):
    """
    把语义检索到的长期记忆格式化成 prompt 上下文。
    """

    memories = search_vector_memory(query=query, k=k)
    if not memories:
        return "暂无相关长期记忆。"

    blocks = ["以下是与当前问题相关的长期记忆，可作为个性化学习参考："]
    for index, memory in enumerate(memories, start=1):
        blocks.append(
            f"""
            长期记忆 {index}
            - 类型：{memory['memory_type']}
            - 来源：{memory['source_event']}
            - 创建时间：{memory['created_at']}
            - 最近访问：{memory.get('last_accessed_at', '')}
            - 访问次数：{memory.get('access_count', 0)}
            - 语义距离：{memory['score']}
            - 内容：{memory['content']}
            """.strip()
        )

    formatted = "\n\n".join(blocks)
    return _shorten(formatted)


def get_memory_stats() -> dict:
    """
    返回长期记忆库基础统计信息。
    """

    if not MEMORY_CHROMA_DIR.exists():
        return {
            "count": 0,
            "active_count": 0,
            "archived_count": 0,
            "path": str(MEMORY_CHROMA_DIR),
        }

    try:
        vectorstore = get_memory_vectorstore()
        data = vectorstore.get()
        ids = data.get("ids") or []
        metadatas = data.get("metadatas") or []

        count = len(ids)
        active_count = 0
        archived_count = 0

        for metadata in metadatas:
            metadata = metadata or {}
            status = metadata.get("status", "active")
            if status == "archived":
                archived_count += 1
            else:
                active_count += 1

    except Exception:
        count = 0
        active_count = 0
        archived_count = 0

    return {
        "count": count,
        "active_count": active_count,
        "archived_count": archived_count,
        "path": str(MEMORY_CHROMA_DIR),
    }


def clear_vector_memory() -> None:
    """
    清空本地长期记忆向量库。
    """

    if MEMORY_CHROMA_DIR.exists():
        shutil.rmtree(MEMORY_CHROMA_DIR)

