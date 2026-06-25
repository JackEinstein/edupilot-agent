
"""
1、保存用户上传的文件，写入knowledge文件夹中，支持.md格式和.txt格式
2、加载knowledge文件夹中的文件，转为Document对象，供切分器切分
3、用递归切分器把Document文件切分成chunks
4、通过嵌入模型把chunks转变为向量保存到Chroma向量数据库
5、用户输入query后，嵌入为向量，查询向量数据库中最相似的k个chunks返回
6、格式化检索到的chunks内容及信息，提供给大模型作参考
"""

import math
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent           # 用Path函数，定义根目录，方便读取和操作
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

SUPPORTED_SUFFIX = ['.md', '.txt']

# =========================
# RAG 检索模式
# =========================
# 只做 Chroma 向量粗召回，速度最快，保留原始逻辑
RETRIEVAL_ROUGH = "rough"

# 先粗召回 fetch_k，再用自定义函数轻量重排
RETRIEVAL_LIGHT_RERANK = "light_rerank"

# 先粗召回 fetch_k，再调用 CrossEncoder rerank 模型重排
RETRIEVAL_MODEL_RERANK = "model_rerank"

# 默认先保持 rough，避免影响原有速度
DEFAULT_RETRIEVAL_MODE = RETRIEVAL_ROUGH

# rerank 模型。base 比 large 更适合本地 Demo
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


# =========================
# 基本操作：上传文件、加载、切分、嵌入向量、更新数据库
# =========================
def save_uploaded_files(uploaded_files):
    """
    保存用户上传的文件，写入知识库路径
    """

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for file in uploaded_files:
        filename = Path(file.name).name
        suffix = Path(filename).suffix.lower()

        if suffix not in SUPPORTED_SUFFIX:
            continue

        save_path = KNOWLEDGE_DIR / filename
        save_path.write_bytes(file.getbuffer())

        saved_files.append(save_path.name)

    return saved_files


def load_documents():
    """
    加载knowledge文件夹中的文件
    """

    if not KNOWLEDGE_DIR.exists():
        return []

    documents = []
    for path in KNOWLEDGE_DIR.iterdir():
        if not path.is_file():
            continue

        if Path(path).suffix.lower() not in SUPPORTED_SUFFIX:
            continue

        content = Path(path).read_text(encoding='utf-8')
        if not content:
            continue

        document = Document(
            page_content=content,
            metadata={
                'source': path.name,
                'path': str(path),
            }
        )

        documents.append(document)

    return documents


def split_documents():
    """
    切分Document对象为chunks
    """

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i

    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    获取嵌入模型。

    FastAPI 服务启动后，一个进程内只加载一次 embedding 模型，
    避免每次 RAG / 长期记忆检索都重复 Loading weights。
    """

    print("[embedding] loading HuggingFaceEmbeddings ...", flush=True)

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[embedding] loaded.", flush=True)

    return model


def rebuild_vectorstore():
    """
    用户提交文件后重建向量数据库，相当于更新向量数据库
    """

    chunks = split_documents()

    if not chunks:
        return {
            'success': False,
            'message': '没有可构建的知识库文件',
            'chunk_count': 0,
        }

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embedding_model = get_embedding_model()
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name='edupilot_knowledge',
        persist_directory=str(CHROMA_DIR),
    )

    return {
        'success': True,
        'message': '知识库重建成功',
        'chunk_count': len(chunks),
    }


# =========================
# rerank 重排序
# =========================
def _get_vectorstore():
    """
    获取 Chroma 向量库实例。
    """

    return Chroma(
        collection_name='edupilot_knowledge',
        embedding_function=get_embedding_model(),
        persist_directory=str(CHROMA_DIR),
    )


def _distance_to_similarity(distance: float) -> float:
    """
    Chroma 返回的是 distance，通常越小越相似。
    这里把 distance 转成 similarity，便于后面和关键词分数融合。
    """

    try:
        distance = float(distance)
    except Exception:
        return 0.0

    if distance < 0:
        distance = 0.0

    return 1.0 / (1.0 + distance)


def _tokenize(text: str) -> list[str]:
    """
    轻量分词函数。
    不引入 jieba，避免增加依赖。
    兼容英文术语、代码变量名和中文文本。
    """

    text = str(text or "").lower()

    # 英文/数字/下划线/短横线，或者连续中文
    raw_tokens = re.findall(r"[a-zA-Z0-9_\-]+|[\u4e00-\u9fff]+", text)

    tokens = []
    for token in raw_tokens:
        token = token.strip("-_ ")
        if not token:
            continue

        tokens.append(token)

        # 中文长词额外拆 bigram，例如“长期记忆” -> “长期”“期记”“记忆”
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) >= 2:
            for i in range(len(token) - 1):
                tokens.append(token[i: i + 2])

    return tokens


def _keyword_score(query: str, content: str) -> float:
    """
    计算 query 和 chunk 内容的关键词重合度。
    分数范围大致在 0~1。
    """

    query_tokens = set(_tokenize(query))
    content_tokens = set(_tokenize(content))

    if not query_tokens or not content_tokens:
        return 0.0

    overlap = query_tokens & content_tokens

    # query 里的关键词被 chunk 命中了多少
    recall = len(overlap) / max(len(query_tokens), 1)

    # chunk 很长时，避免简单重合被过度放大
    length_penalty_base = max(math.sqrt(len(content_tokens)), 1)
    precision_like = len(overlap) / length_penalty_base

    return min(1.0, 0.75 * recall + 0.25 * precision_like)


def _format_result(
    chunk: Document,
    distance: float,
    initial_rank: int,
    retrieval_mode: str,
    keyword_score: float | None = None,
    rerank_score: float | None = None,
    model_rerank_score: float | None = None,
) -> dict:
    """
    统一格式化检索结果，避免三种模式返回字段不一致。
    """

    vector_score = _distance_to_similarity(distance)

    return {
        'content': chunk.page_content,
        'source': chunk.metadata.get('source'),
        'path': chunk.metadata.get('path'),
        'chunk_id': chunk.metadata.get('chunk_id'),

        # 保留原来的 score 字段，避免旧代码潜在依赖
        'score': float(distance),

        # 新增更清晰的字段
        'distance': float(distance),
        'vector_score': round(vector_score, 4),
        'keyword_score': None if keyword_score is None else round(keyword_score, 4),
        'rerank_score': None if rerank_score is None else round(rerank_score, 4),
        'model_rerank_score': None if model_rerank_score is None else round(model_rerank_score, 4),
        'initial_rank': initial_rank,
        'retrieval_mode': retrieval_mode,
    }


def _lightweight_rerank_results(
    query: str,
    raw_results: Iterable[tuple[Document, float]],
) -> list[dict]:
    """
    自定义轻量 rerank。

    思路：
    1. Chroma distance 转成 vector_score；
    2. 计算 query 和 chunk 的 keyword_score；
    3. 二者加权融合得到 rerank_score。
    """

    reranked = []

    for rank, (chunk, distance) in enumerate(raw_results, start=1):
        vector_score = _distance_to_similarity(distance)
        keyword = _keyword_score(query, chunk.page_content)

        rerank_score = 0.7 * vector_score + 0.3 * keyword

        reranked.append(
            _format_result(
                chunk=chunk,
                distance=distance,
                initial_rank=rank,
                retrieval_mode=RETRIEVAL_LIGHT_RERANK,
                keyword_score=keyword,
                rerank_score=rerank_score,
            )
        )

    return sorted(reranked, key=lambda item: item['rerank_score'], reverse=True)


@lru_cache(maxsize=1)
def _get_reranker_model():
    """
    懒加载 CrossEncoder rerank 模型。

    注意：
    1. 只有选择 model_rerank 模式时才会加载；
    2. 第一次运行可能会下载模型；
    3. lru_cache 保证模型只加载一次。
    """

    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL_NAME)


def _model_rerank_results(
    query: str,
    raw_results: Iterable[tuple[Document, float]],
) -> list[dict]:
    """
    使用 CrossEncoder rerank 模型对 Chroma 粗召回结果重排。
    """

    candidates = list(raw_results)
    if not candidates:
        return []

    reranker = _get_reranker_model()

    pairs = [
        [query, chunk.page_content]
        for chunk, _distance in candidates
    ]

    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    reranked = []

    for rank, ((chunk, distance), score) in enumerate(zip(candidates, scores), start=1):
        model_score = float(score)

        reranked.append(
            _format_result(
                chunk=chunk,
                distance=distance,
                initial_rank=rank,
                retrieval_mode=RETRIEVAL_MODEL_RERANK,
                model_rerank_score=model_score,
                rerank_score=model_score,
            )
        )

    return sorted(reranked, key=lambda item: item['model_rerank_score'], reverse=True)


# =========================
# 最终检索关键词向量、形成格式化 prompt
# =========================
def search_knowledge(
    query,
    k=3,
    fetch_k=None,
    retrieval_mode=DEFAULT_RETRIEVAL_MODE,
):
    """
    根据用户提问检索知识库。

    三种模式：
    1. rough：
       只做 Chroma top_k 粗召回，速度最快。

    2. light_rerank：
       Chroma 先召回 fetch_k 个候选，再用自定义轻量 rerank 重排。

    3. model_rerank：
       Chroma 先召回 fetch_k 个候选，再用 CrossEncoder rerank 模型重排。
    """

    query = str(query or "").strip()
    if not query:
        return []

    if not CHROMA_DIR.exists():
        return []

    k = max(int(k or 3), 1)
    retrieval_mode = retrieval_mode or DEFAULT_RETRIEVAL_MODE

    vectorstore = _get_vectorstore()

    # 1. rough retrieve：只做 Chroma 粗召回
    if retrieval_mode == RETRIEVAL_ROUGH:
        try:
            raw_results = vectorstore.similarity_search_with_score(query, k=k)
        except Exception:
            return []

        return [
            _format_result(
                chunk=chunk,
                distance=distance,
                initial_rank=rank,
                retrieval_mode=RETRIEVAL_ROUGH,
            )
            for rank, (chunk, distance) in enumerate(raw_results, start=1)
        ]

    # 2. rerank 类模式：先扩大候选集
    fetch_k = int(fetch_k or max(k * 4, 8))
    fetch_k = max(fetch_k, k)

    try:
        raw_results = vectorstore.similarity_search_with_score(query, k=fetch_k)
    except Exception:
        return []

    # 3. model rerank：如果模型失败，自动降级到 lightweight，避免整个系统崩掉
    if retrieval_mode == RETRIEVAL_MODEL_RERANK:
        try:
            return _model_rerank_results(
                query=query,
                raw_results=raw_results,
            )[:k]
        except Exception as exc:
            fallback_results = _lightweight_rerank_results(
                query=query,
                raw_results=raw_results,
            )[:k]

            for item in fallback_results:
                item['retrieval_mode'] = 'model_rerank_fallback_to_lightweight'
                item['model_rerank_error'] = str(exc)

            return fallback_results

    # 4. 默认 lightweight rerank
    return _lightweight_rerank_results(
        query=query,
        raw_results=raw_results,
    )[:k]


def format_retrieved_chunks(
    query,
    k=3,
    fetch_k=None,
    retrieval_mode=DEFAULT_RETRIEVAL_MODE,
):
    """
    把检索到的 chunks 内容及其他信息整理成格式化文本，提供给大模型参考。
    """

    results = search_knowledge(
        query=query,
        k=k,
        fetch_k=fetch_k,
        retrieval_mode=retrieval_mode,
    )

    if not results:
        return '暂无可用的知识库检索结果，请先上传资料并重新构建知识库。'

    contexts = []

    for i, result in enumerate(results, start=1):
        contexts.append(
            f"""资料（{i}）
            来源：{result.get('source')}
            chunk编号：{result.get('chunk_id')}
            检索模式：{result.get('retrieval_mode')}
            初始向量排名：{result.get('initial_rank')}
            检索距离：{result.get('distance')}
            向量相似度：{result.get('vector_score')}
            关键词分数：{result.get('keyword_score')}
            rerank分数：{result.get('rerank_score')}
            模型rerank分数：{result.get('model_rerank_score')}
            内容：{result.get('content')}
            """
        )

    return '\n\n'.join(contexts)