
"""
1、保存用户上传的文件，写入knowledge文件夹中，支持.md格式和.txt格式
2、加载knowledge文件夹中的文件，转为Document对象，供切分器切分
3、用递归切分器把Document文件切分成chunks
4、通过嵌入模型把chunks转变为向量保存到Chroma向量数据库
5、用户输入query后，嵌入为向量，查询向量数据库中最相似的k个chunks返回
6、格式化检索到的chunks内容及信息，提供给大模型作参考
"""


import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent           # 用Path函数，定义根目录，方便读取和操作
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

SUPPORTED_SUFFIX = ['.md', '.txt']


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


def get_embedding_model():
    """
    获取嵌入模型
    """

    model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs = {"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}            # 归一化向量，检索时只关注方向不关注大小
    )

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


def search_knowledge(query, k):
    """
    根据用户的提问，检索向量数据库中最相似的k个的chunks
    """

    if not CHROMA_DIR.exists():
        return []

    vectorstore = Chroma(
        collection_name='edupilot_knowledge',
        embedding_function=get_embedding_model(),
        persist_directory=str(CHROMA_DIR),
    )

    similar_results = vectorstore.similarity_search_with_score(query, k)

    formatted_results = []
    for chunk, score in similar_results:
        formatted_results.append(
            {
                'content': chunk.page_content,
                'source': chunk.metadata.get('source'),
                'path': chunk.metadata.get('path'),
                'chunk_id': chunk.metadata.get('chunk_id'),
                'score': score,
            }
        )

    return formatted_results


def format_retrieved_chunks(query, k=3):
    """
    把检索到的chunks内容及其他信息整理成格式化的文本，提供给大模型参考
    """

    results = search_knowledge(query, k)
    if not results:
        return '暂无可用的知识库检索结果，请先上传资料并重新构建知识库。'

    contexts = []
    for i, result in enumerate(results, start=1):
        contexts.append(
            f"""资料（{i}）
            来源：{result['source']}
            chunk编号：{result['chunk_id']}
            检索距离：{result['score']}
            内容：{result['content']}
            """
        )

    return '\n\n'.join(contexts)
