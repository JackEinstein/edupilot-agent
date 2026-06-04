from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 用Path函数，方便读取和操作
KNOWLEDGE_PATH = Path('data/knowledge')
CHROMA_DIR = Path("data/chroma_db")


def load_documents():
    """
    加载本地知识库文件，并转换为Document文件
    """

    documents = []
    # 循环遍历知识库里的所有.md文件
    for file_path in KNOWLEDGE_PATH.glob('*.md'):
        # 读取.md文件里的内容
        text = file_path.read_text(encoding='utf-8')
        # 创建Document对象
        document = Document(
            page_content=text,
            metadata={'source': str(file_path)},
        )
        documents.append(document)

    return documents


def build_vectorstore():
    """
    建立向量数据库：加载本地知识库、切分文档为chunks、加载嵌入模型、chroma建立数据库
    """

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(documents)

    embed_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory=str(CHROMA_DIR),
    )

    return vectorstore


def retrieve_context(query, k):
    """
    建立检索函数，检索上一步生成的向量数据库中的文本向量
    """

    embed_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    if CHROMA_DIR.exists():
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embed_model,
        )

    else:
        vectorstore = build_vectorstore()

    # 在向量数据库中检索并返回与问题最相关的k个chunks
    chunks = vectorstore.similarity_search(query, k)

    retrieved_contexts = []
    for i, chunk in enumerate(chunks, start=1):

        content = chunk.page_content
        source = chunk.metadata.get('source', 'unknown')

        context = f'资源{i} | 来源：{source} ｜ \n内容：{content}'
        retrieved_contexts.append(context)

    return '\n\n'.join(retrieved_contexts)