from pathlib import Path
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class IndexBuilder:
    """索引构建模块：子块 → Chroma 向量库"""
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 persist_dir: str = "chroma_db"):
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: Optional[Chroma] = None

    def build_index(self, chunks: List[Document]) -> Chroma:
        """将子块向量化并存储到chroma向量数据库"""
        if not chunks:
            raise ValueError("子块列表不能为空")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="math"
        )
        return self.vectorstore

    def load_index(self) -> Optional[Chroma]:
        """从磁盘读回向量库，索引不存在则返回None"""
        if not Path(self.persist_dir).exists():
            return None
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="math",
        )
        return self.vectorstore