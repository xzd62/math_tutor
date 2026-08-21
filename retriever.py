from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document


class Retriever:
    """检索模块"""

    def __init__(self, vectorstore: Chroma, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks

    def search(self, query: str, k: int = 5) -> List[Document]:
        """向量检索"""

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        return retriever.invoke(query)