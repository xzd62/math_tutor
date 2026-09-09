import hashlib
import warnings
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

# langchain-community 已进入 sunset, 导入会打 DeprecationWarning, 静默掉保持控制台干净
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.retrievers import BM25Retriever


def _chinese_tokenize(text: str) -> List[str]:
    """jieba 中文分词。

    BM25 默认按空白切分, 中文整句会被当成一个词导致检索失效,
    因此必须先用 jieba 切成词再交给 BM25。
    """
    with warnings.catch_warnings():
        # jieba 依赖 pkg_resources, 导入时会打弃用警告, 静默掉
        warnings.simplefilter("ignore")
        import jieba

        jieba.setLogLevel(60)  # 关闭 jieba 建词典日志
    return [w for w in jieba.lcut(text) if w.strip()]


class Retriever:
    """混合检索模块：向量语义检索 + BM25 关键词检索，RRF 融合重排"""

    def __init__(self, vectorstore: Chroma, chunks: List[Document],
                 candidate_k: int = 10):
        self.vectorstore = vectorstore
        self.chunks = chunks
        # 候选集不能超过语料规模
        self.candidate_k = max(1, min(candidate_k, len(chunks)))

        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.candidate_k},
        )
        self.bm25_retriever = BM25Retriever.from_documents(
            chunks,
            k=self.candidate_k,
            preprocess_func=_chinese_tokenize,
        )

    def search(self, query: str, k: int = 5) -> List[Document]:
        """混合检索：向量 + BM25 各取候选集，RRF 融合后取前 k"""
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        merged = self._rrf_rerank(vector_docs, bm25_docs)
        return merged[:k]

    def search_vector_only(self, query: str, k: int = 5) -> List[Document]:
        """纯向量检索（对照实验 / 调试用）"""
        return self.vector_retriever.invoke(query)[:k]

    @staticmethod
    def _doc_key(doc: Document) -> Tuple[str, str]:
        """跨来源去重键：优先 chunk_id，缺失时退化为正文哈希。

        向量检索返回的是从向量库新建的 Document 对象，BM25 返回的是
        chunks 里的原始对象——同一子块是两个不同对象，不能用 id() 去重。
        """
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            return ("id", chunk_id)
        digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        return ("hash", digest)

    def _rrf_rerank(self, vector_results: List[Document],
                    bm25_results: List[Document]) -> List[Document]:
        """RRF（Reciprocal Rank Fusion）：按"排名越靠前分越高"融合两个结果集。

        同一文档在两个结果集中都出现时分数累加，实现互相补位。
        """
        rrf_k = 60  # RRF 常规平滑参数
        scores: dict = {}
        doc_by_key: dict = {}

        for rank, doc in enumerate(vector_results):
            key = self._doc_key(doc)
            doc_by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)

        for rank, doc in enumerate(bm25_results):
            key = self._doc_key(doc)
            doc_by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)

        ordered = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_by_key[key] for key in ordered]
