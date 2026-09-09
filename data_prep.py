import hashlib
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


class DataPrep:
    # 准备数据

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.documents = []  # 父文档
        self.chunks = []  # 子块
        self.parent_child_map = {}

    def load_textbooks(self) -> List[Document]:
        # 读取父文档
        documents = []
        data_dir_obj = Path(self.data_dir)

        for math_text in data_dir_obj.rglob("*.md"):

            with open(math_text, "r", encoding="utf-8") as r:
                content = r.read()

            # 确定性 id：由文件路径生成，保证每次运行同一个文件得到同一个 parent_id，
            # 持久化的向量库才能跨运行和父文档对上（uuid 每次随机会导致索引失效）
            parent_id = hashlib.md5(str(math_text.resolve()).encode("utf-8")).hexdigest()

            doc = Document(
                page_content=content,
                metadata={
                    "source": str(math_text),
                    "parent_id": parent_id,
                    "chapter": math_text.stem,
                    "doc_type": "parent"
                }
            )

            documents.append(doc)

        self.documents = documents
        return documents

    def chunk_textbook(self) -> List[Document]:
        """对父文档进行切块"""
        if not self.documents:
            raise ValueError("请先调用 load_textbooks()")

        all_chunks = []
        spiltter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("##", "section"),
                ("###", "knowledge_point"),
            ],
            strip_headers=False,
        )

        for doc in self.documents:
            md_chunks = spiltter.split_text(doc.page_content)
            parent_id = doc.metadata["parent_id"]

            for i, chunk in enumerate(md_chunks):
                # 确定性 id：由父文档 id + 块序号哈希生成，跨运行稳定。
                # 向量库里的副本与本次运行的子块才能按 chunk_id 正确去重
                child_id = hashlib.md5(f"{parent_id}:{i}".encode("utf-8")).hexdigest()
                chunk.metadata.update(doc.metadata)
                chunk.metadata.update({
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_index": i
                }) 

                self.parent_child_map[child_id] = parent_id
                all_chunks.append(chunk)
    
        self.chunks = all_chunks
        return all_chunks

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """通过子块获取到父文档"""
        relevance: Dict[str, int] = {}
        parent_map: Dict[str, Document] = {}

        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if not parent_id:
                continue
            relevance[parent_id] = relevance.get(parent_id, 0) + 1
            if parent_id not in parent_map:
                for doc in self.documents:
                    if doc.metadata.get("parent_id") == parent_id:
                        parent_map[parent_id] = doc
                        break

        sorted_ids = sorted(relevance, key=lambda x: relevance[x], reverse=True)
        return [parent_map[pid] for pid in sorted_ids if pid in parent_map]