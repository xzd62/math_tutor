import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class IndexBuilder:
    """索引构建模块：子块 → Chroma 向量库（含教材变更自动检测）"""

    MANIFEST_NAME = "manifest.json"
    COLLECTION_NAME = "math"

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

    # ---------- 教材哈希清单 ----------

    @property
    def manifest_path(self) -> Path:
        """清单文件路径：与索引同目录，随 chroma_db 一起被 gitignore"""
        return Path(self.persist_dir) / self.MANIFEST_NAME

    def compute_hashes(self, data_dir: str) -> Dict[str, str]:
        """逐文件内容哈希：{相对路径(正斜杠): md5}"""
        data_path = Path(data_dir)
        hashes: Dict[str, str] = {}
        for md_file in sorted(data_path.rglob("*.md")):
            digest = hashlib.md5(md_file.read_bytes()).hexdigest()
            key = md_file.relative_to(data_path).as_posix()
            hashes[key] = digest
        return hashes

    def check_updates(self, data_dir: str) -> dict:
        """对比教材与索引清单，返回更新状态。

        Returns:
            {
              "index_exists": bool,    # 索引目录与清单均存在且可读
              "model_changed": bool,   # embedding 模型名与建索引时不一致
              "added": [str],          # 新增的教材文件（相对路径）
              "modified": [str],       # 内容变化的已有文件
              "removed": [str],        # 已被删除的教材文件
            }
        """
        current = self.compute_hashes(data_dir)

        if not self.manifest_path.exists():
            return {
                "index_exists": False,
                "model_changed": False,
                "added": sorted(current.keys()),
                "modified": [],
                "removed": [],
            }

        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {
                "index_exists": False,
                "model_changed": False,
                "added": sorted(current.keys()),
                "modified": [],
                "removed": [],
            }

        old = manifest.get("files", {})
        added = sorted(set(current) - set(old))                       # 新增
        removed = sorted(set(old) - set(current))                     # 删除
        modified = sorted(k for k in current if old.get(k) and old.get(k) != current[k])
        return {
            "index_exists": True,
            "model_changed": manifest.get("model_name") != self.model_name,
            "added": added,
            "modified": modified,
            "removed": removed,
        }

    def save_manifest(self, data_dir: str) -> None:
        """记录本次建索引对应的教材哈希与模型名，供下次比对"""
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_name": self.model_name,
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "files": self.compute_hashes(data_dir),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset_index(self) -> None:
        """删除整个索引目录。

        必须在创建 Chroma 客户端之前调用，否则 Windows 下
        sqlite/hnsw 文件被进程占用，删除会失败。
        """
        if Path(self.persist_dir).exists():
            shutil.rmtree(self.persist_dir)

    # ---------- 索引构建 / 加载 ----------

    def build_index(self, chunks: List[Document]) -> Chroma:
        """将子块向量化并存储到chroma向量数据库"""
        if not chunks:
            raise ValueError("子块列表不能为空")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.COLLECTION_NAME,
            # 向量 id 直接使用确定性 chunk_id，保证全量重建与增量入库的 id 一致
            ids=[c.metadata.get("chunk_id") for c in chunks],
        )
        return self.vectorstore

    def load_index(self) -> Optional[Chroma]:
        """从磁盘读回向量库，索引不存在则返回None"""
        if not Path(self.persist_dir).exists():
            return None
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.COLLECTION_NAME,
        )
        return self.vectorstore