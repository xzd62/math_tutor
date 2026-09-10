import gradio as gr
from pathlib import Path

from data_prep import DataPrep
from index_build import IndexBuilder
from retriever import Retriever
from generation import MathTutor

DATA_DIR = "math_textbook"
PERSIST_DIR = "chroma_db"


def setup_pipeline():
    """组装全链路：教材 → 向量库 → 检索器 → 家教"""
    prep = DataPrep(DATA_DIR)
    prep.load_textbooks()
    chunks = prep.chunk_textbook()

    builder = IndexBuilder(persist_dir=PERSIST_DIR)
    updates = builder.check_updates(DATA_DIR)

    # 新增教材 → 增量入库（只向量化新文件）；修改/删除/模型变更 → 全量重建
    need_full_rebuild = (
        not updates["index_exists"]
        or updates["model_changed"]
        or bool(updates["modified"])
        or bool(updates["removed"])
    )

    if need_full_rebuild:
        if updates["modified"] or updates["removed"]:
            print(f"[索引] 检测到教材变更: 修改={updates['modified']}, 删除={updates['removed']}")
        elif updates["model_changed"]:
            print("[索引] embedding 模型变更, 全量重建")
        builder.reset_index()
        vs = builder.build_index(chunks)
        builder.save_manifest(DATA_DIR)
        print(f"[索引] 已全量重建: {len(chunks)} 个知识点子块")
    elif updates["added"]:
        # 增量入库：只对新增教材切块向量化，不动已有索引
        print(f"[索引] 新增教材: {', '.join(updates['added'])}")
        vs = builder.load_index()
        added_set = set(updates["added"])
        new_chunks = [
            c for c in chunks
            if Path(c.metadata["source"]).relative_to(DATA_DIR).as_posix() in added_set
        ]
        vs.add_documents(
            new_chunks,
            ids=[c.metadata["chunk_id"] for c in new_chunks],
        )
        builder.save_manifest(DATA_DIR)
        print(f"[索引] 已增量入库: {len(new_chunks)} 个新子块")
    else:
        vs = builder.load_index()

    # 兜底一致性检查：清单正常但索引异常时重建
    if vs is None or vs._collection.count() != len(chunks):
        print("[索引] 索引与清单不一致, 重建")
        if vs is not None:
            vs.delete_collection()
        vs = builder.build_index(chunks)
        builder.save_manifest(DATA_DIR)

    retriever = Retriever(vs, chunks)
    tutor = MathTutor()
    return prep, retriever, tutor


def format_hits(hits):
    """把命中的知识点拼成 Markdown，便于调试召回质量"""
    if not hits:
        return "未命中任何知识点。"
    lines = []
    for i, doc in enumerate(hits, 1):
        kp = doc.metadata.get("knowledge_point") or "（章节引言）"
        chapter = doc.metadata.get("chapter", "")
        lines.append(f"{i}. **{kp}** — `{chapter}`")
    return "\n".join(lines)


def ask(message, history):
    """Gradio 回调：检索 → 子块换父文档 → 流式讲解，同时返回命中知识点"""
    hits = retriever.search(message, k=5)
    parents = prep.get_parent_documents(hits)
    kp_text = format_hits(hits)

    answer = ""
    for chunk in tutor.answer(message, parents):
        answer += chunk
        yield answer, kp_text


prep, retriever, tutor = setup_pipeline()

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},     # 块级公式
    {"left": "\\[", "right": "\\]", "display": True},   # 块级公式(另一写法)
    {"left": "$", "right": "$", "display": False},       # 行内公式
    {"left": "\\(", "right": "\\)", "display": False},   # 行内公式(另一写法)
]


with gr.Blocks(title="初中数学家教 Agent") as demo:
    gr.Markdown("## 初中数学家教 Agent\n基于教材知识点分步讲解，公式用 LaTeX 呈现。")
    kp_output = gr.Markdown(label="本次命中的知识点", latex_delimiters=LATEX_DELIMITERS)
    chatbot = gr.Chatbot(latex_delimiters=LATEX_DELIMITERS)
    gr.ChatInterface(
        fn=ask,
        title="初中数学家教",
        description="问任意初中数学问题，我会基于教材知识点给你分步讲解。",
        examples=["一元二次方程怎么解", "x的平方减5x加6等于0怎么解", "什么是判别式？"],
        additional_outputs=[kp_output],
        chatbot=chatbot,
    )

if __name__ == "__main__":
    demo.launch()
