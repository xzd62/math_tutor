import os
from pathlib import Path

import gradio as gr

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
    if Path(PERSIST_DIR).exists():
        vs = builder.load_index()
        # 索引缺失或教材变多（子块数量对不上）时重建
        if vs is None or vs._collection.count() != len(chunks):
            vs = builder.build_index(chunks)
    else:
        vs = builder.build_index(chunks)

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

with gr.Blocks(title="初中数学家教 Agent") as demo:
    gr.Markdown("## 初中数学家教 Agent\n基于教材知识点分步讲解，公式用 LaTeX 呈现。")
    kp_output = gr.Markdown(label="本次命中的知识点")
    gr.ChatInterface(
        fn=ask,
        title="初中数学家教",
        description="问任意初中数学问题，我会基于教材知识点给你分步讲解。",
        examples=["一元二次方程怎么解", "x的平方减5x加6等于0怎么解", "什么是判别式？"],
        additional_outputs=[kp_output],
    )

if __name__ == "__main__":
    demo.launch()
