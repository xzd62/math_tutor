import os
from typing import List, Iterator

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()   # 从父目录找 .env，读 DEEPSEEK_API_KEY


class MathTutor:
    """生成模块"""

    def __init__(self, model: str = "deepseek-v4-flash",
                 temperature: float = 0.1, max_tokens: int = 2048):
        self.llm = ChatDeepSeek(
            model=model,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        """把父文档拼成带标题的上下文，超长则截断"""
        if not docs:
            return "暂无相关的知识点"
        parts = []
        current = 0
        for i, doc in enumerate(docs, 1):     
            header = f"[知识点 {i}]"
            kp = doc.metadata.get("knowledge_point")
            if kp:
                header += f"{kp}"
            if doc.metadata.get("chapter"):
                header += f" | 章节: {doc.metadata['chapter']}"
            text = f"{header}\n{doc.page_content}\n"
            if current + len(text) > max_length:
                break
            parts.append(text)
            current += len(text)
        divider = "\n" + "=" * 50 + "\n"
        return divider + divider.join(parts)

    def answer(self, question: str, docs: List[Document]) -> Iterator[str]:
         """流式回答"""
         context = self.build_context(docs)
         prompt = ChatPromptTemplate.from_template("""
你是初中数学家教老师。请严格依据下面提供的教材知识点，回答学生的问题。

要求：
1. 只依据给定知识点回答，不要编造教材外的结论
2. 分步骤讲解，每步说清楚"为什么这么做"
3. 数学公式一律用 LaTeX（行内用 $...$，独立公式用 $$...$$）
4. 先引导学生思考（提示思路），再给出答案
5. 若学生问的是具体题目，先还原知识点，再解题

相关教材知识点:
{context}

学生问题: {question}

回答:""")
         chain = (
              {"question": RunnablePassthrough(),
               "context": lambda _: context}
               | prompt
               | self.llm
               | StrOutputParser()
         )

         for chunk in chain.stream(question):
              yield chunk