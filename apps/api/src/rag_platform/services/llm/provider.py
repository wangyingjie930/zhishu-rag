from typing import Protocol, Sequence

from rag_platform.services.retrieval.hybrid import RetrievedChunk


class LLMProvider(Protocol):
    async def answer(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        ...


class MockLLMProvider:
    async def answer(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        if not contexts:
            return "我没有在当前知识库中检索到足够相关的内容。建议补充文档或换一种问法。"
        bullets = "\n".join(
            f"- [{idx}] {chunk.content[:220].strip()}" for idx, chunk in enumerate(contexts, start=1)
        )
        return (
            "下面是基于知识库检索结果生成的回答草稿。生产环境可在 LLMProvider 中接入企业模型网关、"
            "提示词模板、敏感信息过滤和引用一致性校验。\n\n"
            f"问题：{question}\n\n"
            f"依据：\n{bullets}"
        )

