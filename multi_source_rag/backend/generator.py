from __future__ import annotations

import os
from typing import Sequence

from openai import OpenAI
from dotenv import load_dotenv

from .retrieval import RetrievedCitation


load_dotenv()

SYSTEM_PROMPT = (
    "You are an expert assistant. Answer the question using ONLY the provided context. "
    "If the answer is not in the context, say 'I don't have enough information to answer this.' "
    "You may synthesize an answer by combining facts from multiple sources when the user asks a cross-domain question. "
    "If direct linkage is missing, explicitly state the inferred connection and the evidence used. "
    "Always cite which source(s) you used."
)


class AnswerGenerator:
    def __init__(self, model: str = "gpt-5.4-mini") -> None:
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def build_context(self, citations: Sequence[RetrievedCitation]) -> str:
        context_blocks: list[str] = []
        for idx, citation in enumerate(citations, start=1):
            context_blocks.append(
                f"[Source {idx}] {citation.source_name}\n"
                f"Retriever: {citation.retriever_type}\n"
                f"Content: {citation.full_text}"
            )
        return "\n\n".join(context_blocks)

    def generate(self, question: str, citations: Sequence[RetrievedCitation]) -> str:
        if not citations:
            return "I don't have enough information to answer this."

        context = self.build_context(citations)
        user_prompt = f"Question: {question}\n\nContext:\n{context}"

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text.strip()
