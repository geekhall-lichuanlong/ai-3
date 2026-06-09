import asyncio
import json
import os
from collections.abc import AsyncIterator

import httpx

from app.models import SearchHit


class DemoAnswerGenerator:
    @property
    def model_name(self) -> str:
        return "local-demo-generator"

    async def stream_answer(self, question: str, hits: list[SearchHit]) -> AsyncIterator[str]:
        if not hits:
            answer = "未在当前租户知识库中找到足够相关的信息。"
        else:
            snippets = " ".join(hit.source.text for hit in hits[:2])
            citations = "；".join(
                f"{hit.source.filename}"
                + (f":p{hit.source.page}" if hit.source.page else "")
                + (f":L{hit.source.start_line}-{hit.source.end_line}" if hit.source.start_line else "")
                for hit in hits[:2]
            )
            answer = (
                f"根据当前租户知识库，问题“{question}”的相关答案是："
                f"{snippets[:700]} 来源：{citations}。"
            )
        for token in answer.split():
            await asyncio.sleep(0.005)
            yield token + " "


class DeepSeekAnswerGenerator:
    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")

    async def stream_answer(self, question: str, hits: list[SearchHit]) -> AsyncIterator[str]:
        context = self._build_context(hits)
        payload = {
            "model": self.model,
            "stream": True,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是企业知识库问答助手。只能根据给定上下文回答；"
                        "如果上下文不足，明确说明未找到。回答要简洁，并保留来源文件名。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"问题：{question}\n\n上下文：\n{context}",
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = event.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content")
                    if token:
                        yield token

    def _build_context(self, hits: list[SearchHit]) -> str:
        if not hits:
            return "未检索到相关 chunk。"
        parts = []
        for index, hit in enumerate(hits, start=1):
            source = hit.source
            location = f"p{source.page}" if source.page else f"L{source.start_line}-{source.end_line}"
            parts.append(
                f"[{index}] file={source.filename}, location={location}, text={source.text}"
            )
        return "\n".join(parts)


class FallbackAnswerGenerator:
    def __init__(self) -> None:
        api_key = self._read_api_key()
        self.provider = os.getenv("QA_GENERATOR", "deepseek" if api_key else "demo").lower()
        self.demo = DemoAnswerGenerator()
        self.deepseek = DeepSeekAnswerGenerator(api_key) if api_key else None

    @property
    def model_name(self) -> str:
        if self.provider == "deepseek" and self.deepseek:
            return self.deepseek.model
        return "local-demo-generator"

    def _read_api_key(self) -> str | None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key or api_key == "sk-your-deepseek-api-key":
            return None
        return api_key

    async def stream_answer(self, question: str, hits: list[SearchHit]) -> AsyncIterator[str]:
        if self.provider == "deepseek" and self.deepseek:
            try:
                async for token in self.deepseek.stream_answer(question, hits):
                    yield token
                return
            except httpx.HTTPError as exc:
                yield f"[DeepSeek 调用失败，已回退本地生成：{exc}] "
        async for token in self.demo.stream_answer(question, hits):
            yield token
