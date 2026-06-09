from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IngestionStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class Source(BaseModel):
    file_id: str
    filename: str
    page: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    text: str


class Chunk(BaseModel):
    id: str
    tenant_id: str
    file_id: str
    filename: str
    text: str
    page: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    embedding: list[float]


class IngestionJob(BaseModel):
    id: str
    tenant_id: str
    filename: str
    status: IngestionStatus
    chunk_size: int
    chunk_overlap: int
    chunks_created: int = 0
    error: str | None = None
    created_at: str
    updated_at: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=4, ge=1, le=20)


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    vector_rank: int | None
    keyword_rank: int | None
    source: Source


class SearchResponse(BaseModel):
    tenant_id: str
    hits: list[SearchHit]
    retrieval_ms: float


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=4, ge=1, le=10)


class UsageMetrics(BaseModel):
    tenant_id: str
    date: str
    tokens_used: int
    daily_token_quota: int
    remaining_tokens: int
    events: list[dict[str, Any]]

