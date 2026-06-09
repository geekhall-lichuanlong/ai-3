from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

TENANTS = ("tenant_a", "tenant_b", "tenant_c")


@dataclass(frozen=True)
class TenantConfig:
    model: str
    daily_token_quota: int


TENANT_CONFIGS: dict[str, TenantConfig] = {
    "tenant_a": TenantConfig(model="gpt-4.1-mini", daily_token_quota=2_000),
    "tenant_b": TenantConfig(model="claude-3.5-sonnet", daily_token_quota=1_500),
    "tenant_c": TenantConfig(model="deepseek-v4-flash", daily_token_quota=1_000),
}


DATA_DIR = Path(os.getenv("TENANT_QA_DATA_DIR", "data/tenants"))
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
EMBEDDING_DIMS = 384
RRF_K = 60
