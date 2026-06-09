from datetime import UTC, datetime
from typing import Any

from app.config import TENANT_CONFIGS
from app.storage import TenantStorage


class QuotaExceeded(Exception):
    pass


class QuotaService:
    def today(self) -> str:
        return datetime.now(UTC).date().isoformat()

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def ensure_available(self, tenant_id: str, estimated_tokens: int) -> None:
        metrics = self.get_usage(tenant_id)
        if metrics["remaining_tokens"] < estimated_tokens:
            raise QuotaExceeded(
                f"tenant {tenant_id} daily token quota exceeded: "
                f"remaining={metrics['remaining_tokens']}, required={estimated_tokens}"
            )

    def record(self, tenant_id: str, tokens: int, event: dict[str, Any]) -> dict[str, Any]:
        storage = TenantStorage(tenant_id)
        usage = storage.read_usage()
        today = self.today()
        day = usage["days"].setdefault(today, {"tokens_used": 0, "events": []})
        day["tokens_used"] += tokens
        day["events"].append(event)
        storage.write_usage(usage)
        return self.get_usage(tenant_id)

    def get_usage(self, tenant_id: str) -> dict[str, Any]:
        storage = TenantStorage(tenant_id)
        usage = storage.read_usage()
        today = self.today()
        day = usage["days"].setdefault(today, {"tokens_used": 0, "events": []})
        quota = TENANT_CONFIGS[tenant_id].daily_token_quota
        return {
            "tenant_id": tenant_id,
            "date": today,
            "tokens_used": day["tokens_used"],
            "daily_token_quota": quota,
            "remaining_tokens": max(0, quota - day["tokens_used"]),
            "events": day["events"],
        }

