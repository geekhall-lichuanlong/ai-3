import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import DATA_DIR, TENANTS
from app.models import Chunk, IngestionJob


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_tenant(tenant_id: str) -> str:
    if tenant_id not in TENANTS:
        raise KeyError(f"unknown tenant: {tenant_id}")
    return tenant_id


class TenantStorage:
    def __init__(self, tenant_id: str, base_dir: Path | None = None) -> None:
        self.tenant_id = validate_tenant(tenant_id)
        self.root = (base_dir or DATA_DIR) / self.tenant_id
        self.files_dir = self.root / "files"
        self.manifest_path = self.root / "manifest.json"
        self.vector_index_path = self.root / "vector_index.json"
        self.keyword_index_path = self.root / "keyword_index.json"
        self.usage_path = self.root / "usage.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_json(self.manifest_path, {"jobs": {}})
        self._ensure_json(self.vector_index_path, {"chunks": []})
        self._ensure_json(self.keyword_index_path, {"chunks": []})
        self._ensure_json(self.usage_path, {"days": {}})

    def _ensure_json(self, path: Path, default: dict[str, Any]) -> None:
        if not path.exists():
            path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_upload(self, file_id: str, filename: str, data: bytes) -> Path:
        safe_name = Path(filename).name
        path = self.files_dir / f"{file_id}_{safe_name}"
        path.write_bytes(data)
        return path

    def upsert_job(self, job: IngestionJob) -> None:
        manifest = self._read_json(self.manifest_path)
        manifest["jobs"][job.id] = job.model_dump(mode="json")
        self._write_json(self.manifest_path, manifest)

    def get_job(self, job_id: str) -> IngestionJob | None:
        manifest = self._read_json(self.manifest_path)
        raw = manifest["jobs"].get(job_id)
        return IngestionJob.model_validate(raw) if raw else None

    def list_jobs(self) -> list[IngestionJob]:
        manifest = self._read_json(self.manifest_path)
        return [IngestionJob.model_validate(raw) for raw in manifest["jobs"].values()]

    def replace_file_chunks(self, file_id: str, chunks: list[Chunk]) -> None:
        vector_index = self._read_json(self.vector_index_path)
        keyword_index = self._read_json(self.keyword_index_path)
        remaining_vector = [chunk for chunk in vector_index["chunks"] if chunk["file_id"] != file_id]
        remaining_keyword = [chunk for chunk in keyword_index["chunks"] if chunk["file_id"] != file_id]
        serialized = [chunk.model_dump(mode="json") for chunk in chunks]
        vector_index["chunks"] = remaining_vector + serialized
        keyword_index["chunks"] = remaining_keyword + serialized
        self._write_json(self.vector_index_path, vector_index)
        self._write_json(self.keyword_index_path, keyword_index)

    def load_chunks(self) -> list[Chunk]:
        vector_index = self._read_json(self.vector_index_path)
        return [Chunk.model_validate(raw) for raw in vector_index["chunks"]]

    def read_usage(self) -> dict[str, Any]:
        return self._read_json(self.usage_path)

    def write_usage(self, usage: dict[str, Any]) -> None:
        self._write_json(self.usage_path, usage)
