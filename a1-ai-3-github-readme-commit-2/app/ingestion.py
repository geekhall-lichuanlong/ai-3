import uuid

from fastapi import UploadFile

from app.chunking import chunk_sections
from app.embedding import HashingEmbedding
from app.models import Chunk, IngestionJob, IngestionStatus
from app.parsers import parse_document
from app.storage import TenantStorage, utc_now


class IngestionService:
    def __init__(self, embedding: HashingEmbedding | None = None) -> None:
        self.embedding = embedding or HashingEmbedding()

    async def ingest(
        self,
        tenant_id: str,
        upload: UploadFile,
        chunk_size: int,
        chunk_overlap: int,
    ) -> IngestionJob:
        storage = TenantStorage(tenant_id)
        job_id = str(uuid.uuid4())
        file_id = str(uuid.uuid4())
        now = utc_now()
        job = IngestionJob(
            id=job_id,
            tenant_id=tenant_id,
            filename=upload.filename or "uploaded-file",
            status=IngestionStatus.pending,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            created_at=now,
            updated_at=now,
        )
        storage.upsert_job(job)
        data = await upload.read()
        path = storage.save_upload(file_id, job.filename, data)

        try:
            job.status = IngestionStatus.processing
            job.updated_at = utc_now()
            storage.upsert_job(job)
            sections = parse_document(path, job.filename)
            text_chunks = chunk_sections(sections, chunk_size, chunk_overlap)
            chunks = [
                Chunk(
                    id=f"{file_id}:{index}",
                    tenant_id=tenant_id,
                    file_id=file_id,
                    filename=job.filename,
                    text=text_chunk.text,
                    page=text_chunk.page,
                    start_line=text_chunk.start_line,
                    end_line=text_chunk.end_line,
                    embedding=self.embedding.embed(text_chunk.text),
                )
                for index, text_chunk in enumerate(text_chunks)
            ]
            storage.replace_file_chunks(file_id, chunks)
            job.status = IngestionStatus.done
            job.chunks_created = len(chunks)
            job.updated_at = utc_now()
            storage.upsert_job(job)
            return job
        except Exception as exc:
            job.status = IngestionStatus.failed
            job.error = str(exc)
            job.updated_at = utc_now()
            storage.upsert_job(job)
            return job

