import json
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Path, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from app.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, TENANT_CONFIGS, TENANTS
from app.generator import FallbackAnswerGenerator
from app.ingestion import IngestionService
from app.models import IngestionJob, QARequest, SearchRequest, SearchResponse, UsageMetrics
from app.quota import QuotaExceeded, QuotaService
from app.retrieval import HybridRetriever
from app.storage import TenantStorage

app = FastAPI(title="Multi-tenant AI QA", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ingestion_service = IngestionService()
retriever = HybridRetriever()
quota_service = QuotaService()
generator = FallbackAnswerGenerator()


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>多租户 AI 问答演示</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #667085;
      --line: #d0d5dd;
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --ok: #027a48;
      --warn: #b54708;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.3;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 18px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 16px;
    }
    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin: 12px 0 6px;
    }
    select, input, textarea, button {
      width: 100%;
      font: inherit;
    }
    select, input, textarea {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      background: #fff;
      color: var(--text);
    }
    textarea {
      min-height: 112px;
      resize: vertical;
      line-height: 1.5;
    }
    button {
      margin-top: 12px;
      border: 0;
      border-radius: 6px;
      padding: 10px 12px;
      background: var(--primary);
      color: #fff;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { background: var(--primary-dark); }
    button:disabled {
      background: #98a2b3;
      cursor: not-allowed;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .status {
      min-height: 22px;
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    .answer {
      min-height: 260px;
      white-space: pre-wrap;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      line-height: 1.7;
      background: #fcfcfd;
    }
    .metrics, .sources {
      margin-top: 14px;
      display: grid;
      gap: 10px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric, .source {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .metric strong {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .source h3 {
      margin: 0 0 6px;
      font-size: 14px;
    }
    .source p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 13px;
    }
    .links a {
      color: var(--primary);
      text-decoration: none;
      margin-left: 14px;
      font-size: 14px;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: 1fr 1fr; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <h1>多租户 AI 问答演示</h1>
    <div class="links">
      <a href="/docs" target="_blank">Swagger API</a>
      <a href="/tenants" target="_blank">租户配置</a>
    </div>
  </header>
  <main>
    <section>
      <h2>知识库摄取</h2>
      <label for="tenant">租户</label>
      <select id="tenant">
        <option value="tenant_a">tenant_a</option>
        <option value="tenant_b">tenant_b</option>
        <option value="tenant_c">tenant_c</option>
      </select>
      <label for="file">上传 PDF / Markdown / TXT</label>
      <input id="file" type="file" accept=".pdf,.md,.markdown,.txt">
      <div class="row">
        <div>
          <label for="chunkSize">Chunk 大小</label>
          <input id="chunkSize" type="number" value="500" min="80" max="2000">
        </div>
        <div>
          <label for="chunkOverlap">Overlap</label>
          <input id="chunkOverlap" type="number" value="80" min="0" max="400">
        </div>
      </div>
      <button id="uploadBtn">上传并摄取</button>
      <div id="uploadStatus" class="status"></div>
    </section>

    <section>
      <h2>基于当前租户语料库问答</h2>
      <label for="question">问题</label>
      <textarea id="question">Tenant A 的 SLA 是多少？</textarea>
      <div class="row">
        <div>
          <label for="topK">使用 chunks</label>
          <input id="topK" type="number" value="4" min="1" max="10">
        </div>
        <div>
          <label for="mode">当前接口</label>
          <input id="mode" value="SSE /qa/stream" disabled>
        </div>
      </div>
      <button id="askBtn">向大模型提问</button>
      <div id="qaStatus" class="status"></div>
      <div id="answer" class="answer"></div>
      <div id="metrics" class="metrics"></div>
      <div id="sources" class="sources"></div>
    </section>
  </main>

  <script>
    const tenantEl = document.getElementById("tenant");
    const uploadBtn = document.getElementById("uploadBtn");
    const askBtn = document.getElementById("askBtn");
    const uploadStatus = document.getElementById("uploadStatus");
    const qaStatus = document.getElementById("qaStatus");
    const answerEl = document.getElementById("answer");
    const metricsEl = document.getElementById("metrics");
    const sourcesEl = document.getElementById("sources");

    uploadBtn.addEventListener("click", async () => {
      const file = document.getElementById("file").files[0];
      if (!file) {
        uploadStatus.textContent = "请选择一个文件。";
        return;
      }
      uploadBtn.disabled = true;
      uploadStatus.textContent = "摄取中...";
      const form = new FormData();
      form.append("file", file);
      const params = new URLSearchParams({
        chunk_size: document.getElementById("chunkSize").value,
        chunk_overlap: document.getElementById("chunkOverlap").value
      });
      try {
        const res = await fetch(`/tenants/${tenantEl.value}/files?${params}`, {
          method: "POST",
          body: form
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "上传失败");
        uploadStatus.textContent = `完成：${data.filename}，chunks=${data.chunks_created}，job=${data.id}`;
      } catch (err) {
        uploadStatus.textContent = `失败：${err.message}`;
      } finally {
        uploadBtn.disabled = false;
      }
    });

    askBtn.addEventListener("click", async () => {
      askBtn.disabled = true;
      answerEl.textContent = "";
      metricsEl.innerHTML = "";
      sourcesEl.innerHTML = "";
      qaStatus.textContent = "检索语料库并请求大模型...";

      const tenant = tenantEl.value;
      const question = document.getElementById("question").value.trim();
      const topK = Number(document.getElementById("topK").value || 4);

      try {
        const sourceRes = await fetch(`/tenants/${tenant}/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: question, top_k: topK })
        });
        const sourceData = await sourceRes.json();
        renderSources(sourceData.hits || []);

        const res = await fetch(`/tenants/${tenant}/qa/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, top_k: topK })
        });
        if (!res.ok || !res.body) throw new Error("问答请求失败");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\\n\\n");
          buffer = events.pop() || "";
          for (const eventText of events) {
            handleSseEvent(eventText);
          }
        }
        qaStatus.textContent = "回答完成。";
      } catch (err) {
        qaStatus.textContent = `失败：${err.message}`;
      } finally {
        askBtn.disabled = false;
      }
    });

    function handleSseEvent(raw) {
      const eventLine = raw.split("\\n").find(line => line.startsWith("event: "));
      const dataLine = raw.split("\\n").find(line => line.startsWith("data: "));
      if (!eventLine || !dataLine) return;
      const event = eventLine.replace("event: ", "");
      const data = dataLine.replace("data: ", "");
      if (event === "token") {
        answerEl.textContent += data;
      }
      if (event === "metrics") {
        renderMetrics(JSON.parse(data));
      }
      if (event === "error") {
        qaStatus.textContent = JSON.parse(data).message;
      }
    }

    function renderMetrics(metrics) {
      metricsEl.innerHTML = `
        <div class="metric-grid">
          <div class="metric"><strong>模型</strong>${metrics.model}</div>
          <div class="metric"><strong>Token</strong>${metrics.token_usage}</div>
          <div class="metric"><strong>检索耗时</strong>${metrics.retrieval_ms} ms</div>
          <div class="metric"><strong>生成耗时</strong>${metrics.generation_ms} ms</div>
        </div>
      `;
    }

    function renderSources(hits) {
      sourcesEl.innerHTML = "";
      if (!hits.length) {
        sourcesEl.innerHTML = '<div class="source"><h3>未找到来源</h3><p>请先上传当前租户的知识库文件。</p></div>';
        return;
      }
      for (const hit of hits) {
        const source = hit.source;
        const location = source.page ? `p${source.page}` : `L${source.start_line}-${source.end_line}`;
        const item = document.createElement("div");
        item.className = "source";
        item.innerHTML = `<h3>${source.filename} · ${location}</h3><p>${escapeHtml(source.text.slice(0, 360))}</p>`;
        sourcesEl.appendChild(item);
      }
    }

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[char]));
    }
  </script>
</body>
</html>
"""


def tenant_path(tenant_id: str) -> str:
    if tenant_id not in TENANTS:
        raise HTTPException(status_code=404, detail=f"unknown tenant: {tenant_id}")
    return tenant_id


def sse(event: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/tenants")
def tenants() -> dict[str, object]:
    return {
        "tenants": [
            {
                "id": tenant_id,
                "model": config.model,
                "daily_token_quota": config.daily_token_quota,
            }
            for tenant_id, config in TENANT_CONFIGS.items()
        ]
    }


@app.post("/tenants/{tenant_id}/files", response_model=IngestionJob)
async def upload_file(
    tenant_id: str = Path(...),
    file: UploadFile = File(...),
    chunk_size: int = Query(DEFAULT_CHUNK_SIZE, ge=80, le=2_000),
    chunk_overlap: int = Query(DEFAULT_CHUNK_OVERLAP, ge=0, le=400),
) -> IngestionJob:
    tenant_id = tenant_path(tenant_id)
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=422, detail="chunk_overlap must be smaller than chunk_size")
    job = await ingestion_service.ingest(tenant_id, file, chunk_size, chunk_overlap)
    if job.status == "failed":
        raise HTTPException(status_code=400, detail=job.error)
    return job


@app.get("/tenants/{tenant_id}/ingestions/{job_id}", response_model=IngestionJob)
def ingestion_status(tenant_id: str, job_id: str) -> IngestionJob:
    tenant_id = tenant_path(tenant_id)
    job = TenantStorage(tenant_id).get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="ingestion job not found")
    return job


@app.post("/tenants/{tenant_id}/search", response_model=SearchResponse)
def search(tenant_id: str, request: SearchRequest) -> SearchResponse:
    tenant_id = tenant_path(tenant_id)
    return retriever.search(tenant_id, request.query, request.top_k)


@app.post("/tenants/{tenant_id}/qa/stream")
async def qa_stream(tenant_id: str, request: QARequest) -> StreamingResponse:
    tenant_id = tenant_path(tenant_id)

    async def events() -> AsyncIterator[str]:
        estimated = quota_service.estimate_tokens(request.question) + 120
        try:
            quota_service.ensure_available(tenant_id, estimated)
        except QuotaExceeded as exc:
            yield sse("error", {"message": str(exc)})
            return

        retrieval_started = time.perf_counter()
        result = retriever.search(tenant_id, request.question, request.top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        generated = ""
        generation_started = time.perf_counter()
        async for token in generator.stream_answer(request.question, result.hits):
            generated += token
            yield sse("token", token)
        generation_ms = (time.perf_counter() - generation_started) * 1000

        token_usage = quota_service.estimate_tokens(request.question + generated)
        metrics = {
            "tenant_id": tenant_id,
            "token_usage": token_usage,
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "chunks_used": len(result.hits),
            "model": generator.model_name,
            "tenant_model_preference": TENANT_CONFIGS[tenant_id].model,
        }
        quota_service.record(tenant_id, token_usage, metrics)
        yield sse("metrics", metrics)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/tenants/{tenant_id}/usage", response_model=UsageMetrics)
def usage(tenant_id: str) -> UsageMetrics:
    tenant_id = tenant_path(tenant_id)
    return UsageMetrics.model_validate(quota_service.get_usage(tenant_id))
