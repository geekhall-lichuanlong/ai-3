# 多租户 AI 问答系统核心模块

一个可本地演示的 B2B SaaS 多租户 AI 问答模块。系统支持 `tenant_a`、`tenant_b`、`tenant_c` 三个虚拟租户，提供文件摄取、可配置 chunking、租户物理隔离索引、向量 + BM25 混合检索、RRF 融合、DeepSeek/本地 fallback 流式问答、每日 token 配额控制和浏览器演示页面。

## 功能概览

- 多租户隔离：每个租户独立目录、独立文件、独立向量索引和关键词索引。
- 文件摄取：支持 PDF、Markdown、TXT。
- Chunking：支持配置 `chunk_size` 和 `chunk_overlap`。
- 摄取状态：支持 `pending / processing / done / failed`。
- 混合检索：Hashing embedding 向量检索 + BM25 关键词检索 + RRF 融合。
- 中文检索优化：中文单字/bigram 分词，并对“推荐测试问题”类非答案 chunk 降权。
- 流式问答：`/qa/stream` 使用 SSE 返回 token 和 metrics。
- 大模型接入：配置 `.env` 后使用 DeepSeek；未配置时自动回退本地 demo generator。
- 配额控制：每个租户硬编码每日 token 配额，超限返回清晰错误。
- 演示页面：`GET /` 可上传语料库并直接向大模型提问。

## 快速开始

```powershell
cd C:\Users\33188\Documents\Codex\2026-06-09\a1-ai-3-github-readme-commit-2
pip install -e ".[dev]"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

如果 VS Code 里的 `python` 指向 WindowsApps 占位程序，请使用：

```powershell
& "C:\Users\33188\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

访问：

- 演示页面：<http://127.0.0.1:8000/>
- API 文档：<http://127.0.0.1:8000/docs>
- 租户列表：`GET /tenants`
- 摄取文件：`POST /tenants/{tenant_id}/files`
- 查询摄取状态：`GET /tenants/{tenant_id}/ingestions/{job_id}`
- 混合检索：`POST /tenants/{tenant_id}/search`
- 流式问答：`POST /tenants/{tenant_id}/qa/stream`
- 用量：`GET /tenants/{tenant_id}/usage`

## DeepSeek 配置

项目启动时会自动读取根目录 `.env` 文件。

复制模板：

```powershell
copy .env.example .env
```

在 `.env` 中填写：

```env
DEEPSEEK_API_KEY=sk-你的真实Key
QA_GENERATOR=deepseek
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
TENANT_QA_DATA_DIR=data/tenants
```

`.env` 已被 `.gitignore` 忽略，不会提交到 Git。没有配置真实 `DEEPSEEK_API_KEY` 时，系统会自动使用本地 demo generator。

## 浏览器演示

打开：

<http://127.0.0.1:8000/>

推荐使用测试文件：

```text
samples/tenant_a/demo_knowledge.txt
```

操作顺序：

1. 选择租户 `tenant_a`。
2. 上传 `samples/tenant_a/demo_knowledge.txt`。
3. 设置 `chunk_size=600`，`chunk_overlap=50`，点击“上传并摄取”。
4. 输入问题：`Tenant A 的 token 配额是多少？`
5. 点击“向大模型提问”。
6. 页面会先展示检索到的来源 chunk，再通过 `/qa/stream` 流式显示回答。

预期回答应包含：

```text
Tenant A 每日默认 token 配额为 2000。
```

页面下方 metrics 中，若已配置 DeepSeek，应看到：

```text
模型：deepseek-v4-flash
```



## 工程决策摘要

完整决策见 [docs/DECISIONS.md](docs/DECISIONS.md)。

- 向量方案：使用本地 Hashing Embedding + JSON 向量索引，优先保证离线可跑和租户物理隔离；生产可替换为 pgvector、Qdrant 或 Milvus 的 per-tenant collection。
- Chunking：默认 `chunk_size=500`、`chunk_overlap=80` 字符；中文/英文演示数据都能保留上下文，同时避免单块太大导致召回粗糙。
- 混合检索：向量相似度和 BM25 各取候选后用 RRF 融合，测试覆盖了单一路线不稳定时融合仍召回目标文档。
  
## Swagger 测试

打开：

<http://127.0.0.1:8000/docs>

1. `GET /health`

应返回：

```json
{
  "status": "ok"
}
```

2. `POST /tenants/{tenant_id}/files`

填写：

```text
tenant_id: tenant_a
chunk_size: 600
chunk_overlap: 50
file: samples/tenant_a/demo_knowledge.txt
```

3. `POST /tenants/{tenant_id}/search`

请求体：

```json
{
  "query": "Tenant A 的 token 配额是多少？",
  "top_k": 4
}
```

4. `POST /tenants/{tenant_id}/qa/stream`

请求体：

```json
{
  "question": "Tenant A 的 token 配额是多少？",
  "top_k": 4
}
```

返回内容是 SSE：

```text
event: token
event: metrics
```

`metrics` 包含 `token_usage`、`retrieval_ms`、`generation_ms`、`chunks_used`、`model`。

## 终端 curl 测试

上传：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/tenants/tenant_a/files?chunk_size=600&chunk_overlap=50" `
  -F "file=@samples/tenant_a/demo_knowledge.txt"
```

搜索：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/tenants/tenant_a/search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Tenant A 的 token 配额是多少？\",\"top_k\":4}"
```

流式问答：

```powershell
curl.exe -N -X POST "http://127.0.0.1:8000/tenants/tenant_a/qa/stream" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Tenant A 的 token 配额是多少？\",\"top_k\":4}"
```

## 多租户隔离测试

1. 只向 `tenant_a` 上传 `samples/tenant_a/demo_knowledge.txt`。
2. 用 `tenant_a` 查询 `Tenant A 的 token 配额是多少？`，应能命中 `demo_knowledge.txt` 中的 `2000`。
3. 用 `tenant_b` 查询同样问题，不应返回 `tenant_a` 的内容。

```powershell
curl.exe -X POST "http://127.0.0.1:8000/tenants/tenant_b/search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Tenant A 的 token 配额是多少？\",\"top_k\":4}"
```

## 架构图

```text
Browser / Swagger / curl
  |
  | upload PDF / Markdown / TXT
  v
FastAPI Router
  |
  +--> TenantGuard
  |     - tenant_id 只来自路径
  |     - 校验 tenant_a / tenant_b / tenant_c
  |
  +--> IngestionService
        |
        +--> Parser
        |     - PDF: page
        |     - Markdown/TXT: line
        |
        +--> Configurable Chunker
        |
        +--> HashingEmbedding
        |
        +--> TenantStorage
              |
              +--> data/tenants/{tenant_id}/files
              +--> data/tenants/{tenant_id}/manifest.json
              +--> data/tenants/{tenant_id}/vector_index.json
              +--> data/tenants/{tenant_id}/keyword_index.json

Browser / Swagger / curl
  |
  | POST /tenants/{tenant_id}/qa/stream
  v
FastAPI SSE Endpoint
  |
  +--> QuotaService
  |
  +--> HybridRetriever
  |     |
  |     +--> tenant-local vector search
  |     +--> tenant-local BM25 search
  |     +--> Chinese tokenization + answer-like rerank boost
  |     +--> RRF fusion
  |
  +--> FallbackAnswerGenerator
        |
        +--> DeepSeek /chat/completions stream, if .env has key
        +--> local demo generator, if no key
```

## 测试

```powershell
python -m pytest -q
```

当前覆盖：

- 租户 A/B 数据隔离
- 摄取状态流转和来源信息返回
- 混合检索融合结果
- 中文配额问题优先命中答案 chunk
- SSE 输出 metrics
- 配额超限错误
- 首页演示 UI 可访问
- 测试套件默认不调用真实 DeepSeek，避免消耗 API 配额

## 已知问题

- JSON 索引适合演示，不适合大规模生产。
- 摄取当前为同步执行，生产应拆成队列 worker。
- 真实 token 用量当前是估算值，生产应使用模型供应商返回的 usage。
- DeepSeek 调用失败时会回退本地 generator，生产环境应增加告警和重试策略。
- 当前没有认证系统，生产应从 JWT/OIDC claim 中获取 tenant_id。
