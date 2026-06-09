# 工程决策文档

## 目标

交付一个可以演示核心风险控制的企业多租户 AI 问答模块。重点是租户隔离、可观测摄取状态、混合检索、基于语料库的大模型流式问答、配额控制和工程取舍清楚。

本项目不是生产级 RAG 平台，但保留了可以替换生产组件的边界：embedding provider、tenant storage、retriever、answer generator。

## 架构

```text
                         +--------------------+
                         | Browser / Swagger  |
                         +----------+---------+
                                    |
                                    v
                         +--------------------+
                         |      FastAPI       |
                         +----------+---------+
                                    |
                 validates path tenant_id only
                                    |
        +---------------------------+---------------------------+
        |                                                       |
        v                                                       v
+------------------+                                +-------------------+
| IngestionService |                                |   QA SSE Service   |
+--------+---------+                                +---------+---------+
         |                                                    |
         v                                                    v
+------------------+        +-------------------+     +----------------+
| Parser PDF/MD/TXT| -----> | Configurable      |     | QuotaService   |
+------------------+        | Chunker           |     +-------+--------+
                            +---------+---------+             |
                                      |                       v
                                      v               +---------------+
                            +------------------+      | HybridRetriever|
                            | TenantStorage    |      +-------+-------+
                            +---------+--------+              |
                                      |                       v
                         per-tenant physical files   Vector + BM25 + RRF
                                                              |
                                                              v
                                                   +----------------------+
                                                   | FallbackAnswerGenerator|
                                                   +----------+-----------+
                                                              |
                                                   +----------+-----------+
                                                   | DeepSeek stream / demo|
                                                   +----------------------+
```

## 决策 1：向量数据库 / Embedding 方案

选择本地 `HashingEmbedding` 和每租户 JSON 向量索引。

原因：

- 不需要外部 embedding API key，可 clone 后可以立即运行检索。
- embedding 可复现，测试稳定。
- 每个租户是独立索引文件，不依赖 metadata filter 做隔离，能直接展示物理隔离设计。
- 便于替换生产组件：`HashingEmbedding` 可以换成真实 embedding，`TenantStorage` 可以换成 pgvector/Qdrant/Milvus。

没有选择：

- OpenAI/DeepSeek embedding：效果更好，但会增加 key、费用和联网依赖。
- 单 Qdrant/Milvus collection + tenant metadata filter：实现快，但一旦过滤漏写就会跨租召回。
- pgvector 单表：适合中小规模生产，但如果只靠 `tenant_id where` 条件，仍然要依赖 RLS 或独立 schema 才稳。

生产建议：

- 中小客户：PostgreSQL schema per tenant + pgvector + RLS。
- 大客户或强隔离客户：Qdrant/Milvus collection per tenant，配合独立加密 key、独立备份和审计。

## 决策 2：DeepSeek 问答接入

问答生成通过 `FallbackAnswerGenerator` 实现。

- `.env` 中有真实 `DEEPSEEK_API_KEY` 时，调用 DeepSeek OpenAI-compatible `/chat/completions` 流式接口。
- 默认模型为 `deepseek-v4-flash`，可通过 `DEEPSEEK_MODEL` 覆盖。
- 未配置 key 或使用占位 key 时，自动回退本地 demo generator。
- 自动化测试固定走本地 demo generator，避免消耗真实 API 配额。

为什么这样设计：

- 保证本地离线可跑。
- 保证演示时可以真实调用大模型。
- 保证测试稳定，不依赖外部网络和余额。
- 将模型调用隔离在 `app/generator.py`，未来可替换为 OpenAI、私有模型或多模型路由。

## 决策 3：Chunking 参数

默认值：

- `chunk_size=500`
- `chunk_overlap=80`

演示推荐：

- `samples/tenant_a/demo_knowledge.txt` 可使用 `chunk_size=600`、`chunk_overlap=50`。

选择理由：

- 小于 200 字符时，FAQ 型问题召回更尖锐，但答案上下文容易缺失。
- 大于 800 字符时，上下文完整，但检索粒度下降。
- 500/80 对政策、手册、FAQ 类文档较稳。
- 演示文件使用 600/50 可以让“访问配额”段落保持完整，同时避免和“推荐测试问题”段落混在一起。

## 决策 4：混合检索与融合策略

系统实现两路检索：

- 向量检索：hash embedding + cosine similarity。
- 关键词检索：BM25。

融合策略是 Reciprocal Rank Fusion：

```text
score(doc) = sum(1 / (k + rank_i(doc)))
```

当前 `k=60`。RRF 的好处是不用把 cosine 和 BM25 的分数强行归一到同一尺度；只要某个 chunk 在任一路排序靠前，就会获得稳定加分。

后来针对中文问答增加了两个轻量优化：

- 中文分词从“整段中文 token”改为中文单字 + bigram，提升 BM25 对 `配额`、`多少`、`响应` 等词的召回。
- 对“推荐测试问题”这类非答案 chunk 降权，对包含数值、默认值、必须、目标等答案 cue 的 chunk 加权。

原因：

在实际测试中，问题 `Tenant A 的 token 配额是多少？` 曾命中“推荐测试问题”段落，而不是包含 `Tenant A 每日默认 token 配额为 2000` 的答案段。修复后测试覆盖确保答案段排在前面。

## 矛盾约束取舍

### 混合检索延迟 vs 首 token 快

当前做法是检索阶段先完成，然后立即开始 SSE 输出。这样首 token 会等待检索完成，但回答来源稳定，不会出现生成过程中引用变化。

降低延迟的做法：

- 本地索引按租户读取，候选集小。
- top-k 默认较小。
- 向量检索和 BM25 都在同进程内完成。

生产方案：

- query embedding、BM25、rerank 并行化。
- 缓存热门问题的检索结果。
- 仅在保证引用稳定的前提下考虑 speculative generation。

### 数据库过滤 vs 向量索引物理隔离

误用同一个向量索引会导致物理隔离失效,因此本实现不使用共享向量集合。

- 每个租户一个目录。
- 每个租户一个 `vector_index.json`。
- 每个租户一个 `keyword_index.json`。
- Retriever 只打开当前路径租户的索引。
- 请求体中没有可覆盖 tenant 的字段。

生产中可以用 collection per tenant 或 schema per tenant。若必须共享集群，也要使用 RLS、服务端 tenant claim、审计日志和跨租自动化测试。

### Chunk 大小 vs 检索精度

选择中等 chunk。

- 过小：召回片段精确，但回答缺上下文。
- 过大：上下文完整，但检索分辨率下降。
- 中等值配合 top-k=4 更适合 SaaS 政策、手册、FAQ 类知识库。

## 当前测试覆盖

`tests/test_app.py` 覆盖：

- 摄取状态和来源行号。
- tenant A/B 数据隔离。
- SSE token 和 metrics。
- DeepSeek key 存在时的模型选择。
- 占位 key 回退本地 generator。
- 配额超限错误。
- 未知租户拒绝。
- 混合检索同时具备 keyword/vector rank。
- 中文配额问题优先命中包含 `2000` 的答案 chunk。
- 首页演示 UI 可访问。

## 已知问题与下一步

- JSON 索引不可扩展：下一步换 pgvector/Qdrant/Milvus。
- 摄取是同步执行：下一步拆成 job 表 + worker + 重试。
- 当前 token usage 是估算：下一步使用 DeepSeek/OpenAI 返回的真实 usage。
- 没有认证：下一步接 JWT/OIDC，从 token claim 得到 tenant_id。
- 没有租户内权限：下一步支持 RBAC、文档级 ACL 和审计日志。
- DeepSeek fallback 只返回提示并走本地生成：生产中应增加告警、重试和熔断。
