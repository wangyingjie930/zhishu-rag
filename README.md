# Enterprise Knowledge RAG

企业级知识库 RAG 全栈项目骨架，包含 Python/FastAPI 后端、React 前端、Postgres + pgvector 向量索引、ParadeDB/pg_search BM25 关键词索引、混合检索、文档解析流水线、多租户、权限、审计、反馈和可观测性预留。

## 架构

```text
apps/
  api/                    FastAPI 后端
    src/rag_platform/
      api/v1/             HTTP API 层
      core/               配置、日志、运行时能力
      db/                 数据库连接和 ORM 模型
      services/           业务服务：ingestion / retrieval / llm / security
      workers/            异步任务预留
  web/                    React + Vite 前端工作台
infra/
  postgres/               pgvector、pg_search BM25、核心表结构
docs/
  architecture.md         生产级设计说明
```

## 快速启动

1. 复制环境变量：

```bash
cp .env.example .env
```

2. 启动基础设施：

```bash
docker compose up -d postgres minio
```

已有 `postgres-data` volume 的环境在升级 BM25 关键词检索时，需要先重建 Postgres 镜像，
再执行一次迁移 SQL：

```bash
docker compose build postgres
docker compose up -d postgres
docker compose exec -T postgres psql -U rag -d rag < infra/postgres/migrations/003_use_pg_search_bm25.sql
```

3. 启动后端：

```bash
cd apps/api
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src uvicorn rag_platform.main:app --host 127.0.0.1 --port 8001
```

4. 启动前端：

```bash
cd apps/web
npm install
npm run dev
```

前端默认访问 `http://localhost:5173`；如果端口被占用，Vite 会自动切到 `5174` 等端口。后端 OpenAPI 文档在 `http://localhost:8001/docs`。

## Langfuse 可观测

配置后端环境变量即可把聊天和评测链路上传到 Langfuse。未开启时不影响本地 trace 和现有业务流程。

```bash
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=local
LANGFUSE_SAMPLE_RATE=1
```

上传结构：

- `rag_chat`：一次聊天请求，包含检索和回答。
- `rag_eval_sample`：一次评测样本，包含检索、回答和指标分数。
- `retrieval`：检索主链路。
- `query_transform`、`vector_recall`、`keyword_recall`、`fusion_rrf`、`multi_query_hybrid_recall`：检索阶段 observation。
- `llm_answer`：回答模型 generation。

## 当前能力

- 企业文档上传、落库、解析、清洗、分块、嵌入、索引。
- 预留解析策略、清洗策略、分块策略、嵌入模型和重排模型扩展点。
- 对话式 RAG：查询改写预留、混合检索、RRF 融合、上下文组装、引用返回。
- 多租户、知识库、ACL、审计日志、反馈、会话、消息、检索 trace 的数据边界。
- Postgres/pgvector HNSW 存储和召回向量，ParadeDB/pg_search BM25 负责关键词召回，RRF 融合两路结果。
