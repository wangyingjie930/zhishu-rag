# Enterprise Knowledge RAG

企业级知识库 RAG 全栈项目骨架，包含 Python/FastAPI 后端、React 前端、Postgres + pgvector 向量索引、Postgres 全文检索索引、混合检索、文档解析流水线、多租户、权限、审计、反馈和可观测性预留。

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
  postgres/init.sql       pgvector、全文索引、核心表结构
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

## 当前能力

- 企业文档上传、落库、解析、清洗、分块、嵌入、索引。
- 预留解析策略、清洗策略、分块策略、嵌入模型和重排模型扩展点。
- 对话式 RAG：查询改写预留、混合检索、RRF 融合、上下文组装、引用返回。
- 多租户、知识库、ACL、审计日志、反馈、会话、消息、检索 trace 的数据边界。
- Postgres/pgvector 存储向量，Postgres `tsvector` + GIN 做关键词检索；如需严格 BM25，可将 lexical retriever 替换为 ParadeDB/pg_search，接口已隔离在 retrieval 服务层。
