# 企业级知识库 RAG 架构

## 目标

该项目按生产系统边界设计：上传文档不是直接塞进向量库，而是经过解析、治理、切分、嵌入、索引、检索、回答、反馈和审计的完整链路。每个环节都保留策略接口，便于接入不同业务线、模型网关和安全要求。

## 分层

- `api/v1`：HTTP 契约、参数校验、响应模型。
- `services/ingestion`：解析、清洗、分块、嵌入、索引流水线。
- `services/retrieval`：向量召回、关键词召回、融合、重排预留。
- `services/llm`：大模型网关、提示词模板、引用约束预留。
- `services/security`：认证、租户、RBAC/ABAC、数据权限预留。
- `db`：ORM 模型、连接池和事务边界。
- `infra`：Postgres/pgvector、对象存储和初始化 SQL。

## 大厂生产常见能力映射

- 多租户隔离：所有核心表包含 `tenant_id`，后续可加 PostgreSQL Row Level Security。
- 知识库和权限：知识库独立配置检索策略，文档和 chunk 都在 tenant/kb 范围内过滤。
- 文档治理：保留 `metadata`、`checksum`、`status`、`parser`、错误信息，方便去重、回滚和重建索引。
- 异步化：当前演示同步入库，`workers` 目录预留 Celery/Arq/Temporal，用于大文件、OCR、批量重建。
- 混合检索：pgvector 做向量相似度，ParadeDB/pg_search BM25 做关键词召回，RRF 融合。
- 可替换 lexical retriever：如果后续要外置搜索引擎，可用 OpenSearch 替换 BM25 层，API 和数据契约不变。
- 可观测性：`retrieval_trace` 记录召回数量、融合策略、重排策略，便于调优和排障。
- 反馈闭环：`feedback` 表用于收集答案质量，后续驱动评测集、提示词优化和索引策略调参。
- 审计合规：`audit_logs` 表用于记录上传、删除、权限变更、问答访问等行为。

## 文档处理策略

生产环境建议按来源和类型选择策略：

- Office/PDF：使用结构化解析，保留标题、页码、表格、图片说明和层级路径。
- 扫描件：OCR 后做版面恢复，区分页眉页脚和正文。
- HTML/Confluence/Notion：保留 DOM 层级、链接、更新时间和作者。
- 代码仓库：按语言 AST 或符号粒度切分，而不是按固定长度。
- 表格：按 sheet/table/row group 建 chunk，并把表头注入上下文。

## 检索与回答链路

1. 根据用户问题和会话上下文生成检索查询。
2. 先执行租户、知识库、ACL 过滤。
3. 向量召回语义相关片段，关键词召回精确术语片段。
4. 使用 RRF 融合结果，必要时接入 reranker。
5. 组装 prompt，要求回答必须绑定引用。
6. 返回答案、引用、trace，并记录消息。

## 下一步建议

- 接入真实 embedding 和 chat model。
- 接入对象存储 SDK，并把本地 `uploads` 替换为 MinIO/S3。
- 增加 Alembic 迁移和 CI。
- 为检索质量建立 golden set，持续评估 recall、MRR、faithfulness。
- 把索引重建改成后台任务，增加进度、取消和重试。
- 增加知识库成员管理和 API key 管理。
