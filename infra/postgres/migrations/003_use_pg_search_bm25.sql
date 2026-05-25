CREATE EXTENSION IF NOT EXISTS pg_search;

DROP INDEX IF EXISTS idx_document_chunks_search;
DROP INDEX IF EXISTS idx_document_chunks_bm25;

ALTER TABLE document_chunks
  DROP COLUMN IF EXISTS search_vector;

DROP TEXT SEARCH CONFIGURATION IF EXISTS public.zhcfg;

-- BM25 负责精确词项召回；tenant/kb/document 进入索引以便权限过滤能尽量下推。
CREATE INDEX idx_document_chunks_bm25
  ON document_chunks USING bm25 (id, (content::pdb.jieba), tenant_id, kb_id, document_id)
  WITH (key_field = 'id');
