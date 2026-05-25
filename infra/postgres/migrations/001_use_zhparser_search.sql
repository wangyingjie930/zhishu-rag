CREATE EXTENSION IF NOT EXISTS zhparser;

DROP INDEX IF EXISTS idx_document_chunks_search;

ALTER TABLE document_chunks
  DROP COLUMN IF EXISTS search_vector;

DROP TEXT SEARCH CONFIGURATION IF EXISTS public.zhcfg;

CREATE TEXT SEARCH CONFIGURATION public.zhcfg (PARSER = zhparser);

ALTER TEXT SEARCH CONFIGURATION public.zhcfg
  ADD MAPPING FOR a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,v,x,y,z
  WITH simple;

ALTER TABLE document_chunks
  ADD COLUMN search_vector tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('public.zhcfg', coalesce(content, '')), 'A')
  ) STORED;

CREATE INDEX IF NOT EXISTS idx_document_chunks_search
  ON document_chunks USING gin (search_vector);
