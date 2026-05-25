CREATE TABLE IF NOT EXISTS eval_datasets (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  kb_id uuid NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  name text NOT NULL,
  description text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_samples (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  dataset_id uuid NOT NULL REFERENCES eval_datasets(id) ON DELETE CASCADE,
  source_message_id uuid REFERENCES chat_messages(id) ON DELETE SET NULL,
  user_input text NOT NULL,
  reference text NOT NULL DEFAULT '',
  expected_context_ids jsonb NOT NULL DEFAULT '[]',
  tags jsonb NOT NULL DEFAULT '[]',
  original_response text NOT NULL DEFAULT '',
  original_citations jsonb NOT NULL DEFAULT '[]',
  original_retrieval_trace jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_runs (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  dataset_id uuid NOT NULL REFERENCES eval_datasets(id) ON DELETE CASCADE,
  kb_id uuid NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  status text NOT NULL DEFAULT 'pending',
  metrics jsonb NOT NULL DEFAULT '{}',
  config jsonb NOT NULL DEFAULT '{}',
  error_message text NOT NULL DEFAULT '',
  completed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_run_results (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  run_id uuid NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
  sample_id uuid NOT NULL REFERENCES eval_samples(id) ON DELETE CASCADE,
  user_input text NOT NULL,
  response text NOT NULL DEFAULT '',
  reference text NOT NULL DEFAULT '',
  retrieved_contexts jsonb NOT NULL DEFAULT '[]',
  citations jsonb NOT NULL DEFAULT '[]',
  retrieval_trace jsonb NOT NULL DEFAULT '{}',
  metrics jsonb NOT NULL DEFAULT '{}',
  reasons jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_eval_datasets_scope ON eval_datasets (tenant_id, kb_id);
CREATE INDEX IF NOT EXISTS idx_eval_samples_dataset ON eval_samples (tenant_id, dataset_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_dataset ON eval_runs (tenant_id, dataset_id);
CREATE INDEX IF NOT EXISTS idx_eval_run_results_run ON eval_run_results (tenant_id, run_id);
