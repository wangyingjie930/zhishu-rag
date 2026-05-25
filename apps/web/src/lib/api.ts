import type { ChunkingPolicyInput } from "./chunkingPolicy";

export type KnowledgeBase = {
  id: string;
  name: string;
  description: string;
  visibility: string;
  retrieval_policy: {
    top_k: number;
    vector_weight: number;
    keyword_weight: number;
    reranker: string;
    score_threshold?: number;
  };
  ingestion_policy: {
    embedding?: { model?: string };
    chunker?: Record<string, unknown>;
  };
  created_at: string;
};

export type RetrievalPolicyInput = KnowledgeBase["retrieval_policy"];

export type EmbeddingModelOption = {
  id: string;
  label: string;
  provider: string;
  model: string;
  dimensions: number;
  enabled: boolean;
  reason: string;
};

export type DocumentRecord = {
  id: string;
  kb_id: string;
  filename: string;
  mime_type: string;
  status: "pending" | "processing" | "indexed" | "failed";
  parser: string;
  metadata: Record<string, unknown>;
  error_message?: string;
  created_at: string;
};

export type ChunkPreviewItem = {
  index: number;
  content: string;
  token_count: number;
  character_count: number;
  metadata: Record<string, unknown>;
};

export type DocumentChunkPreview = {
  filename: string;
  mime_type: string;
  parser: string;
  chunking_policy: Record<string, unknown>;
  clean_text_length: number;
  total_chunks: number;
  chunks: ChunkPreviewItem[];
  truncated: boolean;
};

export type Citation = {
  chunk_id: string;
  document_id: string;
  filename: string;
  score: number;
  content: string;
  metadata: Record<string, unknown>;
};

export type ChatResponse = {
  session_id: string;
  answer: string;
  citations: Citation[];
  retrieval_trace: Record<string, unknown>;
};

export type EvalCandidate = {
  id: string;
  session_id: string;
  user_message_id: string;
  assistant_message_id: string;
  user_input: string;
  response: string;
  citations: Citation[];
  retrieval_trace: Record<string, unknown>;
  created_at: string;
};

export type EvalDataset = {
  id: string;
  kb_id: string;
  name: string;
  description: string;
  sample_count: number;
  created_at: string;
};

export type EvalSample = {
  id: string;
  dataset_id: string;
  source_message_id?: string | null;
  user_input: string;
  reference: string;
  expected_context_ids: string[];
  tags: string[];
  original_response: string;
  original_citations: Citation[];
  original_retrieval_trace: Record<string, unknown>;
  created_at: string;
};

export type EvalMetricMap = Record<string, number | null | undefined>;

export type EvalRunResult = {
  id: string;
  run_id: string;
  sample_id: string;
  user_input: string;
  response: string;
  reference: string;
  retrieved_contexts: string[];
  citations: Citation[];
  retrieval_trace: Record<string, unknown>;
  metrics: EvalMetricMap;
  reasons: Record<string, string>;
  created_at: string;
};

export type EvalRun = {
  id: string;
  dataset_id: string;
  kb_id: string;
  status: "pending" | "running" | "completed" | "failed";
  metrics: EvalMetricMap;
  config: Record<string, unknown>;
  error_message: string;
  created_at: string;
  completed_at?: string | null;
  results: EvalRunResult[];
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8001/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export function listKnowledgeBases() {
  return request<KnowledgeBase[]>("/knowledge-bases");
}

export function listEmbeddingModels() {
  return request<EmbeddingModelOption[]>("/embedding-models");
}

export function createKnowledgeBase(
  payload: Pick<KnowledgeBase, "name" | "description" | "visibility"> & {
    retrieval_policy?: RetrievalPolicyInput;
  },
) {
  return request<KnowledgeBase>("/knowledge-bases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function updateKnowledgeBaseRetrievalPolicy(
  kbId: string,
  retrievalPolicy: RetrievalPolicyInput,
) {
  return request<KnowledgeBase>(`/knowledge-bases/${kbId}/retrieval-policy`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(retrievalPolicy),
  });
}

export function deleteKnowledgeBase(kbId: string) {
  return request<void>(`/knowledge-bases/${kbId}`, {
    method: "DELETE",
  });
}

export function listDocuments(kbId: string) {
  return request<DocumentRecord[]>(`/documents?kb_id=${kbId}`);
}

function buildDocumentFormData(
  kbId: string,
  file: File,
  parser: string,
  embeddingModel: string,
  chunkingPolicy?: ChunkingPolicyInput,
) {
  const body = new FormData();
  body.set("kb_id", kbId);
  body.set("parser", parser);
  if (embeddingModel) body.set("embedding_model", embeddingModel);
  if (chunkingPolicy) body.set("chunking_policy", JSON.stringify(chunkingPolicy));
  body.set("file", file);
  return body;
}

export function previewDocumentChunks(
  kbId: string,
  file: File,
  parser = "auto",
  embeddingModel = "",
  chunkingPolicy?: ChunkingPolicyInput,
) {
  return request<DocumentChunkPreview>("/documents/preview", {
    method: "POST",
    body: buildDocumentFormData(kbId, file, parser, embeddingModel, chunkingPolicy),
  });
}

export function uploadDocument(
  kbId: string,
  file: File,
  parser = "auto",
  embeddingModel = "",
  chunkingPolicy?: ChunkingPolicyInput,
) {
  return request<DocumentRecord>("/documents/upload", {
    method: "POST",
    body: buildDocumentFormData(kbId, file, parser, embeddingModel, chunkingPolicy),
  });
}

export function deleteDocument(documentId: string) {
  return request<void>(`/documents/${documentId}`, {
    method: "DELETE",
  });
}

export function previewDocumentReindex(
  documentId: string,
  parser = "auto",
  embeddingModel = "",
  chunkingPolicy?: ChunkingPolicyInput,
) {
  return request<DocumentChunkPreview>(`/documents/${documentId}/preview-reindex`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      parser,
      embedding_model: embeddingModel,
      chunking_policy: chunkingPolicy ?? {},
    }),
  });
}

export function reindexDocument(
  documentId: string,
  parser = "auto",
  embeddingModel = "",
  chunkingPolicy?: ChunkingPolicyInput,
) {
  return request<DocumentRecord>(`/documents/${documentId}/reindex`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      parser,
      embedding_model: embeddingModel,
      chunking_policy: chunkingPolicy ?? {},
    }),
  });
}

export function sendChat(
  kbId: string,
  message: string,
  sessionId?: string,
  topK = 8,
  hydeEnabled = false,
  queryExpansionEnabled = false,
) {
  return request<ChatResponse>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      kb_id: kbId,
      message,
      session_id: sessionId,
      top_k: topK,
      hyde_enabled: hydeEnabled,
      query_expansion_enabled: queryExpansionEnabled,
    }),
  });
}

export function listEvalCandidates(kbId: string) {
  return request<EvalCandidate[]>(`/eval/candidates?kb_id=${kbId}`);
}

export function listEvalDatasets(kbId: string) {
  return request<EvalDataset[]>(`/eval/datasets?kb_id=${kbId}`);
}

export function createEvalDataset(kbId: string, name: string, description = "") {
  return request<EvalDataset>("/eval/datasets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kb_id: kbId, name, description }),
  });
}

export function deleteEvalDataset(datasetId: string) {
  return request<void>(`/eval/datasets/${datasetId}`, {
    method: "DELETE",
  });
}

export function listEvalSamples(datasetId: string) {
  return request<EvalSample[]>(`/eval/datasets/${datasetId}/samples`);
}

export function addEvalSample(
  datasetId: string,
  payload: {
    source_message_id?: string | null;
    user_input: string;
    reference: string;
    expected_context_ids: string[];
    tags: string[];
    original_response: string;
    original_citations: Citation[];
    original_retrieval_trace: Record<string, unknown>;
  },
) {
  return request<EvalSample>(`/eval/datasets/${datasetId}/samples`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function updateEvalSample(
  datasetId: string,
  sampleId: string,
  payload: Partial<Pick<EvalSample, "reference" | "expected_context_ids" | "tags">>,
) {
  return request<EvalSample>(`/eval/datasets/${datasetId}/samples/${sampleId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function createEvalRun(datasetId: string, queryExpansionEnabled = false) {
  return request<EvalRun>("/eval/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: datasetId,
      query_expansion_enabled: queryExpansionEnabled,
    }),
  });
}

export function getEvalRun(runId: string) {
  return request<EvalRun>(`/eval/runs/${runId}`);
}
