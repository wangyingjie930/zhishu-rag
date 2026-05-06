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
  };
  ingestion_policy: {
    embedding?: { model?: string };
    chunker?: Record<string, unknown>;
  };
  created_at: string;
};

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

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8001/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    throw new Error(await response.text());
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
    ingestion_policy?: Record<string, unknown>;
  },
) {
  return request<KnowledgeBase>("/knowledge-bases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
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

export function sendChat(kbId: string, message: string, sessionId?: string) {
  return request<ChatResponse>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kb_id: kbId, message, session_id: sessionId, top_k: 8 }),
  });
}
