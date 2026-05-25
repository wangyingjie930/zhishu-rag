import {
  type ChunkingPolicyInput,
  DEFAULT_CHUNKING_POLICY,
} from "./chunkingPolicy";

export function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

export function readChunkingPolicyFromIngestionPolicy(value: unknown): ChunkingPolicyInput {
  const ingestionPolicy = asRecord(value);
  const chunker = asRecord(ingestionPolicy.chunker);

  return {
    ...DEFAULT_CHUNKING_POLICY,
    strategy: String(
      chunker.strategy ?? DEFAULT_CHUNKING_POLICY.strategy,
    ) as ChunkingPolicyInput["strategy"],
    language: "zh",
    chunk_size: readNumber(chunker.chunk_size, DEFAULT_CHUNKING_POLICY.chunk_size),
    overlap_ratio: readNumber(chunker.overlap_ratio, DEFAULT_CHUNKING_POLICY.overlap_ratio),
    max_chunk_size: readNumber(chunker.max_chunk_size, DEFAULT_CHUNKING_POLICY.max_chunk_size),
    semantic_buffer_size: readNumber(
      chunker.semantic_buffer_size,
      DEFAULT_CHUNKING_POLICY.semantic_buffer_size,
    ),
    semantic_threshold: readNumber(
      chunker.semantic_threshold,
      DEFAULT_CHUNKING_POLICY.semantic_threshold,
    ),
    window_size: readNumber(chunker.window_size, DEFAULT_CHUNKING_POLICY.window_size),
  };
}

export function readEmbeddingModelFromIngestionPolicy(value: unknown) {
  const ingestionPolicy = asRecord(value);
  const embedding = asRecord(ingestionPolicy.embedding);
  return typeof embedding.model === "string" ? embedding.model : "";
}

export function readChunkingPolicyFromDocumentMetadata(value: unknown): ChunkingPolicyInput {
  const metadata = asRecord(value);
  return readChunkingPolicyFromIngestionPolicy(metadata.ingestion_policy);
}

export function readEmbeddingModelFromDocumentMetadata(value: unknown) {
  const metadata = asRecord(value);
  return readEmbeddingModelFromIngestionPolicy(metadata.ingestion_policy);
}

function readNumber(value: unknown, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}
