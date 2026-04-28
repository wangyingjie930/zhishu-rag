# Retrieval Contract

The API returns citations with stable chunk identifiers, source document identifiers, filename,
content preview, score and metadata. Retrieval services may swap implementations as long as this
contract stays stable.

## Required Retrieval Stages

1. Normalize and optionally rewrite the query.
2. Apply tenant, knowledge base and document ACL filters before scoring.
3. Retrieve vector and lexical candidates independently.
4. Fuse candidates with reciprocal rank fusion or a configured policy.
5. Optionally rerank with a cross-encoder or LLM reranker.
6. Return citations and a retrieval trace for auditability.

