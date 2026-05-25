export type ChunkingStrategy =
  | "adaptive"
  | "semantic_hybrid"
  | "semantic"
  | "markdown_section"
  | "parent_child"
  | "sentence"
  | "sentence_window"
  | "token"
  | "recursive_text";

export type ChunkingPolicyInput = {
  strategy: ChunkingStrategy;
  language: "zh";
  chunk_size: number;
  overlap_ratio: number;
  max_chunk_size: number;
  semantic_buffer_size: number;
  semantic_threshold: number;
  window_size: number;
};

export type ChunkingPolicyField =
  | "chunk_size"
  | "overlap_ratio"
  | "max_chunk_size"
  | "semantic_buffer_size"
  | "semantic_threshold"
  | "window_size";

export const DEFAULT_CHUNKING_POLICY: ChunkingPolicyInput = {
  strategy: "adaptive",
  language: "zh",
  chunk_size: 900,
  overlap_ratio: 0.15,
  max_chunk_size: 1200,
  semantic_buffer_size: 1,
  semantic_threshold: 95,
  window_size: 2,
};

export const CHUNKING_STRATEGY_OPTIONS: { label: string; value: ChunkingStrategy }[] = [
  { value: "adaptive", label: "动态路由 (Adaptive)" },
  { value: "semantic_hybrid", label: "语义混合 (Semantic Hybrid)" },
  { value: "semantic", label: "语义拆分 (Semantic)" },
  { value: "markdown_section", label: "结构切片 (Markdown)" },
  { value: "parent_child", label: "父子分段 (Parent Child)" },
  { value: "sentence_window", label: "句子滑窗 (Sentence Window)" },
  { value: "sentence", label: "句子拆分 (Sentence)" },
  { value: "token", label: "Token拆分 (Token)" },
  { value: "recursive_text", label: "递归文本拆分 (Recursive Text)" },
];

export const CHUNKING_STRATEGY_HELP: Record<ChunkingStrategy, string> = {
  adaptive: "根据文档长度与结构自动选择结构切片、句子滑窗或语义混合策略。",
  semantic_hybrid: "先按语义断点切分，遇到过大的语义块再按句子边界二次切分。",
  semantic: "只根据句子间语义变化寻找断点，适合语义段落清晰的长文本。",
  markdown_section: "优先按 Markdown 标题或表格结构切分，结构段落过大时再二次切分。",
  parent_child: "先生成较大的父块保存完整上下文，再切成较小子块用于检索召回。",
  sentence_window: "按句子边界聚合成块，并为每块保留前后窗口上下文。",
  sentence: "按句子边界组成固定大小的块，适合普通中文文本。",
  token: "按 token 长度硬性切块，适合需要稳定 token 预算的场景。",
  recursive_text: "按段落优先切分，段落过长时按字符长度递归切块。",
};

export const CHUNKING_STRATEGY_FIELDS: Record<ChunkingStrategy, ChunkingPolicyField[]> = {
  adaptive: [
    "chunk_size",
    "overlap_ratio",
    "max_chunk_size",
    "semantic_buffer_size",
    "semantic_threshold",
    "window_size",
  ],
  semantic_hybrid: [
    "chunk_size",
    "overlap_ratio",
    "max_chunk_size",
    "semantic_buffer_size",
    "semantic_threshold",
  ],
  semantic: ["semantic_buffer_size", "semantic_threshold"],
  markdown_section: ["chunk_size", "overlap_ratio", "max_chunk_size"],
  parent_child: ["chunk_size", "overlap_ratio", "max_chunk_size"],
  sentence_window: ["chunk_size", "overlap_ratio", "window_size"],
  sentence: ["chunk_size", "overlap_ratio"],
  token: ["chunk_size", "overlap_ratio"],
  recursive_text: ["chunk_size", "overlap_ratio"],
};

export const CHUNKING_FIELD_LABELS: Record<ChunkingPolicyField, string> = {
  chunk_size: "块长",
  overlap_ratio: "重叠",
  max_chunk_size: "最大块",
  semantic_buffer_size: "Buffer",
  semantic_threshold: "阈值",
  window_size: "窗口",
};

export const CHUNKING_FIELD_HELP: Record<ChunkingPolicyField, string> = {
  chunk_size: "控制每个块的目标长度。数值越小，通常块越多、内容越短。",
  overlap_ratio: "相邻块保留的重叠比例，会换算成 chunk_overlap，帮助减少边界信息丢失。",
  max_chunk_size: "语义块或结构块超过该长度时会二次切分；父子分段中它代表父块目标大小。",
  semantic_buffer_size: "语义断点判断时向前后参考的句子缓冲数量，数值越大越平滑。",
  semantic_threshold: "语义断点阈值。数值越高，通常越不容易断开，块会更长。",
  window_size: "句子滑窗上下文范围，只影响每块 metadata 中的 window_context。",
};
