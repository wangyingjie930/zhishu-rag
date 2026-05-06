import {
  AlertCircle,
  CheckCircle2,
  FileText,
  FileUp,
  Loader2,
  MoveRight,
  RefreshCw,
  Scissors,
  Settings2,
  X,
} from "lucide-react";
import type { ChangeEvent } from "react";
import { ChunkingPolicyFields } from "../../components/ChunkingPolicyFields";
import { FancySelect } from "../../components/FancySelect";
import type {
  ChunkPreviewItem,
  DocumentChunkPreview,
  DocumentRecord,
  KnowledgeBase,
} from "../../lib/api";
import type { ChunkingPolicyInput } from "../../lib/chunkingPolicy";
import { useDocumentIngestionWizard } from "./useDocumentIngestionWizard";

type SelectOption = {
  value: string;
  label: string;
  disabled?: boolean;
  hint?: string;
};

type ParentPreviewGroup = {
  index: number;
  tokenCount: number;
  characterCount: number;
  children: ChunkPreviewItem[];
};

const CHILD_SEGMENT_TONES = [
  "tone-blue",
  "tone-green",
  "tone-amber",
  "tone-violet",
  "tone-cyan",
  "tone-rose",
];

type DocumentIngestionWizardProps = {
  activeKb?: KnowledgeBase;
  activeKbId: string;
  documents: DocumentRecord[];
  embeddingModel: string;
  embeddingOptions: SelectOption[];
  initialFile: File | null;
  onEmbeddingModelChange: (model: string) => void;
  onInitialFileConsumed: () => void;
  onNeedKnowledgeBase: () => void;
  onSaved: (
    document: DocumentRecord,
    chunkingPolicy: ChunkingPolicyInput,
    embeddingModel: string,
  ) => void;
};

const PARSER_OPTIONS = [
  { value: "auto", label: "自动解析", hint: "根据文件类型选择解析器" },
  { value: "text", label: "纯文本", hint: "适合 txt、md、csv、json" },
];

export function DocumentIngestionWizard({
  activeKb,
  activeKbId,
  documents,
  embeddingModel,
  embeddingOptions,
  initialFile,
  onEmbeddingModelChange,
  onInitialFileConsumed,
  onNeedKnowledgeBase,
  onSaved,
}: DocumentIngestionWizardProps) {
  const wizard = useDocumentIngestionWizard({
    activeKbId,
    embeddingModel,
    initialFile,
    onInitialFileConsumed,
    onSaved,
  });

  function onPickFile(event: ChangeEvent<HTMLInputElement>) {
    wizard.chooseFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  }

  if (!activeKbId) {
    return (
      <div className="ingestion-empty">
        <FileText size={24} />
        <strong>请先选择知识库</strong>
        <p>文档会保存到当前知识库，并按提交的分段参数写入数据库。</p>
        <button className="btn-primary" onClick={onNeedKnowledgeBase} type="button">
          选择知识库
        </button>
      </div>
    );
  }

  return (
    <div className="ingestion-wizard">
      <div className="ingestion-head">
        <div>
          <p className="panel-section-label">文档导入</p>
          <strong>{activeKb?.name ?? "当前知识库"}</strong>
        </div>
        {wizard.selectedFile && (
          <button className="icon-text-btn" onClick={wizard.resetWizard} type="button">
            <X size={13} />
            重置
          </button>
        )}
      </div>

      <div className="ingestion-steps" aria-label="导入步骤">
        <StepPill
          active={wizard.step === "source"}
          done={Boolean(wizard.selectedFile)}
          label="数据源"
        />
        <MoveRight size={13} />
        <StepPill
          active={wizard.step === "processing"}
          done={Boolean(wizard.preview)}
          label="分段清洗"
        />
        <MoveRight size={13} />
        <StepPill active={false} done={false} label="保存" />
      </div>

      <div className="ingestion-workspace">
        <div className="ingestion-config-pane">
          {wizard.step === "source" ? (
            <section className="ingestion-card">
              <div className="ingestion-card-title">
                <FileUp size={15} />
                <span>选择数据源</span>
              </div>
              <label className="source-dropzone">
                <FileUp size={28} />
                <strong>{wizard.selectedFile ? wizard.selectedFile.name : "上传文件"}</strong>
                <span>
                  {wizard.selectedFile
                    ? `${formatFileSize(wizard.selectedFile.size)} · ${
                        wizard.selectedFile.type || "未知类型"
                      }`
                    : "支持拖拽到页面或点击选择本地文件"}
                </span>
                <input type="file" onChange={onPickFile} />
              </label>
              <button
                className="btn-primary"
                disabled={!wizard.selectedFile}
                onClick={wizard.goToProcessing}
                type="button"
              >
                下一步
                <MoveRight size={14} />
              </button>
            </section>
          ) : (
            <section className="ingestion-card">
              <div className="ingestion-card-title">
                <Settings2 size={15} />
                <span>文本分段与清洗</span>
              </div>
              <div className="selected-source-row">
                <FileText size={14} />
                <span>{wizard.selectedFile?.name}</span>
                <button onClick={() => wizard.setStep("source")} type="button">
                  更换
                </button>
              </div>
              <FancySelect
                onChange={wizard.setParser}
                options={PARSER_OPTIONS}
                value={wizard.parser}
              />
              <ChunkingPolicyFields
                onChange={wizard.setChunkingPolicy}
                value={wizard.chunkingPolicy}
              />
              <div className="upload-model-row ingestion-model-row">
                <Scissors size={13} />
                <FancySelect
                  buttonClassName="embedding-select-trigger"
                  className="embedding-select-wrap"
                  menuClassName="embedding-select-menu"
                  onChange={onEmbeddingModelChange}
                  options={embeddingOptions}
                  value={embeddingModel}
                />
              </div>
              {wizard.error && (
                <div className="ingestion-error">
                  <AlertCircle size={14} />
                  <span>{wizard.error}</span>
                </div>
              )}
              <div className="ingestion-actions">
                <button
                  className="btn-secondary"
                  disabled={!wizard.canPreview}
                  onClick={wizard.previewChunks}
                  type="button"
                >
                  {wizard.previewing ? (
                    <Loader2 className="spin" size={14} />
                  ) : (
                    <RefreshCw size={14} />
                  )}
                  预览块
                </button>
                <button
                  className="btn-primary"
                  disabled={!wizard.selectedFile || wizard.saving}
                  onClick={wizard.saveDocument}
                  type="button"
                >
                  {wizard.saving ? (
                    <Loader2 className="spin" size={14} />
                  ) : (
                    <CheckCircle2 size={14} />
                  )}
                  保存
                </button>
              </div>
            </section>
          )}

          <RecentDocuments documents={documents} />
        </div>

        <aside className="ingestion-preview-pane">
          {wizard.preview ? (
            <>
              <div className="preview-summary">
                <span>预估块数</span>
                <strong>{wizard.preview.total_chunks}</strong>
                <small>
                  清洗后 {wizard.preview.clean_text_length} 字符
                  {wizard.preview.truncated ? " · 仅展示前 80 块" : ""}
                </small>
              </div>
              <div className="preview-policy-line">
                {String(wizard.preview.chunking_policy.strategy)} · 块长{" "}
                {String(wizard.preview.chunking_policy.chunk_size)} · 重叠{" "}
                {String(wizard.preview.chunking_policy.chunk_overlap)} · 窗口{" "}
                {String(wizard.preview.chunking_policy.window_size)}
              </div>
              {isParentChildPreview(wizard.preview) ? (
                <ParentChildPreview preview={wizard.preview} />
              ) : (
                <StandardChunkPreview chunks={wizard.preview.chunks} />
              )}
            </>
          ) : (
            <div className="preview-empty">
              <Scissors size={24} />
              <strong>预览会显示在这里</strong>
              <p>点击“预览块”后可以检查块数量、每个块的内容和长度，再决定是否保存。</p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

function StandardChunkPreview({ chunks }: { chunks: ChunkPreviewItem[] }) {
  return (
    <div className="preview-chunk-list">
      {chunks.map((chunk) => {
        const windowContext = readPreviewWindowContext(chunk.metadata);

        return (
          <article className="preview-chunk" key={chunk.index}>
            <div className="preview-chunk-head">
              <strong>Chunk {chunk.index + 1}</strong>
              <span>
                {chunk.character_count} 字符 · {chunk.token_count} tokens
              </span>
            </div>
            <p>{chunk.content}</p>
            {windowContext && windowContext !== chunk.content ? (
              <div className="preview-window-context">
                <span>窗口上下文</span>
                <p>{windowContext}</p>
              </div>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}

function ParentChildPreview({ preview }: { preview: DocumentChunkPreview }) {
  const groups = buildParentPreviewGroups(preview.chunks);

  return (
    <div className="preview-parent-list">
      {groups.map((group) => (
        <article className="preview-parent-chunk" key={group.index}>
          <div className="preview-chunk-head">
            <strong>Chunk {group.index + 1}</strong>
            <span>
              {group.characterCount} 字符 · {group.tokenCount} tokens · {group.children.length} 子块
            </span>
          </div>
          <div className="preview-child-segments">
            {group.children.map((child, childIndex) => (
              <section
                className={`preview-child-segment ${
                  CHILD_SEGMENT_TONES[childIndex % CHILD_SEGMENT_TONES.length]
                }`}
                key={child.index}
              >
                <div className="preview-child-segment-head">
                  <span>子块 {readPreviewChildIndex(child.metadata) + 1}</span>
                  <small>
                    {child.character_count} 字符 · {child.token_count} tokens
                  </small>
                </div>
                <p>{child.content}</p>
              </section>
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}

function StepPill({ active, done, label }: { active: boolean; done: boolean; label: string }) {
  return (
    <span className={`step-pill ${active ? "active" : ""} ${done ? "done" : ""}`}>
      {done ? <CheckCircle2 size={12} /> : null}
      {label}
    </span>
  );
}

function RecentDocuments({ documents }: { documents: DocumentRecord[] }) {
  if (!documents.length) {
    return (
      <div className="panel-empty compact-empty">
        <FileText size={18} />
        <span>暂无文档</span>
      </div>
    );
  }

  return (
    <section className="recent-documents">
      <p className="panel-section-label">最近文档</p>
      <div className="doc-list">
        {documents.slice(0, 6).map((doc) => (
          <div className="doc-item" key={doc.id}>
            <div className="doc-item-icon">
              <FileText size={13} />
            </div>
            <div className="doc-item-info">
              <span className="doc-item-name">{doc.filename}</span>
              <span className="doc-item-meta">
                {doc.parser} ·{" "}
                {new Date(doc.created_at).toLocaleDateString("zh-CN", {
                  month: "short",
                  day: "numeric",
                })}
              </span>
            </div>
            <span className={`badge ${doc.status}`}>{doc.status}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function formatFileSize(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function readPreviewWindowContext(metadata: Record<string, unknown>) {
  const context = metadata.window_context;
  return typeof context === "string" ? context.trim() : "";
}

function isParentChildPreview(preview: DocumentChunkPreview) {
  return preview.chunking_policy.strategy === "parent_child";
}

function buildParentPreviewGroups(chunks: ChunkPreviewItem[]) {
  const groups = new Map<number, ParentPreviewGroup>();

  for (const chunk of chunks) {
    const parent = readPreviewParent(chunk.metadata);
    if (!parent) {
      groups.set(chunk.index, {
        index: chunk.index,
        tokenCount: chunk.token_count,
        characterCount: chunk.character_count,
        children: [chunk],
      });
      continue;
    }

    const group = groups.get(parent.index);
    if (group) {
      group.children.push(chunk);
    } else {
      groups.set(parent.index, {
        index: parent.index,
        tokenCount: parent.tokenCount,
        characterCount: parent.characterCount,
        children: [chunk],
      });
    }
  }

  return Array.from(groups.values()).sort((left, right) => left.index - right.index);
}

function readPreviewParent(metadata: Record<string, unknown>) {
  const text = metadata.parent_text;
  const index = metadata.parent_index;
  if (typeof text !== "string" || typeof index !== "number") return null;

  const tokenCount = metadata.parent_token_count;
  const characterCount = metadata.parent_character_count;
  return {
    index,
    text: text.trim(),
    tokenCount: typeof tokenCount === "number" ? tokenCount : 0,
    characterCount: typeof characterCount === "number" ? characterCount : text.length,
  };
}

function readPreviewChildIndex(metadata: Record<string, unknown>) {
  const index = metadata.child_index;
  return typeof index === "number" ? index : 0;
}
