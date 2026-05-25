import {
  type ChangeEvent,
  type DragEvent as ReactDragEvent,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AlertCircle,
  BarChart3,
  BookOpen,
  Bot,
  ChevronDown,
  ChevronRight,
  Database,
  FileText,
  FileUp,
  Key,
  Loader2,
  Menu,
  MessageSquare,
  Plus,
  RefreshCw,
  Save,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Telescope,
  Trash2,
  X,
} from "lucide-react";
import {
  type ChatResponse,
  type DocumentRecord,
  type EmbeddingModelOption,
  type KnowledgeBase,
  type RetrievalPolicyInput,
  createKnowledgeBase,
  deleteKnowledgeBase,
  listEmbeddingModels,
  listDocuments,
  listKnowledgeBases,
  sendChat,
  updateKnowledgeBaseRetrievalPolicy,
} from "../lib/api";
import type { ChunkingPolicyInput } from "../lib/chunkingPolicy";
import { DocumentIngestionWizard } from "../features/documentIngestion/DocumentIngestionWizard";
import { EvaluationWorkbench } from "../features/evaluation/EvaluationWorkbench";

type ChatTurn = {
  role: "user" | "assistant";
  content: string;
  citations?: ChatResponse["citations"];
  retrievalTrace?: ChatResponse["retrieval_trace"];
};

type PanelTab = "kb" | "docs" | "governance";
type WorkspaceMode = "chat" | "eval";

const SUGGESTED_QUESTIONS = [
  "产品的核心功能有哪些？",
  "如何申请相关权限？",
  "查看最新流程规范",
];

const DEFAULT_EMBEDDING_MODEL = "";
const DEFAULT_RETRIEVAL_POLICY: RetrievalPolicyInput = {
  top_k: 3,
  vector_weight: 0.7,
  keyword_weight: 0.3,
  reranker: "none",
  score_threshold: 0,
};

function normalizeRetrievalPolicy(policy?: Partial<RetrievalPolicyInput>): RetrievalPolicyInput {
  const vectorWeight = Number(policy?.vector_weight ?? DEFAULT_RETRIEVAL_POLICY.vector_weight);
  return {
    ...DEFAULT_RETRIEVAL_POLICY,
    ...policy,
    top_k: Number(policy?.top_k ?? DEFAULT_RETRIEVAL_POLICY.top_k),
    vector_weight: vectorWeight,
    keyword_weight: Number((1 - vectorWeight).toFixed(2)),
    reranker: policy?.reranker ?? DEFAULT_RETRIEVAL_POLICY.reranker,
    score_threshold: Number(policy?.score_threshold ?? DEFAULT_RETRIEVAL_POLICY.score_threshold),
  };
}

type RetrievalSettingsFieldsProps = {
  namePrefix: string;
  policy: RetrievalPolicyInput;
  onChange: (patch: Partial<RetrievalPolicyInput>) => void;
};

function RetrievalSettingsFields({ namePrefix, policy, onChange }: RetrievalSettingsFieldsProps) {
  const normalizedPolicy = normalizeRetrievalPolicy(policy);

  return (
    <div className="retrieval-settings">
      <div className="retrieval-settings-head">
        <div>
          <strong>混合检索</strong>
          <span>语义检索和关键词检索并行召回后融合排序</span>
        </div>
        <span className="recommend-badge">推荐</span>
      </div>

      <div className="retrieval-mode-grid">
        <label className="retrieval-mode-card active">
          <input checked name={`${namePrefix}_retrieval_mode`} readOnly type="radio" value="weighted" />
          <Database size={15} />
          <span>
            <strong>权重设置</strong>
            <small>调节语义和关键词匹配的占比</small>
          </span>
        </label>
        <label className="retrieval-mode-card">
          <input
            checked={normalizedPolicy.reranker !== "none"}
            name={`${namePrefix}_reranker_enabled`}
            onChange={(event) =>
              onChange({
                reranker: event.target.checked ? "default" : "none",
              })
            }
            type="checkbox"
          />
          <Search size={15} />
          <span>
            <strong>Rerank 模型</strong>
            <small>创建后可接入模型重排</small>
          </span>
        </label>
      </div>

      <div className="retrieval-weight-control">
        <input
          aria-label="语义权重"
          max="1"
          min="0"
          onChange={(event) => {
            const vectorWeight = Number(event.target.value);
            onChange({
              vector_weight: vectorWeight,
              keyword_weight: Number((1 - vectorWeight).toFixed(2)),
            });
          }}
          step="0.05"
          type="range"
          value={normalizedPolicy.vector_weight}
        />
        <div className="retrieval-weight-labels">
          <span>语义 {normalizedPolicy.vector_weight.toFixed(2)}</span>
          <span>关键词 {normalizedPolicy.keyword_weight.toFixed(2)}</span>
        </div>
      </div>

      <div className="retrieval-number-grid">
        <label className="compact-field">
          <span>Top K</span>
          <input
            max="30"
            min="1"
            onChange={(event) => onChange({ top_k: Number(event.target.value) })}
            type="number"
            value={normalizedPolicy.top_k}
          />
        </label>
        <label className="compact-field">
          <span>Score 阈值</span>
          <input
            max="1"
            min="0"
            onChange={(event) => onChange({ score_threshold: Number(event.target.value) })}
            step="0.05"
            type="number"
            value={normalizedPolicy.score_threshold ?? 0}
          />
        </label>
      </div>
    </div>
  );
}

function getTraceString(trace: ChatResponse["retrieval_trace"] | undefined, key: string) {
  const value = trace?.[key];
  return typeof value === "string" ? value : "";
}

function getHydeStatusLabel(status: string) {
  if (status === "applied") return "已触发";
  if (status === "fallback") return "已回退";
  if (status === "empty") return "无扩展";
  return status || "未知";
}

function HyDETracePanel({ trace }: { trace?: ChatResponse["retrieval_trace"] }) {
  const hydeEnabled = trace?.hyde_enabled === true || trace?.query_transform === "hyde";
  if (!hydeEnabled) return null;

  const status = getTraceString(trace, "hyde_status");
  const generatedText =
    getTraceString(trace, "hyde_hypothetical_document") ||
    getTraceString(trace, "hyde_hypothetical_document_preview");
  const errorText = getTraceString(trace, "hyde_error");
  const embeddingTextCount =
    typeof trace?.hyde_embedding_text_count === "number" ? trace.hyde_embedding_text_count : 0;

  return (
    <div className={`hyde-trace ${status === "applied" ? "applied" : "fallback"}`}>
      <div className="hyde-trace-head">
        <span>
          <Sparkles size={12} />
          HyDE 查询扩展
        </span>
        <strong>{getHydeStatusLabel(status)}</strong>
      </div>
      {generatedText ? (
        <p>{generatedText}</p>
      ) : (
        <p>{errorText || "没有生成假设答案，已使用原始问题检索。"}</p>
      )}
      <small>
        向量召回文本 {embeddingTextCount || 1} 条
        {errorText ? ` · ${errorText}` : ""}
      </small>
    </div>
  );
}

function getTraceStringList(trace: ChatResponse["retrieval_trace"] | undefined, key: string) {
  const value = trace?.[key];
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function QueryExpansionTracePanel({ trace }: { trace?: ChatResponse["retrieval_trace"] }) {
  const enabled =
    trace?.query_expansion_enabled === true ||
    getTraceString(trace, "query_transform").includes("query_expansion");
  if (!enabled) return null;

  const status = getTraceString(trace, "query_expansion_status");
  const expandedQueries = getTraceStringList(trace, "expanded_queries");
  const errorText = getTraceString(trace, "query_expansion_error");

  return (
    <div className={`hyde-trace query-expansion-trace ${status === "applied" ? "applied" : "fallback"}`}>
      <div className="hyde-trace-head">
        <span>
          <Search size={12} />
          查询扩展
        </span>
        <strong>{getHydeStatusLabel(status)}</strong>
      </div>
      {expandedQueries.length > 0 ? (
        <ol>
          {expandedQueries.map((query) => (
            <li key={query}>{query}</li>
          ))}
        </ol>
      ) : (
        <p>{errorText || "没有生成额外查询，已使用原始问题检索。"}</p>
      )}
      <small>
        扩展查询 {expandedQueries.length} 条
        {errorText ? ` · ${errorText}` : ""}
      </small>
    </div>
  );
}

function getCitationDisplayContent(citation: ChatResponse["citations"][number]) {
  const matchedChildContent = citation.metadata?.matched_child_content;
  if (typeof matchedChildContent === "string" && matchedChildContent.trim()) {
    return matchedChildContent.trim();
  }
  return citation.content.trim();
}

export function App() {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [activeKbId, setActiveKbId] = useState("");
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [message, setMessage] = useState("");
  const [useHyde, setUseHyde] = useState(false);
  const [useQueryExpansion, setUseQueryExpansion] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [busy, setBusy] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [status, setStatus] = useState("连接中");
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("chat");
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [panelTab, setPanelTab] = useState<PanelTab>("kb");
  const [isDragging, setIsDragging] = useState(false);
  const [pendingIngestionFile, setPendingIngestionFile] = useState<File | null>(null);
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModelOption[]>([]);
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState(DEFAULT_EMBEDDING_MODEL);
  const [newKbRetrievalPolicy, setNewKbRetrievalPolicy] = useState(DEFAULT_RETRIEVAL_POLICY);
  const [activeKbRetrievalPolicy, setActiveKbRetrievalPolicy] =
    useState<RetrievalPolicyInput>(DEFAULT_RETRIEVAL_POLICY);
  const [isCreateKbFormOpen, setIsCreateKbFormOpen] = useState(false);
  const [isSavingRetrievalPolicy, setIsSavingRetrievalPolicy] = useState(false);
  const [isDeletingKnowledgeBase, setIsDeletingKnowledgeBase] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  const activeKb = useMemo(
    () => knowledgeBases.find((item) => item.id === activeKbId),
    [knowledgeBases, activeKbId],
  );
  const defaultEmbeddingModel = useMemo(
    () => embeddingModels[0]?.id ?? DEFAULT_EMBEDDING_MODEL,
    [embeddingModels],
  );

  async function refresh() {
    try {
      const [kbs, models] = await Promise.all([listKnowledgeBases(), listEmbeddingModels()]);
      setKnowledgeBases(kbs);
      setEmbeddingModels(models);
      const selected = activeKbId || kbs[0]?.id || "";
      setActiveKbId(selected);
      if (selected) setDocuments(await listDocuments(selected));
      setSelectedEmbeddingModel(models[0]?.id ?? DEFAULT_EMBEDDING_MODEL);
      setStatus("在线");
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : "连接失败");
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (!activeKbId) return;
    listDocuments(activeKbId)
      .then(setDocuments)
      .catch((e: unknown) => setStatus(e instanceof Error ? e.message : "加载失败"));
  }, [activeKbId]);

  useEffect(() => {
    setActiveKbRetrievalPolicy(normalizeRetrievalPolicy(activeKb?.retrieval_policy));
  }, [activeKb]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, isTyping]);

  function openPanel(tab: PanelTab) {
    setPanelTab(tab);
    setIsPanelOpen(true);
  }

  function autoResize(el: HTMLTextAreaElement) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
  }

  function startDocumentIngestion(file: File) {
    if (!file || !activeKbId) return;
    setPendingIngestionFile(file);
    openPanel("docs");
  }

  function onUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) startDocumentIngestion(file);
    event.target.value = "";
  }

  function onDrop(event: ReactDragEvent<HTMLElement>) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer?.files[0];
    if (file) startDocumentIngestion(file);
  }

  function onDocumentSaved(
    document: DocumentRecord,
    _chunkingPolicy: ChunkingPolicyInput,
    _embeddingModel: string,
  ) {
    setDocuments((prev) => {
      const withoutCurrent = prev.filter((item) => item.id !== document.id);
      return [document, ...withoutCurrent];
    });
    setStatus("文档索引已更新");
  }

  function onDocumentDeleted(documentId: string) {
    setDocuments((prev) => prev.filter((document) => document.id !== documentId));
    setStatus("文档已彻底删除");
  }

  async function onCreateKb(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const retrievalPolicy = {
      ...newKbRetrievalPolicy,
      keyword_weight: Number((1 - newKbRetrievalPolicy.vector_weight).toFixed(2)),
    };
    const kb = await createKnowledgeBase({
      name: String(data.get("name") || ""),
      description: String(data.get("description") || ""),
      visibility: "private",
      retrieval_policy: retrievalPolicy,
    });
    event.currentTarget.reset();
    setNewKbRetrievalPolicy(DEFAULT_RETRIEVAL_POLICY);
    try {
      setKnowledgeBases(await listKnowledgeBases());
    } catch {
      setKnowledgeBases((prev) => [...prev, kb]);
    }
    setActiveKbId(kb.id);
    setDocuments([]);
    openPanel("kb");
    setIsCreateKbFormOpen(false);
  }

  function updateActiveKbRetrievalPolicy(patch: Partial<RetrievalPolicyInput>) {
    setActiveKbRetrievalPolicy((current) => normalizeRetrievalPolicy({ ...current, ...patch }));
  }

  async function onSaveActiveRetrievalPolicy() {
    if (!activeKbId) return;
    setIsSavingRetrievalPolicy(true);
    try {
      const updated = await updateKnowledgeBaseRetrievalPolicy(
        activeKbId,
        normalizeRetrievalPolicy(activeKbRetrievalPolicy),
      );
      setKnowledgeBases((prev) => prev.map((kb) => (kb.id === updated.id ? updated : kb)));
      setStatus("检索设置已保存");
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : "保存失败");
    } finally {
      setIsSavingRetrievalPolicy(false);
    }
  }

  async function onDeleteActiveKnowledgeBase() {
    if (!activeKb) return;
    const confirmed = window.confirm(`确定删除知识库「${activeKb.name}」吗？相关文档也会一并删除。`);
    if (!confirmed) return;

    setIsDeletingKnowledgeBase(true);
    try {
      await deleteKnowledgeBase(activeKb.id);
      const remaining = knowledgeBases.filter((kb) => kb.id !== activeKb.id);
      setKnowledgeBases(remaining);
      setActiveKbId(remaining[0]?.id ?? "");
      if (!remaining.length) setDocuments([]);
      setStatus("知识库已删除");
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : "删除失败");
    } finally {
      setIsDeletingKnowledgeBase(false);
    }
  }

  async function onChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = message.trim();
    if (!text || !activeKbId || busy) return;

    setMessage("");
    if (chatInputRef.current) chatInputRef.current.style.height = "auto";
    setBusy(true);
    setIsTyping(true);
    setTurns((prev) => [...prev, { role: "user", content: text }]);

    try {
      const response = await sendChat(
        activeKbId,
        text,
        sessionId,
        activeKb?.retrieval_policy.top_k ?? 8,
        useHyde,
        useQueryExpansion,
      );
      setSessionId(response.session_id);
      setIsTyping(false);
      setTurns((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.answer,
          citations: response.citations,
          retrievalTrace: response.retrieval_trace,
        },
      ]);
    } catch {
      setIsTyping(false);
      setTurns((prev) => [
        ...prev,
        { role: "assistant", content: "抱歉，请求失败，请稍后重试。" },
      ]);
    } finally {
      setBusy(false);
      chatInputRef.current?.focus();
    }
  }

  function onChatKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      (event.currentTarget.closest("form") as HTMLFormElement | null)?.requestSubmit();
    }
  }

  function fillQuestion(q: string) {
    setMessage(q);
    chatInputRef.current?.focus();
  }

  // ── Panel tabs ────────────────────────────────────────────────

  const panelTabs: { id: PanelTab; label: string; icon: ReactNode }[] = [
    { id: "kb", label: "知识库", icon: <BookOpen size={14} /> },
    { id: "docs", label: "文档", icon: <FileText size={14} /> },
    { id: "governance", label: "治理", icon: <ShieldCheck size={14} /> },
  ];

  const embeddingSelectOptions = useMemo(
    () =>
      embeddingModels.map((model) => ({
        value: model.id,
        label: model.label,
        disabled: !model.enabled,
        hint: `${model.dimensions}d${model.enabled ? "" : ` · ${model.reason}`}`,
      })),
    [embeddingModels],
  );

  // ── KB tab ────────────────────────────────────────────────────

  const kbPanel = (
    <div className="panel-section">
      <p className="panel-section-label">选择知识库</p>
      <div className="kb-list">
        {knowledgeBases.map((kb) => (
          <button
            className={`kb-row ${kb.id === activeKbId ? "selected" : ""}`}
            key={kb.id}
            onClick={() => setActiveKbId(kb.id)}
            type="button"
          >
            <div className="kb-row-icon">
              <Database size={14} />
            </div>
            <div className="kb-row-info">
              <strong>{kb.name}</strong>
              <small>
                {kb.visibility} · top-{kb.retrieval_policy.top_k}
              </small>
            </div>
            <ChevronRight size={14} className="kb-row-arrow" />
          </button>
        ))}
        {!knowledgeBases.length && (
          <div className="panel-empty">
            <Database size={20} />
            <span>暂无知识库</span>
          </div>
        )}
      </div>

      <div className="panel-divider" />
      <p className="panel-section-label">当前检索设置</p>
      {activeKb ? (
        <div className="active-kb-settings">
          <div className="active-kb-settings-title">
            <strong>{activeKb.name}</strong>
            <span>
              {activeKb.visibility} · top-{activeKb.retrieval_policy.top_k}
            </span>
          </div>
          <RetrievalSettingsFields
            namePrefix="active_kb"
            onChange={updateActiveKbRetrievalPolicy}
            policy={activeKbRetrievalPolicy}
          />
          <div className="kb-settings-actions">
            <button
              className="btn-secondary"
              disabled={isSavingRetrievalPolicy}
              onClick={onSaveActiveRetrievalPolicy}
              type="button"
            >
              {isSavingRetrievalPolicy ? <Loader2 size={14} className="spin" /> : <Save size={14} />}
              保存设置
            </button>
            <button
              className="btn-danger"
              disabled={isDeletingKnowledgeBase}
              onClick={onDeleteActiveKnowledgeBase}
              type="button"
            >
              {isDeletingKnowledgeBase ? <Loader2 size={14} className="spin" /> : <Trash2 size={14} />}
              删除
            </button>
          </div>
        </div>
      ) : (
        <div className="panel-empty compact">
          <Database size={18} />
          <span>选择一个知识库后可调整检索参数</span>
        </div>
      )}

      <div className="panel-divider" />
      <button
        aria-controls="create-kb-form"
        aria-expanded={isCreateKbFormOpen}
        className={`create-kb-toggle ${isCreateKbFormOpen ? "open" : ""}`}
        onClick={() => setIsCreateKbFormOpen((current) => !current)}
        type="button"
      >
        <span>
          <Plus size={14} />
          新建知识库
        </span>
        {isCreateKbFormOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      {isCreateKbFormOpen && (
        <form className="create-kb-form" id="create-kb-form" onSubmit={onCreateKb}>
          <input aria-label="知识库名称" name="name" placeholder="知识库名称" required />
          <input aria-label="知识库描述" name="description" placeholder="描述（选填）" />
          <RetrievalSettingsFields
            namePrefix="new_kb"
            onChange={(patch) =>
              setNewKbRetrievalPolicy((current) => normalizeRetrievalPolicy({ ...current, ...patch }))
            }
            policy={newKbRetrievalPolicy}
          />
          <div className="create-kb-actions">
            <button className="btn-primary" type="submit">
              <Plus size={14} />
              创建
            </button>
            <button
              className="btn-secondary"
              onClick={() => {
                setIsCreateKbFormOpen(false);
                setNewKbRetrievalPolicy(DEFAULT_RETRIEVAL_POLICY);
              }}
              type="button"
            >
              取消
            </button>
          </div>
        </form>
      )}
    </div>
  );

  // ── Docs tab ──────────────────────────────────────────────────

  const docsPanel = (
    <DocumentIngestionWizard
      activeKb={activeKb}
      activeKbId={activeKbId}
      documents={documents}
      embeddingModel={selectedEmbeddingModel}
      embeddingOptions={embeddingSelectOptions}
      initialFile={pendingIngestionFile}
      onDocumentParametersReset={() => setSelectedEmbeddingModel(defaultEmbeddingModel)}
      onEmbeddingModelChange={setSelectedEmbeddingModel}
      onInitialFileConsumed={() => setPendingIngestionFile(null)}
      onNeedKnowledgeBase={() => setPanelTab("kb")}
      onDeleted={onDocumentDeleted}
      onSaved={onDocumentSaved}
    />
  );

  // ── Governance tab ────────────────────────────────────────────

  const governanceCards: { icon: ReactNode; title: string; desc: string; onClick: () => void }[] =
    [
      {
        icon: <RefreshCw size={15} />,
        title: "文档生命周期",
        desc: "上传、解析、预览分块、单文档重建和失败处理。",
        onClick: () => setPanelTab("docs"),
      },
      {
        icon: <Telescope size={15} />,
        title: "检索可观测",
        desc: "记录 top_k、融合策略和引用，便于评测调优。",
        onClick: () => chatInputRef.current?.focus(),
      },
      {
        icon: <Key size={15} />,
        title: "租户与权限",
        desc: "核心数据带 tenant_id，预留 RBAC 和行级隔离扩展。",
        onClick: () => {},
      },
    ];

  const governancePanel = (
    <div className="panel-section">
      {governanceCards.map((card) => (
        <div className="governance-card" key={card.title}>
          <div className="governance-card-icon">{card.icon}</div>
          <div className="governance-card-body">
            <strong>{card.title}</strong>
            <p>{card.desc}</p>
            <button className="governance-card-link" onClick={card.onClick} type="button">
              了解更多 <ChevronRight size={12} />
            </button>
          </div>
        </div>
      ))}
    </div>
  );

  // ── Render ────────────────────────────────────────────────────

  return (
    <div className="app">
      {/* ── App Header ─────────────────────────────────────────── */}
      <header className="app-header">
        <button
          aria-label="切换侧边栏"
          className={`panel-toggle ${isPanelOpen ? "active" : ""}`}
          onClick={() => setIsPanelOpen((v) => !v)}
          type="button"
        >
          <Menu size={18} />
        </button>

        <div className="header-brand">
          <div className="header-brand-icon">
            <Database size={15} />
          </div>
          <span>KnowledgeRAG</span>
        </div>

        <div className="header-sep" />

        <button className="kb-selector" onClick={() => openPanel("kb")} type="button">
          <BookOpen size={13} />
          <span className="kb-selector-name">{activeKb?.name ?? "选择知识库"}</span>
          <ChevronDown size={12} />
        </button>

        <div className="workspace-switch" aria-label="工作区">
          <button
            className={workspaceMode === "chat" ? "active" : ""}
            onClick={() => setWorkspaceMode("chat")}
            type="button"
          >
            <MessageSquare size={13} />
            提问
          </button>
          <button
            className={workspaceMode === "eval" ? "active" : ""}
            onClick={() => setWorkspaceMode("eval")}
            type="button"
          >
            <BarChart3 size={13} />
            评测
          </button>
        </div>

        <div className="header-spacer" />

        <button
          className="header-icon-btn"
          onClick={() => openPanel("docs")}
          title={`${documents.length} 篇文档`}
          type="button"
        >
          <FileText size={15} />
          <span>{documents.length}</span>
        </button>

        <label className="header-upload-btn" title="上传文档到当前知识库">
          <FileUp size={14} />
          <span>上传</span>
          <input
            accept=".pdf,.txt,.md,.markdown,.csv,.json,.log,text/*,application/pdf"
            type="file"
            onChange={onUpload}
          />
        </label>

        <div className={`status-dot-pill ${status === "在线" ? "online" : "offline"}`}>
          <span className="status-dot" />
          {status}
        </div>
      </header>

      {/* ── App Body ───────────────────────────────────────────── */}
      <div className="app-body">
        {/* ── Side Panel ─────────────────────────────────────── */}
        <aside
          className={`side-panel ${isPanelOpen ? "open" : ""} ${
            panelTab === "docs" ? "docs-panel" : ""
          }`}
        >
          <div className="panel-header">
            <div className="panel-tabs">
              {panelTabs.map((tab) => (
                <button
                  className={`panel-tab ${panelTab === tab.id ? "active" : ""}`}
                  key={tab.id}
                  onClick={() => setPanelTab(tab.id)}
                  type="button"
                >
                  {tab.icon}
                  <span>{tab.label}</span>
                </button>
              ))}
            </div>
            <button
              aria-label="关闭面板"
              className="panel-close-btn"
              onClick={() => setIsPanelOpen(false)}
              type="button"
            >
              <X size={15} />
            </button>
          </div>

          <div className="panel-body">
            {panelTab === "kb" && kbPanel}
            {panelTab === "docs" && docsPanel}
            {panelTab === "governance" && governancePanel}
          </div>
        </aside>

        {/* ── Chat Main ──────────────────────────────────────── */}
        <main
          className={`chat-main ${isDragging ? "dragging" : ""}`}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDragging(false);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDrop={onDrop}
        >
          {workspaceMode === "chat" && isDragging && (
            <div className="drop-overlay">
              <FileUp size={44} />
              <strong>松开鼠标上传文件</strong>
              <span>将上传至「{activeKb?.name ?? "当前知识库"}」</span>
            </div>
          )}

          {workspaceMode === "eval" ? (
            <EvaluationWorkbench
              activeKbId={activeKbId}
              activeKbName={activeKb?.name}
              onStatus={setStatus}
            />
          ) : (
            <>
              {/* ── Messages ─────────────────────────────────────── */}
              <div className="chat-messages">
            {turns.length === 0 && (
              <div className="chat-empty-state">
                <div className="chat-empty-icon">
                  <MessageSquare size={28} />
                </div>
                <h2>向知识库提问</h2>
                <p>基于企业文档获取智能解答，支持引用溯源</p>

                {!activeKbId ? (
                  <button className="btn-primary" onClick={() => openPanel("kb")} type="button">
                    <BookOpen size={14} />
                    选择知识库
                  </button>
                ) : (
                  <div className="suggested-questions">
                    <span className="suggested-label">试试这些问题</span>
                    {SUGGESTED_QUESTIONS.map((q) => (
                      <button
                        className="suggested-q"
                        key={q}
                        onClick={() => fillQuestion(q)}
                        type="button"
                      >
                        <Search size={13} />
                        {q}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {turns.map((turn, index) => (
              <article className={`turn ${turn.role}`} key={`${turn.role}-${index}`}>
                <div className="avatar">
                  {turn.role === "assistant" ? <Bot size={14} /> : "我"}
                </div>
                <div className="turn-body">
                  <div className="turn-bubble">
                    <p>{turn.content}</p>
                  </div>
                  <HyDETracePanel trace={turn.retrievalTrace} />
                  <QueryExpansionTracePanel trace={turn.retrievalTrace} />
                  {turn.citations && turn.citations.length > 0 && (
                    <div className="citations">
                      <span className="citations-label">
                        <Search size={11} />
                        引用来源
                      </span>
                      {turn.citations.map((citation) => (
                        <article className="citation" key={citation.chunk_id}>
                          <header className="citation-header">
                            <FileText size={12} />
                            <span className="citation-name">{citation.filename}</span>
                            <span className="citation-score">{citation.score.toFixed(3)}</span>
                          </header>
                          <p className="citation-content">
                            {getCitationDisplayContent(citation) || "没有可展示的引用文本"}
                          </p>
                        </article>
                      ))}
                    </div>
                  )}
                </div>
              </article>
            ))}

            {isTyping && (
              <article className="turn assistant">
                <div className="avatar">
                  <Bot size={14} />
                </div>
                <div className="turn-body">
                  <div className="turn-bubble">
                    <div className="typing-indicator">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </div>
              </article>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* ── Composer ───────────────────────────────────────── */}
          <div className="chat-composer">
            {!activeKbId && (
              <div className="composer-warning">
                <AlertCircle size={13} />
                <span>请先选择知识库</span>
                <button onClick={() => openPanel("kb")} type="button">
                  去选择 →
                </button>
              </div>
            )}

            <div className="composer-options">
              <label className={`composer-option-toggle ${useHyde ? "active" : ""}`}>
                <input
                  checked={useHyde}
                  disabled={!activeKbId || busy}
                  onChange={(event) => setUseHyde(event.target.checked)}
                  type="checkbox"
                />
                <Sparkles size={13} />
                <span>HyDE 查询扩展</span>
              </label>
              <label className={`composer-option-toggle ${useQueryExpansion ? "active" : ""}`}>
                <input
                  checked={useQueryExpansion}
                  disabled={!activeKbId || busy}
                  onChange={(event) => setUseQueryExpansion(event.target.checked)}
                  type="checkbox"
                />
                <Search size={13} />
                <span>查询扩展</span>
              </label>
            </div>

            <form className="composer-box" onSubmit={onChat}>
              <textarea
                aria-label="对话内容"
                className="composer-input"
                disabled={!activeKbId || busy}
                onChange={(e) => {
                  setMessage(e.target.value);
                  autoResize(e.target);
                }}
                onKeyDown={onChatKeyDown}
                placeholder={
                  activeKbId
                    ? "询问企业制度、流程、产品资料…"
                    : "请先选择知识库"
                }
                ref={chatInputRef}
                rows={1}
                value={message}
              />
              <button
                aria-label="发送"
                className={`send-btn ${message.trim() && activeKbId && !busy ? "ready" : ""}`}
                disabled={busy || !message.trim() || !activeKbId}
                type="submit"
              >
                {busy ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
              </button>
            </form>

            <p className="composer-tip">
              Enter 发送 · Shift+Enter 换行 · 拖拽文件到页面上传
            </p>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
