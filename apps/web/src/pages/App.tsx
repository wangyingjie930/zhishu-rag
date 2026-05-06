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
  Search,
  Send,
  ShieldCheck,
  Telescope,
  X,
} from "lucide-react";
import {
  type ChatResponse,
  type DocumentRecord,
  type EmbeddingModelOption,
  type KnowledgeBase,
  createKnowledgeBase,
  listEmbeddingModels,
  listDocuments,
  listKnowledgeBases,
  sendChat,
} from "../lib/api";
import { ChunkingPolicyFields } from "../components/ChunkingPolicyFields";
import { FancySelect } from "../components/FancySelect";
import {
  buildChunkingPolicyFromForm,
  type ChunkingPolicyInput,
} from "../lib/chunkingPolicy";
import { DocumentIngestionWizard } from "../features/documentIngestion/DocumentIngestionWizard";

type ChatTurn = {
  role: "user" | "assistant";
  content: string;
  citations?: ChatResponse["citations"];
};

type PanelTab = "kb" | "docs" | "governance";

const SUGGESTED_QUESTIONS = [
  "产品的核心功能有哪些？",
  "如何申请相关权限？",
  "查看最新流程规范",
];

const DEFAULT_EMBEDDING_MODEL = "google:gemini-embedding-001";

export function App() {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [activeKbId, setActiveKbId] = useState("");
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [busy, setBusy] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [status, setStatus] = useState("连接中");
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [panelTab, setPanelTab] = useState<PanelTab>("kb");
  const [isDragging, setIsDragging] = useState(false);
  const [pendingIngestionFile, setPendingIngestionFile] = useState<File | null>(null);
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModelOption[]>([]);
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState(DEFAULT_EMBEDDING_MODEL);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  const activeKb = useMemo(
    () => knowledgeBases.find((item) => item.id === activeKbId),
    [knowledgeBases, activeKbId],
  );

  async function refresh() {
    try {
      const [kbs, models] = await Promise.all([listKnowledgeBases(), listEmbeddingModels()]);
      setKnowledgeBases(kbs);
      setEmbeddingModels(models);
      const selected = activeKbId || kbs[0]?.id || "";
      setActiveKbId(selected);
      if (selected) setDocuments(await listDocuments(selected));
      const activePolicy = kbs.find((item) => item.id === selected)?.ingestion_policy;
      setSelectedEmbeddingModel(activePolicy?.embedding?.model ?? models[0]?.id ?? DEFAULT_EMBEDDING_MODEL);
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
    const activePolicy = knowledgeBases.find((item) => item.id === activeKbId)?.ingestion_policy;
    setSelectedEmbeddingModel(
      activePolicy?.embedding?.model ?? embeddingModels[0]?.id ?? DEFAULT_EMBEDDING_MODEL,
    );
    listDocuments(activeKbId)
      .then(setDocuments)
      .catch((e: unknown) => setStatus(e instanceof Error ? e.message : "加载失败"));
  }, [activeKbId, knowledgeBases, embeddingModels]);

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
    chunkingPolicy: ChunkingPolicyInput,
    embeddingModel: string,
  ) {
    setDocuments((prev) => [document, ...prev]);
    setKnowledgeBases((prev) =>
      prev.map((kb) =>
        kb.id === activeKbId
          ? {
              ...kb,
              ingestion_policy: {
                ...kb.ingestion_policy,
                embedding: { model: embeddingModel },
                chunker: chunkingPolicy,
              },
            }
          : kb,
      ),
    );
  }

  async function onCreateKb(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const embeddingModel = String(data.get("embedding_model") || selectedEmbeddingModel);
    const kb = await createKnowledgeBase({
      name: String(data.get("name") || ""),
      description: String(data.get("description") || ""),
      visibility: "private",
      ingestion_policy: {
        embedding: { model: embeddingModel },
        chunker: buildChunkingPolicyFromForm(data),
      },
    });
    event.currentTarget.reset();
    setKnowledgeBases((prev) => [...prev, kb]);
    setActiveKbId(kb.id);
    setSelectedEmbeddingModel(embeddingModel);
    openPanel("docs");
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
      const response = await sendChat(activeKbId, text, sessionId);
      setSessionId(response.session_id);
      setIsTyping(false);
      setTurns((prev) => [
        ...prev,
        { role: "assistant", content: response.answer, citations: response.citations },
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

  const embeddingSelectOptions = embeddingModels.map((model) => ({
    value: model.id,
    label: model.label,
    disabled: !model.enabled,
    hint: `${model.dimensions}d${model.enabled ? "" : ` · ${model.reason}`}`,
  }));

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
      <p className="panel-section-label">新建知识库</p>
      <form className="create-kb-form" onSubmit={onCreateKb}>
        <input aria-label="知识库名称" name="name" placeholder="知识库名称" required />
        <input aria-label="知识库描述" name="description" placeholder="描述（选填）" />
        <ChunkingPolicyFields />
        <FancySelect
          className="create-kb-model-select"
          onChange={setSelectedEmbeddingModel}
          options={embeddingSelectOptions}
          value={selectedEmbeddingModel}
        />
        <input name="embedding_model" type="hidden" value={selectedEmbeddingModel} />
        <button className="btn-primary" type="submit">
          <Plus size={14} />
          创建
        </button>
      </form>
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
      onEmbeddingModelChange={setSelectedEmbeddingModel}
      onInitialFileConsumed={() => setPendingIngestionFile(null)}
      onNeedKnowledgeBase={() => setPanelTab("kb")}
      onSaved={onDocumentSaved}
    />
  );

  // ── Governance tab ────────────────────────────────────────────

  const governanceCards: { icon: ReactNode; title: string; desc: string; onClick: () => void }[] =
    [
      {
        icon: <RefreshCw size={15} />,
        title: "文档生命周期",
        desc: "上传、解析、索引、失败重试，后续可扩展版本管理。",
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

        <FancySelect
          buttonClassName="header-model-trigger"
          className="header-model-select"
          menuClassName="header-model-menu"
          onChange={setSelectedEmbeddingModel}
          options={embeddingSelectOptions}
          value={selectedEmbeddingModel}
        />

        <label className="header-upload-btn" title="上传文档到当前知识库">
          <FileUp size={14} />
          <span>上传</span>
          <input type="file" onChange={onUpload} />
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
          {isDragging && (
            <div className="drop-overlay">
              <FileUp size={44} />
              <strong>松开鼠标上传文件</strong>
              <span>将上传至「{activeKb?.name ?? "当前知识库"}」</span>
            </div>
          )}

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
                  {turn.citations && turn.citations.length > 0 && (
                    <div className="citations">
                      <span className="citations-label">
                        <Search size={11} />
                        引用来源
                      </span>
                      {turn.citations.map((citation) => (
                        <div className="citation" key={citation.chunk_id}>
                          <FileText size={12} />
                          <span className="citation-name">{citation.filename}</span>
                          <span className="citation-score">{citation.score.toFixed(3)}</span>
                        </div>
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
        </main>
      </div>
    </div>
  );
}
