import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import {
  Activity,
  Bot,
  CheckCircle2,
  ChevronRight,
  FileUp,
  Loader2,
  Plus,
  Search,
  Send,
  Settings2,
} from "lucide-react";
import { Shell } from "../components/Shell";
import {
  ChatResponse,
  DocumentRecord,
  KnowledgeBase,
  createKnowledgeBase,
  listDocuments,
  listKnowledgeBases,
  sendChat,
  uploadDocument,
} from "../lib/api";

type ChatTurn = {
  role: "user" | "assistant";
  content: string;
  citations?: ChatResponse["citations"];
};

export function App() {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [activeKbId, setActiveKbId] = useState("");
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("连接中");

  const activeKb = useMemo(
    () => knowledgeBases.find((item) => item.id === activeKbId),
    [knowledgeBases, activeKbId],
  );

  async function refresh() {
    const kbs = await listKnowledgeBases();
    setKnowledgeBases(kbs);
    const selected = activeKbId || kbs[0]?.id || "";
    setActiveKbId(selected);
    if (selected) {
      setDocuments(await listDocuments(selected));
    }
    setStatus("在线");
  }

  useEffect(() => {
    refresh().catch((error) => setStatus(error.message));
  }, []);

  useEffect(() => {
    if (!activeKbId) return;
    listDocuments(activeKbId).then(setDocuments).catch((error) => setStatus(error.message));
  }, [activeKbId]);

  async function onCreateKb(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const kb = await createKnowledgeBase({
      name: String(data.get("name") || ""),
      description: String(data.get("description") || ""),
      visibility: "private",
    });
    event.currentTarget.reset();
    setKnowledgeBases((items) => [...items, kb]);
    setActiveKbId(kb.id);
  }

  async function onUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !activeKbId) return;
    setBusy(true);
    try {
      const doc = await uploadDocument(activeKbId, file);
      setDocuments((items) => [doc, ...items]);
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  }

  async function onChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = message.trim();
    if (!text || !activeKbId) return;
    setMessage("");
    setBusy(true);
    setTurns((items) => [...items, { role: "user", content: text }]);
    try {
      const response = await sendChat(activeKbId, text, sessionId);
      setSessionId(response.session_id);
      setTurns((items) => [
        ...items,
        { role: "assistant", content: response.answer, citations: response.citations },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Shell>
      <header className="topbar">
        <div>
          <p className="eyebrow">企业知识库</p>
          <h1>RAG 运维与问答工作台</h1>
        </div>
        <div className="status-pill">
          <Activity size={16} />
          {status}
        </div>
      </header>

      <section className="workspace">
        <div className="left-pane" id="knowledge">
          <section className="panel">
            <div className="panel-head">
              <h2>知识库</h2>
              <Settings2 size={18} />
            </div>
            <div className="kb-list">
              {knowledgeBases.map((kb) => (
                <button
                  className={`kb-row ${kb.id === activeKbId ? "selected" : ""}`}
                  key={kb.id}
                  onClick={() => setActiveKbId(kb.id)}
                  type="button"
                >
                  <span>
                    <strong>{kb.name}</strong>
                    <small>{kb.visibility} · top {kb.retrieval_policy.top_k}</small>
                  </span>
                  <ChevronRight size={16} />
                </button>
              ))}
            </div>
            <form className="inline-form" onSubmit={onCreateKb}>
              <input aria-label="知识库名称" name="name" placeholder="新知识库" required />
              <input aria-label="知识库描述" name="description" placeholder="描述" />
              <button aria-label="创建知识库" className="icon-button" type="submit">
                <Plus size={18} />
              </button>
            </form>
          </section>

          <section className="panel metrics" id="governance">
            <div>
              <span className="metric-value">{documents.length}</span>
              <span className="metric-label">文档</span>
            </div>
            <div>
              <span className="metric-value">RRF</span>
              <span className="metric-label">融合</span>
            </div>
            <div>
              <span className="metric-value">ACL</span>
              <span className="metric-label">权限</span>
            </div>
          </section>
        </div>

        <section className="center-pane" id="documents">
          <div className="panel-head">
            <h2>{activeKb?.name ?? "知识库"}</h2>
            <label className="upload-button">
              {busy ? <Loader2 className="spin" size={18} /> : <FileUp size={18} />}
              上传
              <input type="file" onChange={onUpload} />
            </label>
          </div>
          <div className="doc-table">
            <div className="table-head">
              <span>文件</span>
              <span>状态</span>
              <span>解析器</span>
              <span>时间</span>
            </div>
            {documents.map((doc) => (
              <div className="table-row" key={doc.id}>
                <span className="file-cell">
                  <CheckCircle2 size={16} />
                  {doc.filename}
                </span>
                <span className={`badge ${doc.status}`}>{doc.status}</span>
                <span>{doc.parser}</span>
                <span>{new Date(doc.created_at).toLocaleString()}</span>
              </div>
            ))}
            {!documents.length && <div className="empty">暂无文档</div>}
          </div>
        </section>

        <section className="right-pane" id="chat">
          <div className="chat-head">
            <div>
              <h2>对话检索</h2>
              <p>{activeKb?.name ?? "未选择知识库"}</p>
            </div>
            <Search size={18} />
          </div>
          <div className="chat-log">
            {turns.map((turn, index) => (
              <article className={`turn ${turn.role}`} key={`${turn.role}-${index}`}>
                <div className="avatar">{turn.role === "assistant" ? <Bot size={16} /> : "我"}</div>
                <div>
                  <p>{turn.content}</p>
                  {turn.citations?.map((citation) => (
                    <blockquote key={citation.chunk_id}>
                      {citation.filename} · {citation.score.toFixed(4)}
                    </blockquote>
                  ))}
                </div>
              </article>
            ))}
          </div>
          <form className="chat-form" onSubmit={onChat}>
            <input
              aria-label="对话内容"
              onChange={(event) => setMessage(event.target.value)}
              placeholder="询问企业制度、流程、产品资料"
              value={message}
            />
            <button aria-label="发送" className="icon-button primary" disabled={busy} type="submit">
              {busy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
            </button>
          </form>
        </section>
      </section>
    </Shell>
  );
}

