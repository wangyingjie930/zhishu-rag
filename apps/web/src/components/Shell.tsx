import { BookOpen, Database, FileText, MessageSquare, ShieldCheck } from "lucide-react";
import type { ReactNode } from "react";

export function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Database size={22} />
          <span>Knowledge RAG</span>
        </div>
        <nav className="nav">
          <a className="nav-item active" href="#knowledge">
            <BookOpen size={18} />
            知识库
          </a>
          <a className="nav-item" href="#documents">
            <FileText size={18} />
            文档
          </a>
          <a className="nav-item" href="#chat">
            <MessageSquare size={18} />
            对话
          </a>
          <a className="nav-item" href="#governance">
            <ShieldCheck size={18} />
            治理
          </a>
        </nav>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}

