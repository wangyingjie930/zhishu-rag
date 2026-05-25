import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  BarChart3,
  Check,
  ClipboardCheck,
  Database,
  FileText,
  Filter,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  Save,
  Search,
  Trash2,
  X,
} from "lucide-react";
import {
  type Citation,
  type EvalCandidate,
  type EvalDataset,
  type EvalMetricMap,
  type EvalRun,
  type EvalRunResult,
  type EvalSample,
  addEvalSample,
  createEvalDataset,
  createEvalRun,
  deleteEvalDataset,
  getEvalRun,
  listEvalCandidates,
  listEvalDatasets,
  listEvalSamples,
  updateEvalSample,
} from "../../lib/api";

const METRIC_DEFS = [
  { key: "faithfulness", labelCn: "忠实度", labelEn: "Faithfulness", shortLabel: "忠实" },
  {
    key: "response_relevancy",
    labelCn: "回答相关性",
    labelEn: "Response Relevancy",
    shortLabel: "相关",
  },
  {
    key: "answer_correctness",
    labelCn: "答案正确性",
    labelEn: "Answer Correctness",
    shortLabel: "正确",
  },
  {
    key: "context_precision",
    labelCn: "上下文精确率",
    labelEn: "Context Precision",
    shortLabel: "精确",
  },
  {
    key: "context_recall",
    labelCn: "上下文召回率",
    labelEn: "Context Recall",
    shortLabel: "召回",
  },
  { key: "hit_rate", labelCn: "命中率", labelEn: "Hit Rate", shortLabel: "命中" },
  { key: "mrr", labelCn: "首个相关排名", labelEn: "MRR", shortLabel: "MRR" },
  { key: "precision", labelCn: "检索精确率", labelEn: "Precision", shortLabel: "Prec" },
  { key: "recall", labelCn: "检索召回率", labelEn: "Recall", shortLabel: "Rec" },
  { key: "ap", labelCn: "平均精确率", labelEn: "AP", shortLabel: "AP" },
  { key: "ndcg", labelCn: "排序增益", labelEn: "nDCG", shortLabel: "nDCG" },
];

const METRIC_GROUPS = [
  {
    key: "generation",
    labelCn: "生成指标",
    labelEn: "Generation",
    metricKeys: ["faithfulness", "response_relevancy", "answer_correctness"],
  },
  {
    key: "retrieval",
    labelCn: "RAGAS 检索指标",
    labelEn: "RAGAS Retrieval",
    metricKeys: ["context_precision", "context_recall"],
  },
  {
    key: "deterministic-retrieval",
    labelCn: "确定性检索指标",
    labelEn: "LlamaIndex RetrieverEvaluator",
    metricKeys: ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"],
  },
] as const;

type EvaluationWorkbenchProps = {
  activeKbId: string;
  activeKbName?: string;
  onStatus?: (message: string) => void;
};

type DetailState =
  | { kind: "sample"; sample: EvalSample }
  | { kind: "result"; result: EvalRunResult }
  | null;

function preview(text: string, limit = 120) {
  const normalized = text.replace(/\s+/g, " ").trim();
  return normalized.length > limit ? `${normalized.slice(0, limit)}...` : normalized;
}

function formatScore(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "未评分";
}

function metricLabel(metric: (typeof METRIC_DEFS)[number]) {
  return `${metric.labelCn} (${metric.labelEn})`;
}

function metricDefsForGroup(metricKeys: readonly string[]) {
  return metricKeys
    .map((metricKey) => METRIC_DEFS.find((metric) => metric.key === metricKey))
    .filter((metric): metric is (typeof METRIC_DEFS)[number] => Boolean(metric));
}

function formatDateTime(value: string) {
  if (!value) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function lowestMetric(metrics: EvalMetricMap) {
  const values = METRIC_DEFS.map((metric) => metrics[metric.key]).filter(
    (value): value is number => typeof value === "number" && Number.isFinite(value),
  );
  return values.length ? Math.min(...values) : 0;
}

function tagTextToList(text: string) {
  return text
    .split(/[,，\s]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function traceBadge(trace: Record<string, unknown>) {
  const labels = [];
  if (trace.hyde_enabled === true || trace.query_transform === "hyde") labels.push("HyDE");
  if (
    trace.query_expansion_enabled === true ||
    String(trace.query_transform || "").includes("query_expansion")
  ) {
    labels.push("查询扩展");
  }
  return labels;
}

function citationIds(citations: Citation[]) {
  return citations.map((citation) => citation.chunk_id).filter(Boolean);
}

export function EvaluationWorkbench({
  activeKbId,
  activeKbName,
  onStatus,
}: EvaluationWorkbenchProps) {
  const [candidates, setCandidates] = useState<EvalCandidate[]>([]);
  const [datasets, setDatasets] = useState<EvalDataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [samples, setSamples] = useState<EvalSample[]>([]);
  const [latestRun, setLatestRun] = useState<EvalRun | null>(null);
  const [detail, setDetail] = useState<DetailState>(null);
  const [referenceDraft, setReferenceDraft] = useState("");
  const [tagsDraft, setTagsDraft] = useState("");
  const [expectedIdsDraft, setExpectedIdsDraft] = useState<string[]>([]);
  const [filterMode, setFilterMode] = useState<"all" | "low-context-recall">("all");
  const [evalQueryExpansionEnabled, setEvalQueryExpansionEnabled] = useState(false);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [savingSample, setSavingSample] = useState(false);
  const [deletingDataset, setDeletingDataset] = useState(false);
  const [error, setError] = useState("");

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId),
    [datasets, selectedDatasetId],
  );

  const sortedResults = useMemo(() => {
    const results = latestRun?.results ?? [];
    return [...results]
      .filter((result) =>
        filterMode === "low-context-recall"
          ? (result.metrics.context_recall ?? 0) < 0.7
          : true,
      )
      .sort((left, right) => lowestMetric(left.metrics) - lowestMetric(right.metrics));
  }, [latestRun, filterMode]);

  useEffect(() => {
    setCandidates([]);
    setDatasets([]);
    setSamples([]);
    setLatestRun(null);
    setDetail(null);
    setSelectedDatasetId("");
    if (!activeKbId) return;
    void loadEvaluationHome(activeKbId);
  }, [activeKbId]);

  useEffect(() => {
    if (!selectedDatasetId) {
      setSamples([]);
      return;
    }
    void loadSamples(selectedDatasetId);
  }, [selectedDatasetId]);

  useEffect(() => {
    if (!latestRun || !["pending", "running"].includes(latestRun.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const run = await getEvalRun(latestRun.id);
        setLatestRun(run);
        if (!["pending", "running"].includes(run.status)) {
          setRunning(false);
          window.clearInterval(timer);
        }
      } catch {
        setRunning(false);
        window.clearInterval(timer);
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [latestRun]);

  useEffect(() => {
    if (detail?.kind !== "sample") return;
    setReferenceDraft(detail.sample.reference);
    setTagsDraft(detail.sample.tags.join(", "));
    setExpectedIdsDraft(detail.sample.expected_context_ids);
  }, [detail]);

  async function loadEvaluationHome(kbId: string) {
    setLoading(true);
    setError("");
    try {
      const [candidateRows, datasetRows] = await Promise.all([
        listEvalCandidates(kbId),
        listEvalDatasets(kbId),
      ]);
      setCandidates(candidateRows);
      setDatasets(datasetRows);
      setSelectedDatasetId(datasetRows[0]?.id ?? "");
    } catch (loadError: unknown) {
      setError(loadError instanceof Error ? loadError.message : "评测数据加载失败");
    } finally {
      setLoading(false);
    }
  }

  async function loadSamples(datasetId: string) {
    try {
      setSamples(await listEvalSamples(datasetId));
    } catch (loadError: unknown) {
      setError(loadError instanceof Error ? loadError.message : "样本加载失败");
    }
  }

  async function ensureDataset() {
    if (selectedDataset) return selectedDataset;
    const dataset = await createEvalDataset(
      activeKbId,
      `${activeKbName || "知识库"} 回归评测集`,
      "从历史对话沉淀的 RAGAS 评测集",
    );
    setDatasets((current) => [dataset, ...current]);
    setSelectedDatasetId(dataset.id);
    return dataset;
  }

  async function onAddCandidate(candidate: EvalCandidate) {
    if (!activeKbId) return;
    setError("");
    try {
      const dataset = await ensureDataset();
      const sample = await addEvalSample(dataset.id, {
        source_message_id: candidate.assistant_message_id,
        user_input: candidate.user_input,
        reference: candidate.response,
        expected_context_ids: citationIds(candidate.citations),
        tags: ["历史对话"],
        original_response: candidate.response,
        original_citations: candidate.citations,
        original_retrieval_trace: candidate.retrieval_trace,
      });
      setSamples((current) => [sample, ...current]);
      setDatasets((current) =>
        current.map((item) =>
          item.id === dataset.id ? { ...item, sample_count: item.sample_count + 1 } : item,
        ),
      );
      setDetail({ kind: "sample", sample });
      onStatus?.("已加入评测集");
    } catch (addError: unknown) {
      setError(addError instanceof Error ? addError.message : "加入样本失败");
    }
  }

  async function onSaveSample() {
    if (detail?.kind !== "sample" || !selectedDatasetId) return;
    setSavingSample(true);
    try {
      const updated = await updateEvalSample(selectedDatasetId, detail.sample.id, {
        reference: referenceDraft,
        tags: tagTextToList(tagsDraft),
        expected_context_ids: expectedIdsDraft,
      });
      setSamples((current) => current.map((sample) => (sample.id === updated.id ? updated : sample)));
      setDetail({ kind: "sample", sample: updated });
      onStatus?.("评测样本已保存");
    } catch (saveError: unknown) {
      setError(saveError instanceof Error ? saveError.message : "保存样本失败");
    } finally {
      setSavingSample(false);
    }
  }

  async function onRunEvaluation() {
    if (!selectedDatasetId || running) return;
    setRunning(true);
    setError("");
    try {
      const run = await createEvalRun(selectedDatasetId, evalQueryExpansionEnabled);
      setLatestRun(run);
      const hydrated = await getEvalRun(run.id);
      setLatestRun(hydrated);
      setRunning(["pending", "running"].includes(hydrated.status));
      onStatus?.(hydrated.status === "completed" ? "评测运行完成" : "评测运行已创建");
    } catch (runError: unknown) {
      setRunning(false);
      setError(runError instanceof Error ? runError.message : "评测运行失败");
    }
  }

  async function onDeleteDataset() {
    if (!selectedDataset || deletingDataset || running) return;
    const confirmed = window.confirm(
      `确定删除评测集「${selectedDataset.name}」吗？样本和历史运行结果也会一并删除。`,
    );
    if (!confirmed) return;

    setDeletingDataset(true);
    setError("");
    try {
      await deleteEvalDataset(selectedDataset.id);
      const remainingDatasets = datasets.filter((dataset) => dataset.id !== selectedDataset.id);
      const nextDatasetId = remainingDatasets[0]?.id ?? "";
      setDatasets(remainingDatasets);
      setSelectedDatasetId(nextDatasetId);
      setSamples([]);
      setLatestRun(null);
      setDetail(null);
      onStatus?.("评测集已删除");
      if (nextDatasetId) await loadSamples(nextDatasetId);
    } catch (deleteError: unknown) {
      setError(deleteError instanceof Error ? deleteError.message : "删除评测集失败");
    } finally {
      setDeletingDataset(false);
    }
  }

  function toggleExpectedContext(chunkId: string) {
    setExpectedIdsDraft((current) =>
      current.includes(chunkId)
        ? current.filter((item) => item !== chunkId)
        : [...current, chunkId],
    );
  }

  if (!activeKbId) {
    return (
      <section className="eval-empty">
        <Database size={24} />
        <strong>选择知识库后开始评测</strong>
      </section>
    );
  }

  return (
    <section className="eval-workbench">
      <aside className="eval-candidates">
        <div className="eval-pane-head">
          <span>
            <ClipboardCheck size={14} />
            历史候选样本
          </span>
          <button
            className="icon-button"
            disabled={loading}
            onClick={() => void loadEvaluationHome(activeKbId)}
            type="button"
          >
            {loading ? <Loader2 className="spin" size={14} /> : <RefreshCw size={14} />}
          </button>
        </div>

        {error && (
          <div className="eval-inline-error">
            <AlertCircle size={13} />
            {error}
          </div>
        )}

        <div className="eval-candidate-list">
          {candidates.map((candidate) => {
            const badges = traceBadge(candidate.retrieval_trace);
            return (
              <button
                className="eval-candidate"
                key={candidate.id}
                onClick={() => void onAddCandidate(candidate)}
                type="button"
              >
                <span className="eval-candidate-question">{preview(candidate.user_input, 72)}</span>
                <span className="eval-candidate-answer">{preview(candidate.response, 110)}</span>
                <span className="eval-candidate-meta">
                  <FileText size={11} />
                  {candidate.citations.length} 引用
                  <time>{formatDateTime(candidate.created_at)}</time>
                  {badges.map((badge) => (
                    <em key={badge}>{badge}</em>
                  ))}
                </span>
                <Plus size={14} className="eval-candidate-add" />
              </button>
            );
          })}
          {!loading && candidates.length === 0 && (
            <div className="eval-panel-empty">暂无可沉淀的历史问答</div>
          )}
        </div>
      </aside>

      <div className="eval-main">
        <div className="eval-toolbar">
          <div>
            <strong>评测实验台</strong>
            <span>{activeKbName || "当前知识库"}</span>
          </div>
          <select
            aria-label="评测集"
            onChange={(event) => setSelectedDatasetId(event.target.value)}
            value={selectedDatasetId}
          >
            <option value="">默认评测集</option>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.name}
              </option>
            ))}
          </select>
          <button
            className="btn-secondary"
            disabled={running}
            onClick={() => void ensureDataset()}
            type="button"
          >
            <Plus size={14} />
            新建
          </button>
          <button
            className="btn-danger"
            disabled={!selectedDataset || deletingDataset || running}
            onClick={onDeleteDataset}
            type="button"
          >
            {deletingDataset ? <Loader2 className="spin" size={14} /> : <Trash2 size={14} />}
            删除
          </button>
          <label className="eval-run-option">
            <input
              checked={evalQueryExpansionEnabled}
              disabled={running}
              onChange={(event) => setEvalQueryExpansionEnabled(event.target.checked)}
              type="checkbox"
            />
            <span>查询扩展</span>
          </label>
          <button
            className="btn-primary"
            disabled={!samples.length || running}
            onClick={onRunEvaluation}
            type="button"
          >
            {running ? <Loader2 className="spin" size={14} /> : <Play size={14} />}
            运行评测
          </button>
        </div>

        <div className="eval-metric-groups">
          {METRIC_GROUPS.map((group) => (
            <section className="eval-metric-group" key={group.key}>
              <div className="eval-metric-group-title">
                <span>{group.labelCn}</span>
                <small>{group.labelEn}</small>
              </div>
              <div className="eval-metric-grid">
                {metricDefsForGroup(group.metricKeys).map((metric) => (
                  <div className="eval-metric-card" key={metric.key}>
                    <span>{metric.labelCn}</span>
                    <small>{metric.labelEn}</small>
                    <strong>{formatScore(latestRun?.metrics?.[metric.key])}</strong>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>

        <div className="eval-content-grid">
          <div className="eval-section">
            <div className="eval-section-head">
              <span>当前评测集</span>
              <strong>{samples.length} 条样本</strong>
            </div>
            <div className="eval-sample-list">
              {samples.map((sample) => (
                <button
                  className={`eval-sample-row ${
                    detail?.kind === "sample" && detail.sample.id === sample.id ? "active" : ""
                  }`}
                  key={sample.id}
                  onClick={() => setDetail({ kind: "sample", sample })}
                  type="button"
                >
                  <span>{preview(sample.user_input, 86)}</span>
                  <small>
                    {sample.expected_context_ids.length} 期望引用 ·{" "}
                    {sample.tags.length ? sample.tags.join(" / ") : "未标记"}
                  </small>
                </button>
              ))}
              {samples.length === 0 && (
                <div className="eval-panel-empty">从左侧历史问答加入样本</div>
              )}
            </div>
          </div>

          <div className="eval-section">
            <div className="eval-section-head">
              <span>运行结果</span>
              <label className="eval-filter">
                <Filter size={12} />
                <select
                  aria-label="结果过滤"
                  onChange={(event) =>
                    setFilterMode(event.target.value as "all" | "low-context-recall")
                  }
                  value={filterMode}
                >
                  <option value="all">全部</option>
                  <option value="low-context-recall">上下文召回率 &lt; 0.7</option>
                </select>
              </label>
            </div>

            <div className="eval-result-list">
              {latestRun?.status === "failed" && (
                <div className="eval-inline-error">
                  <AlertCircle size={13} />
                  {latestRun.error_message || "评测运行失败"}
                </div>
              )}
              {sortedResults.map((result) => {
                const lowScore = lowestMetric(result.metrics) < 0.7;
                return (
                  <button
                    className={`eval-result-row ${lowScore ? "low" : ""} ${
                      detail?.kind === "result" && detail.result.id === result.id ? "active" : ""
                    }`}
                    key={result.id}
                    onClick={() => setDetail({ kind: "result", result })}
                    type="button"
                  >
                    <span>{preview(result.user_input, 70)}</span>
                    <div className="eval-result-metrics">
                      {METRIC_DEFS.map((metric) => (
                        <em key={metric.key} title={metricLabel(metric)}>
                          <span>{metric.shortLabel}</span>
                          <small>{metric.labelEn}</small>
                          <strong>{formatScore(result.metrics[metric.key])}</strong>
                        </em>
                      ))}
                    </div>
                  </button>
                );
              })}
              {latestRun && sortedResults.length === 0 && (
                <div className="eval-panel-empty">当前过滤条件下没有结果</div>
              )}
              {!latestRun && <div className="eval-panel-empty">运行后展示样本评分</div>}
            </div>
          </div>
        </div>
      </div>

      <aside className={`eval-detail ${detail ? "open" : ""}`}>
        <div className="eval-detail-head">
          <span>{detail?.kind === "result" ? "结果详情" : "样本详情"}</span>
          <button className="icon-button" onClick={() => setDetail(null)} type="button">
            <X size={14} />
          </button>
        </div>
        {detail?.kind === "sample" && (
          <SampleDetail
            expectedIdsDraft={expectedIdsDraft}
            onSaveSample={onSaveSample}
            onToggleExpectedContext={toggleExpectedContext}
            referenceDraft={referenceDraft}
            sample={detail.sample}
            savingSample={savingSample}
            setReferenceDraft={setReferenceDraft}
            setTagsDraft={setTagsDraft}
            tagsDraft={tagsDraft}
          />
        )}
        {detail?.kind === "result" && <ResultDetail result={detail.result} />}
      </aside>
    </section>
  );
}

function SampleDetail({
  sample,
  referenceDraft,
  tagsDraft,
  expectedIdsDraft,
  savingSample,
  setReferenceDraft,
  setTagsDraft,
  onToggleExpectedContext,
  onSaveSample,
}: {
  sample: EvalSample;
  referenceDraft: string;
  tagsDraft: string;
  expectedIdsDraft: string[];
  savingSample: boolean;
  setReferenceDraft: (value: string) => void;
  setTagsDraft: (value: string) => void;
  onToggleExpectedContext: (chunkId: string) => void;
  onSaveSample: () => void;
}) {
  return (
    <div className="eval-detail-body">
      <section>
        <label className="eval-field">
          <span>问题</span>
          <textarea readOnly value={sample.user_input} />
        </label>
        <label className="eval-field">
          <span>标准答案</span>
          <textarea
            onChange={(event) => setReferenceDraft(event.target.value)}
            value={referenceDraft}
          />
        </label>
        <label className="eval-field">
          <span>标签</span>
          <input onChange={(event) => setTagsDraft(event.target.value)} value={tagsDraft} />
        </label>
      </section>

      <section>
        <div className="eval-detail-title">
          <Search size={13} />
          期望引用
        </div>
        <div className="eval-citation-checks">
          {sample.original_citations.map((citation) => (
            <label key={citation.chunk_id}>
              <input
                checked={expectedIdsDraft.includes(citation.chunk_id)}
                onChange={() => onToggleExpectedContext(citation.chunk_id)}
                type="checkbox"
              />
              <span>
                <strong>{citation.filename}</strong>
                <small>{citation.score.toFixed(3)}</small>
                <p>{preview(citation.content, 160)}</p>
              </span>
            </label>
          ))}
          {sample.original_citations.length === 0 && (
            <div className="eval-panel-empty">原回答没有引用</div>
          )}
        </div>
      </section>

      <button className="btn-primary eval-save-btn" onClick={onSaveSample} type="button">
        {savingSample ? <Loader2 className="spin" size={14} /> : <Save size={14} />}
        保存样本
      </button>
    </div>
  );
}

function ResultDetail({ result }: { result: EvalRunResult }) {
  return (
    <div className="eval-detail-body">
      <section>
        <div className="eval-detail-title">
          <BarChart3 size={13} />
          指标
        </div>
        {METRIC_GROUPS.map((group) => (
          <div className="eval-detail-metric-group" key={group.key}>
            <div className="eval-detail-metric-group-title">
              {group.labelCn}
              <small>{group.labelEn}</small>
            </div>
            <div className="eval-detail-metrics">
              {metricDefsForGroup(group.metricKeys).map((metric) => (
                <div key={metric.key}>
                  <span>{metric.labelCn}</span>
                  <small className="eval-detail-metric-en">{metric.labelEn}</small>
                  <strong>{formatScore(result.metrics[metric.key])}</strong>
                  <small>{result.reasons[metric.key] || "未返回原因"}</small>
                </div>
              ))}
            </div>
          </div>
        ))}
      </section>

      <section>
        <label className="eval-field">
          <span>问题</span>
          <textarea readOnly value={result.user_input} />
        </label>
        <label className="eval-field">
          <span>新跑回答</span>
          <textarea readOnly value={result.response} />
        </label>
        <label className="eval-field">
          <span>标准答案</span>
          <textarea readOnly value={result.reference || "未填写"} />
        </label>
      </section>

      <section>
        <div className="eval-detail-title">
          <FileText size={13} />
          Retrieved Contexts
        </div>
        <div className="eval-context-list">
          {result.citations.map((citation, index) => (
            <article key={`${citation.chunk_id}-${index}`}>
              <header>
                <span>{citation.filename}</span>
                <strong>{citation.score.toFixed(3)}</strong>
              </header>
              <p>{citation.content}</p>
            </article>
          ))}
        </div>
      </section>

      <section>
        <div className="eval-detail-title">
          <Check size={13} />
          Retrieval Trace
        </div>
        <pre className="eval-trace-json">
          {JSON.stringify(result.retrieval_trace, null, 2)}
        </pre>
      </section>
    </div>
  );
}
