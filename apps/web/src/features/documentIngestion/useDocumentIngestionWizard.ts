import { useEffect, useMemo, useState } from "react";
import {
  type DocumentChunkPreview,
  type DocumentRecord,
  previewDocumentChunks,
  previewDocumentReindex,
  reindexDocument,
  uploadDocument,
} from "../../lib/api";
import {
  type ChunkingPolicyInput,
  DEFAULT_CHUNKING_POLICY,
} from "../../lib/chunkingPolicy";

type SaveHandler = (
  document: DocumentRecord,
  chunkingPolicy: ChunkingPolicyInput,
  embeddingModel: string,
) => void;

type UseDocumentIngestionWizardOptions = {
  activeKbId: string;
  embeddingModel: string;
  initialFile: File | null;
  onInitialFileConsumed: () => void;
  onDocumentParametersReset: () => void;
  onSaved: SaveHandler;
};

export type IngestionStep = "source" | "processing";

export function useDocumentIngestionWizard({
  activeKbId,
  embeddingModel,
  initialFile,
  onInitialFileConsumed,
  onDocumentParametersReset,
  onSaved,
}: UseDocumentIngestionWizardOptions) {
  const [step, setStep] = useState<IngestionStep>("source");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [reindexingDocument, setReindexingDocument] = useState<DocumentRecord | null>(null);
  const [parser, setParser] = useState("auto");
  const [chunkingPolicy, setChunkingPolicy] =
    useState<ChunkingPolicyInput>(DEFAULT_CHUNKING_POLICY);
  const [preview, setPreview] = useState<DocumentChunkPreview | null>(null);
  const [error, setError] = useState("");
  const [previewing, setPreviewing] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!initialFile) return;
    setSelectedFile(initialFile);
    setStep("source");
    setPreview(null);
    setError("");
    resetDocumentParameters();
    onInitialFileConsumed();
  }, [initialFile, onInitialFileConsumed]);

  const canPreview = useMemo(
    () => Boolean(activeKbId && (selectedFile || reindexingDocument) && !previewing && !saving),
    [activeKbId, selectedFile, reindexingDocument, previewing, saving],
  );

  async function previewChunks() {
    if (!activeKbId || (!selectedFile && !reindexingDocument)) return;
    setPreviewing(true);
    setError("");
    try {
      const result = reindexingDocument
        ? await previewDocumentReindex(
            reindexingDocument.id,
            parser,
            embeddingModel,
            chunkingPolicy,
          )
        : await previewDocumentChunks(
            activeKbId,
            selectedFile as File,
            parser,
            embeddingModel,
            chunkingPolicy,
          );
      setPreview(result);
    } catch (reason: unknown) {
      setPreview(null);
      setError(reason instanceof Error ? reason.message : "预览失败");
    } finally {
      setPreviewing(false);
    }
  }

  async function saveDocument() {
    if (!activeKbId || (!selectedFile && !reindexingDocument)) return;
    setSaving(true);
    setError("");
    try {
      const document = reindexingDocument
        ? await reindexDocument(reindexingDocument.id, parser, embeddingModel, chunkingPolicy)
        : await uploadDocument(
            activeKbId,
            selectedFile as File,
            parser,
            embeddingModel,
            chunkingPolicy,
          );
      onSaved(document, chunkingPolicy, embeddingModel);
      resetWizard({ resetParameters: true });
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "保存失败");
    } finally {
      setSaving(false);
    }
  }

  function chooseFile(file: File | null) {
    setSelectedFile(file);
    setReindexingDocument(null);
    setPreview(null);
    setError("");
    resetDocumentParameters();
  }

  function chooseDocumentForReindex(document: DocumentRecord, policy: ChunkingPolicyInput) {
    setSelectedFile(null);
    setReindexingDocument(document);
    setParser(document.parser || "auto");
    setChunkingPolicy(policy);
    setPreview(null);
    setError("");
    setStep("processing");
  }

  function goToProcessing() {
    if (!selectedFile) return;
    setStep("processing");
  }

  function resetDocumentParameters() {
    // 每个文档的导入参数独立生效，换文件或保存后回到默认配置，避免误沿用上一份文档。
    setParser("auto");
    setChunkingPolicy(DEFAULT_CHUNKING_POLICY);
    onDocumentParametersReset();
  }

  function resetWizard(options: { resetParameters?: boolean } = {}) {
    setStep("source");
    setSelectedFile(null);
    setReindexingDocument(null);
    setPreview(null);
    setError("");
    if (options.resetParameters) resetDocumentParameters();
  }

  return {
    canPreview,
    chooseFile,
    chooseDocumentForReindex,
    chunkingPolicy,
    error,
    goToProcessing,
    parser,
    preview,
    previewChunks,
    previewing,
    resetWizard,
    saveDocument,
    saving,
    selectedFile,
    reindexingDocument,
    setChunkingPolicy,
    setError,
    setParser,
    setStep,
    step,
  };
}
