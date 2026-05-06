import { useEffect, useMemo, useState } from "react";
import {
  type DocumentChunkPreview,
  type DocumentRecord,
  previewDocumentChunks,
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
  onSaved: SaveHandler;
};

export type IngestionStep = "source" | "processing";

export function useDocumentIngestionWizard({
  activeKbId,
  embeddingModel,
  initialFile,
  onInitialFileConsumed,
  onSaved,
}: UseDocumentIngestionWizardOptions) {
  const [step, setStep] = useState<IngestionStep>("source");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
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
    onInitialFileConsumed();
  }, [initialFile, onInitialFileConsumed]);

  const canPreview = useMemo(
    () => Boolean(activeKbId && selectedFile && !previewing && !saving),
    [activeKbId, selectedFile, previewing, saving],
  );

  async function previewChunks() {
    if (!activeKbId || !selectedFile) return;
    setPreviewing(true);
    setError("");
    try {
      const result = await previewDocumentChunks(
        activeKbId,
        selectedFile,
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
    if (!activeKbId || !selectedFile) return;
    setSaving(true);
    setError("");
    try {
      const document = await uploadDocument(
        activeKbId,
        selectedFile,
        parser,
        embeddingModel,
        chunkingPolicy,
      );
      onSaved(document, chunkingPolicy, embeddingModel);
      resetWizard();
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "保存失败");
    } finally {
      setSaving(false);
    }
  }

  function chooseFile(file: File | null) {
    setSelectedFile(file);
    setPreview(null);
    setError("");
  }

  function goToProcessing() {
    if (!selectedFile) return;
    setStep("processing");
  }

  function resetWizard() {
    setStep("source");
    setSelectedFile(null);
    setPreview(null);
    setError("");
  }

  return {
    canPreview,
    chooseFile,
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
    setChunkingPolicy,
    setParser,
    setStep,
    step,
  };
}
