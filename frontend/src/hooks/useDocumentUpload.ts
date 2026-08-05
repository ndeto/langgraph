import { useCallback, useEffect, useState } from "react";

import { atlasApi } from "../lib/api";
import type { DocumentSummary, IngestionEvent, UploadState } from "../lib/types";

const EMPTY_UPLOAD_STATE: UploadState = {
  status: "idle",
  fileName: null,
  fileSizeBytes: null,
  logs: [],
  document: null,
  errorMessage: null,
};

function appendLog(logs: string[], text: string | null | undefined): string[] {
  if (!text) {
    return logs;
  }

  if (logs[logs.length - 1] === text) {
    return logs;
  }

  return [...logs, text];
}

export function useDocumentUpload(options: {
  activeDocument: DocumentSummary | null;
  onUploadComplete?: (document: DocumentSummary) => void;
}) {
  const [state, setState] = useState<UploadState>({
    ...EMPTY_UPLOAD_STATE,
    document: options.activeDocument,
  });

  useEffect(() => {
    setState((current) => {
      if (current.document || !options.activeDocument) {
        return current;
      }
      return { ...current, document: options.activeDocument };
    });
  }, [options.activeDocument]);

  const applyEvent = useCallback((event: IngestionEvent) => {
    setState((current) => {
      switch (event.type) {
        case "file":
          return {
            ...current,
            status: "uploading",
            fileName: event.fileName,
            logs: appendLog(current.logs, `Queued ${event.fileName}`),
          };
        case "state":
          return {
            ...current,
            status:
              event.state === "uploading"
                ? "uploading"
                : event.state === "validating"
                  ? "validating"
                  : "processing",
            logs: appendLog(current.logs, event.text),
          };
        case "stats":
          return {
            ...current,
            status: "processing",
            logs: appendLog(current.logs, "Storing document artifacts."),
            document: current.document
              ? {
                  ...current.document,
                  elements: event.elements,
                  chunks: event.chunks,
                  docs: event.docs,
                }
              : current.document,
          };
        case "done": {
          const document: DocumentSummary = {
            id: event.documentId ?? globalThis.crypto?.randomUUID?.() ?? `doc-${Date.now()}`,
            name: event.fileName ?? current.fileName ?? "Uploaded document",
            status: "ready",
            sizeBytes: current.fileSizeBytes ?? undefined,
            elements: event.elements,
            chunks: event.chunks,
            docs: event.docs,
            message: event.text,
          };
          options.onUploadComplete?.(document);
          return {
            ...current,
            status: "ready",
            document,
            logs: appendLog(current.logs, event.text),
            errorMessage: null,
          };
        }
        case "error":
          return {
            ...current,
            status: "error",
            errorMessage: event.text,
            logs: appendLog(current.logs, event.text),
          };
      }
    });
  }, [options]);

  const beginUpload = useCallback(async (file: File) => {
    setState({
      status: "validating",
      fileName: file.name,
      fileSizeBytes: file.size,
      logs: ["Preparing upload."],
      document: options.activeDocument,
      errorMessage: null,
    });

    try {
      await atlasApi.uploadDocument(file, { onEvent: applyEvent });
    } catch (error) {
      setState((current) => ({
        ...current,
        status: "error",
        errorMessage: error instanceof Error ? error.message : "Upload failed.",
        logs: appendLog(
          current.logs,
          error instanceof Error ? error.message : "Upload failed.",
        ),
      }));
    }
  }, [applyEvent, options.activeDocument]);

  const resetUpload = useCallback((document: DocumentSummary | null = null) => {
    setState({
      ...EMPTY_UPLOAD_STATE,
      document,
    });
  }, []);

  return {
    state,
    beginUpload,
    resetUpload,
  };
}
