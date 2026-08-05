export type QuotaBucket = {
  limit: number;
  used: number;
  remaining: number;
};

export type TokenUsage = {
  input: number;
  output: number;
  total: number;
};

export type SessionData = {
  userId: string;
  expiresAt: string;
  activeDocument: DocumentSummary | null;
  activeThread: ThreadSummary | null;
  quota: {
    questions: QuotaBucket;
    uploads: QuotaBucket;
    tokens: TokenUsage;
  };
};

export type ThreadSummary = {
  id: string;
  mode: "server" | "legacy";
  documentId: string | null;
};

export type DocumentSummary = {
  id: string;
  name: string;
  status: "idle" | "uploading" | "processing" | "ready" | "error";
  sizeBytes?: number;
  elements?: number;
  chunks?: number;
  docs?: number;
  message?: string;
};

export type AssetRef = {
  assetId: string;
  mimeType: string;
};

export type ConversationMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "streaming" | "done" | "error";
  assets: AssetRef[];
  stageText?: string;
};

export type ChatEvent =
  | { type: "status"; stage?: string; text: string }
  | { type: "token"; text: string }
  | { type: "sources"; assets: AssetRef[] }
  | {
      type: "usage";
      questionsRemaining?: number;
      inputTokens?: number;
      outputTokens?: number;
      totalTokens?: number;
    }
  | { type: "done"; messageId?: string }
  | { type: "error"; code?: string; text: string }
  | { type: "cancelled"; text?: string }
  | { type: "legacy_markdown"; markdown: string };

export type IngestionEvent =
  | { type: "state"; state: string; text: string }
  | { type: "file"; fileName: string }
  | { type: "stats"; elements?: number; chunks?: number; docs?: number }
  | {
      type: "done";
      documentId?: string;
      jobId?: string;
      fileName?: string;
      elements?: number;
      chunks?: number;
      docs?: number;
      text?: string;
    }
  | { type: "error"; code?: string; text: string };

export type SessionState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: SessionData };

export type UploadState = {
  status: "idle" | "validating" | "uploading" | "processing" | "ready" | "error";
  fileName: string | null;
  fileSizeBytes: number | null;
  logs: string[];
  document: DocumentSummary | null;
  errorMessage: string | null;
};

