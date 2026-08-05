import { sessionSchema, serverThreadSchema } from "./schemas";
import { parseChatEventLine, parseIngestionEventLine, streamNdjson } from "./stream";
import type {
  ChatEvent,
  DocumentSummary,
  IngestionEvent,
  SessionData,
  ThreadSummary,
} from "./types";

const MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024;

type StreamHandlers = {
  onEvent: (event: ChatEvent) => void;
  signal?: AbortSignal;
};

type UploadHandlers = {
  onEvent: (event: IngestionEvent) => void;
  signal?: AbortSignal;
};

function mapSession(payload: ReturnType<typeof sessionSchema.parse>): SessionData {
  const activeDocument =
    typeof payload.active_document === "string"
      ? {
          id: payload.active_document,
          name: "Active document",
          status: "ready" as DocumentSummary["status"],
        }
      : payload.active_document
        ? {
            id: payload.active_document.id ?? `doc-${payload.user_id}`,
            name: payload.active_document.name ?? "Active document",
            status:
              (payload.active_document.status as DocumentSummary["status"]) ?? "ready",
          }
        : null;

  const activeThread =
    typeof payload.active_thread === "string"
      ? {
          id: payload.active_thread,
          mode: "server" as const,
          documentId: null,
        }
      : payload.active_thread
        ? {
            id:
              payload.active_thread.thread_id ??
              payload.active_thread.id ??
              `thread-${payload.user_id}`,
            mode: "server" as const,
            documentId: payload.active_thread.document_id ?? null,
          }
        : null;

  return {
    userId: payload.user_id,
    expiresAt: payload.expires_at,
    activeDocument,
    activeThread,
    quota: payload.quota,
  };
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
  } catch {
    return `Request failed with status ${response.status}.`;
  }

  return `Request failed with status ${response.status}.`;
}

function makeLegacyThread(): ThreadSummary {
  return {
    id: globalThis.crypto?.randomUUID?.() ?? `thread-${Date.now()}`,
    mode: "legacy",
    documentId: null,
  };
}

export class AtlasApiClient {
  async getSession(): Promise<SessionData> {
    const response = await fetch("/api/v1/session", {
      credentials: "same-origin",
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    return mapSession(sessionSchema.parse(await response.json()));
  }

  async createThread(documentId: string | null = null): Promise<ThreadSummary> {
    try {
      const response = await fetch("/api/v1/threads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(documentId ? { document_id: documentId } : {}),
      });

      if (response.status === 404 || response.status === 405) {
        return makeLegacyThread();
      }

      if (!response.ok) {
        throw new Error(await readError(response));
      }

      const payload = serverThreadSchema.parse(await response.json());
      return {
        id: payload.thread_id ?? payload.id ?? makeLegacyThread().id,
        mode: "server",
        documentId: payload.document_id ?? documentId,
      };
    } catch (error) {
      if (error instanceof TypeError) {
        return makeLegacyThread();
      }
      throw error;
    }
  }

  async streamMessage(
    thread: ThreadSummary,
    userInput: string,
    handlers: StreamHandlers,
  ): Promise<void> {
    if (thread.mode === "server") {
      const response = await fetch(`/api/v1/threads/${thread.id}/messages`, {
        method: "POST",
        headers: {
          Accept: "application/x-ndjson",
          "Content-Type": "application/json",
        },
        credentials: "same-origin",
        body: JSON.stringify({ user_input: userInput }),
        signal: handlers.signal,
      });

      if (response.status === 404 || response.status === 405) {
        return this.streamLegacyMessage(thread, userInput, handlers);
      }

      if (!response.ok) {
        throw new Error(await readError(response));
      }

      return streamNdjson(response, handlers.onEvent, parseChatEventLine);
    }

    return this.streamLegacyMessage(thread, userInput, handlers);
  }

  private async streamLegacyMessage(
    thread: ThreadSummary,
    userInput: string,
    handlers: StreamHandlers,
  ): Promise<void> {
    const response = await fetch("/invoke?stream_format=ndjson", {
      method: "POST",
      headers: {
        Accept: "application/x-ndjson",
        "Content-Type": "application/json",
      },
      credentials: "same-origin",
      body: JSON.stringify({ user_input: userInput, thread_id: thread.id }),
      signal: handlers.signal,
    });

    if (!response.ok) {
      throw new Error(await readError(response));
    }

    await streamNdjson(response, handlers.onEvent, parseChatEventLine);
  }

  async uploadDocument(file: File, handlers: UploadHandlers): Promise<void> {
    if (file.size > MAX_UPLOAD_SIZE_BYTES) {
      throw new Error("PDF exceeds the 25 MB upload limit.");
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/ingest/pdf?stream_format=ndjson", {
      method: "POST",
      headers: { Accept: "application/x-ndjson" },
      credentials: "same-origin",
      body: formData,
      signal: handlers.signal,
    });

    if (!response.ok) {
      throw new Error(await readError(response));
    }

    await streamNdjson(response, handlers.onEvent, parseIngestionEventLine);
  }
}

export const atlasApi = new AtlasApiClient();
