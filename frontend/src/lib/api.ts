import { documentAcceptedSchema, sessionSchema, serverThreadSchema } from "./schemas";
import { parseChatEventLine, parseIngestionEventLine, streamNdjson, streamSse } from "./stream";
import { getClientId } from "./clientIdentity";
import type {
  ChatEvent,
  DocumentAccepted,
  DocumentSummary,
  IngestionEvent,
  SessionData,
  ThreadDetail,
  ThreadSummary,
} from "./types";

const MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024;
const MAX_USER_INPUT_CHARS = 8000;
const USER_INPUT_TOO_LARGE_MESSAGE = "Message exceeds the 8,000 character limit.";

type StreamHandlers = {
  onEvent: (event: ChatEvent) => void;
  signal?: AbortSignal;
};

type UploadHandlers = {
  onEvent: (event: IngestionEvent) => void;
  signal?: AbortSignal;
};

type ErrorDetail = {
  type?: unknown;
  loc?: unknown;
  msg?: unknown;
  ctx?: unknown;
};

function isUserInputTooLargeDetail(detail: unknown): boolean {
  if (!Array.isArray(detail)) {
    return false;
  }

  return detail.some((item: ErrorDetail) => {
    const loc = Array.isArray(item.loc) ? item.loc.map(String) : [];
    const ctx = item.ctx && typeof item.ctx === "object" ? item.ctx : {};
    const maxLength =
      "max_length" in ctx && typeof ctx.max_length === "number"
        ? ctx.max_length
        : null;

    return (
      loc.includes("user_input") &&
      (String(item.type).includes("too_long") || maxLength === MAX_USER_INPUT_CHARS)
    );
  });
}

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
    uploadedDocuments: payload.uploaded_documents.map((document) => ({
      id: document.id,
      name: document.name,
      status: document.status as DocumentSummary["status"],
    })),
    activeThread,
    quota: payload.quota,
  };
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown; message?: unknown };
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (isUserInputTooLargeDetail(payload.detail)) {
      return USER_INPUT_TOO_LARGE_MESSAGE;
    }
    if (typeof payload.message === "string") {
      return payload.message;
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

function buildHeaders(headers?: HeadersInit): HeadersInit {
  const merged = new Headers(headers);
  merged.set("X-Atlas-Client-Key", getClientId());
  return merged;
}

async function atlasFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  return fetch(input, {
    ...init,
    headers: buildHeaders(init?.headers),
    credentials: "same-origin",
  });
}

export class AtlasApiClient {
  async getSession(): Promise<SessionData> {
    const response = await atlasFetch("/api/v1/session");
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    return mapSession(sessionSchema.parse(await response.json()));
  }

  async rotateSession(): Promise<SessionData> {
    const response = await atlasFetch("/api/v1/session/rotate", {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    return mapSession(sessionSchema.parse(await response.json()));
  }

  async createThread(documentId: string | null = null): Promise<ThreadSummary> {
    try {
      const response = await atlasFetch("/api/v1/threads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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

  async getThread(threadId: string): Promise<ThreadDetail> {
    const response = await atlasFetch(`/api/v1/threads/${threadId}`);
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const payload = serverThreadSchema.parse(await response.json());
    return {
      id: payload.thread_id ?? payload.id ?? threadId,
      mode: "server",
      documentId: payload.document_id ?? null,
      createdAt: payload.created_at,
      expiresAt: payload.expires_at,
      messages: (payload.messages ?? []).map((message) => ({
        messageId: message.message_id,
        role: message.role,
        content: message.content,
        createdAt: message.created_at,
        assets: (message.assets ?? []).map((asset) => ({
          assetId: asset.asset_id,
          mimeType: asset.mime_type,
        })),
        status: message.status ?? "done",
      })),
    };
  }

  async streamMessage(
    thread: ThreadSummary,
    userInput: string,
    handlers: StreamHandlers,
  ): Promise<void> {
    if (userInput.length > MAX_USER_INPUT_CHARS) {
      throw new Error(USER_INPUT_TOO_LARGE_MESSAGE);
    }

    if (thread.mode === "server") {
      const response = await atlasFetch(`/api/v1/threads/${thread.id}/messages`, {
        method: "POST",
        headers: {
          Accept: "application/x-ndjson",
          "Content-Type": "application/json",
        },
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
    if (userInput.length > MAX_USER_INPUT_CHARS) {
      throw new Error(USER_INPUT_TOO_LARGE_MESSAGE);
    }

    const response = await atlasFetch("/invoke?stream_format=ndjson", {
      method: "POST",
      headers: {
        Accept: "application/x-ndjson",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ user_input: userInput, thread_id: thread.id }),
      signal: handlers.signal,
    });

    if (!response.ok) {
      throw new Error(await readError(response));
    }

    await streamNdjson(response, handlers.onEvent, parseChatEventLine);
  }

  async uploadDocument(file: File, handlers: UploadHandlers): Promise<DocumentAccepted> {
    if (file.size > MAX_UPLOAD_SIZE_BYTES) {
      throw new Error("PDF exceeds the 10 MB upload limit.");
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await atlasFetch("/api/v1/documents", {
      method: "POST",
      body: formData,
      signal: handlers.signal,
    });

    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const acceptedPayload = documentAcceptedSchema.parse(await response.json());
    const accepted: DocumentAccepted = {
      documentId: acceptedPayload.document_id,
      jobId: acceptedPayload.job_id,
      status: acceptedPayload.status,
    };

    const eventsResponse = await atlasFetch(`/api/v1/ingestions/${accepted.jobId}/events`, {
      headers: { Accept: "text/event-stream" },
      signal: handlers.signal,
    });

    if (!eventsResponse.ok) {
      throw new Error(await readError(eventsResponse));
    }

    await streamSse(eventsResponse, handlers.onEvent, (data) =>
      parseIngestionEventLine(data, {
        documentId: accepted.documentId,
        fileName: file.name,
      }),
    );

    return accepted;
  }
}

export const atlasApi = new AtlasApiClient();
