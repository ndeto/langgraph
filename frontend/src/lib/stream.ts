import {
  currentChatEventSchema,
  currentIngestionEventSchema,
  futureIngestionEventSchema,
  futureChatEventSchema,
} from "./schemas";
import type { ChatEvent, IngestionEvent } from "./types";

export async function streamNdjson<T>(
  response: Response,
  onEvent: (value: T) => void,
  mapLine: (line: string) => T,
): Promise<void> {
  if (!response.body) {
    throw new Error("Streaming responses are not supported in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    let splitIndex = buffer.indexOf("\n");
    while (splitIndex !== -1) {
      const line = buffer.slice(0, splitIndex).trim();
      buffer = buffer.slice(splitIndex + 1);
      if (line) {
        onEvent(mapLine(line));
      }
      splitIndex = buffer.indexOf("\n");
    }

    if (done) {
      break;
    }
  }

  const trailing = buffer.trim();
  if (trailing) {
    onEvent(mapLine(trailing));
  }
}

export async function streamSse<T>(
  response: Response,
  onEvent: (value: T) => void,
  mapData: (data: string) => T,
): Promise<void> {
  if (!response.body) {
    throw new Error("Streaming responses are not supported in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let dataLines: string[] = [];

  function flushEvent() {
    if (!dataLines.length) {
      return;
    }
    onEvent(mapData(dataLines.join("\n")));
    dataLines = [];
  }

  function processLine(line: string) {
    if (!line) {
      flushEvent();
      return;
    }

    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    let splitIndex = buffer.indexOf("\n");
    while (splitIndex !== -1) {
      const rawLine = buffer.slice(0, splitIndex);
      buffer = buffer.slice(splitIndex + 1);
      const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
      processLine(line);

      splitIndex = buffer.indexOf("\n");
    }

    if (done) {
      break;
    }
  }

  if (buffer) {
    const trailingLines = buffer.split("\n");
    for (const rawLine of trailingLines) {
      const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
      processLine(line);
    }
  }
  flushEvent();
}

export function parseChatEventLine(line: string): ChatEvent {
  const parsed = JSON.parse(line) as unknown;
  if (typeof parsed === "object" && parsed !== null) {
    const eventType = (parsed as { type?: unknown }).type;

    switch (eventType) {
      case "final":
        return { type: "done" };
      case "done": {
        const doneEvent = parsed as { message_id?: unknown };
        return {
          type: "done",
          messageId: typeof doneEvent.message_id === "string" ? doneEvent.message_id : undefined,
        };
      }
      case "usage": {
        const usageEvent = parsed as {
          questions_remaining?: unknown;
          input_tokens?: unknown;
          output_tokens?: unknown;
          total_tokens?: unknown;
        };
        return {
          type: "usage",
          questionsRemaining:
            typeof usageEvent.questions_remaining === "number"
              ? usageEvent.questions_remaining
              : undefined,
          inputTokens:
            typeof usageEvent.input_tokens === "number" ? usageEvent.input_tokens : undefined,
          outputTokens:
            typeof usageEvent.output_tokens === "number" ? usageEvent.output_tokens : undefined,
          totalTokens:
            typeof usageEvent.total_tokens === "number" ? usageEvent.total_tokens : undefined,
        };
      }
      case "error": {
        const errorEvent = parsed as { code?: unknown; text?: unknown };
        return {
          type: "error",
          code: typeof errorEvent.code === "string" ? errorEvent.code : undefined,
          text: typeof errorEvent.text === "string" ? errorEvent.text : "Request failed.",
        };
      }
      case "cancelled": {
        const cancelledEvent = parsed as { text?: unknown };
        return {
          type: "cancelled",
          text: typeof cancelledEvent.text === "string" ? cancelledEvent.text : undefined,
        };
      }
      default:
        break;
    }
  }

  const future = futureChatEventSchema.safeParse(parsed);
  if (future.success) {
    switch (future.data.type) {
      case "status":
        return {
          type: "status",
          text: future.data.text,
          stage: future.data.stage,
        };
      case "token":
        return { type: "token", text: future.data.text };
      case "sources":
        return {
          type: "sources",
          assets: future.data.assets.map((asset) => ({
            assetId: asset.asset_id,
            mimeType: asset.mime_type,
          })),
        };
      case "usage":
        return {
          type: "usage",
          questionsRemaining: future.data.questions_remaining,
          inputTokens: future.data.input_tokens,
          outputTokens: future.data.output_tokens,
          totalTokens: future.data.total_tokens,
        };
      case "done":
        return { type: "done", messageId: future.data.message_id };
      case "error":
        return {
          type: "error",
          code: future.data.code,
          text: future.data.text,
        };
      case "cancelled":
        return { type: "cancelled", text: future.data.text };
    }
  }

  const current = currentChatEventSchema.parse(parsed);
  switch (current.type) {
    case "status":
      return { type: "status", text: current.text };
    case "token":
      return { type: "token", text: current.text };
    case "rag_images":
      return { type: "legacy_markdown", markdown: current.markdown };
    case "done":
      return { type: "done" };
  }
}

export function parseIngestionEventLine(
  line: string,
  context?: { documentId?: string; fileName?: string },
): IngestionEvent {
  const parsed = JSON.parse(line) as unknown;
  const future = futureIngestionEventSchema.safeParse(parsed);
  if (future.success) {
    switch (future.data.type) {
      case "queued":
        return {
          type: "state",
          state: "uploading",
          text: future.data.text ?? "Document queued.",
        };
      case "validating":
        return {
          type: "state",
          state: "validating",
          text:
            future.data.text ??
            (future.data.file_name
              ? `Validating ${future.data.file_name}`
              : "Validating document."),
        };
      case "processing":
        return {
          type: "state",
          state: "processing",
          text: future.data.text ?? "Processing document.",
        };
      case "storing":
        return {
          type: "stats",
          elements: future.data.elements,
          chunks: future.data.chunks,
          docs: future.data.docs,
        };
      case "ready":
        return {
          type: "done",
          documentId: context?.documentId,
          fileName: future.data.file_name ?? context?.fileName,
          elements: future.data.elements,
          chunks: future.data.chunks,
          docs: future.data.docs,
          text: future.data.text,
        };
      case "failed":
        return {
          type: "error",
          code: future.data.code,
          text: future.data.text ?? "Document processing failed.",
        };
    }
  }

  const current = currentIngestionEventSchema.parse(parsed);
  switch (current.type) {
    case "file":
      return { type: "file", fileName: current.file_name };
    case "log":
      return { type: "state", state: "processing", text: current.text };
    case "stats":
      return {
        type: "stats",
        elements: current.elements,
        chunks: current.chunks,
        docs: current.docs,
      };
    case "done":
      return {
        type: "done",
        documentId: context?.documentId,
        fileName: current.file_name,
        elements: current.elements,
        chunks: current.chunks,
        docs: current.docs,
        text: current.text,
      };
  }
}
