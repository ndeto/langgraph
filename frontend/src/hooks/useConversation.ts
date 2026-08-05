import { useCallback, useEffect, useMemo, useState } from "react";

import { atlasApi } from "../lib/api";
import type {
  ChatEvent,
  ConversationMessage,
  ThreadSummary,
} from "../lib/types";

type Options = {
  activeThread: ThreadSummary | null;
  documentId: string | null;
  onUsage?: (usage: {
    input?: number;
    output?: number;
    total?: number;
    questionsRemaining?: number;
  }) => void;
  onConversationSettled?: () => void;
};

function createAssistantMessage(): ConversationMessage {
  return {
    id: globalThis.crypto?.randomUUID?.() ?? `assistant-${Date.now()}`,
    role: "assistant",
    content: "",
    status: "streaming",
    assets: [],
  };
}

export function useConversation(options: Options) {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [thread, setThread] = useState<ThreadSummary | null>(options.activeThread);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamStage, setStreamStage] = useState<string | null>(null);

  useEffect(() => {
    if (options.activeThread) {
      setThread(options.activeThread);
    }
  }, [options.activeThread]);

  const ensureThread = useCallback(async () => {
    if (thread) {
      return thread;
    }
    const created = await atlasApi.createThread(options.documentId);
    setThread(created);
    return created;
  }, [options.documentId, thread]);

  const resetConversation = useCallback(() => {
    setMessages([]);
    setThread(null);
    setStreamStage(null);
  }, []);

  const applyEvent = useCallback((event: ChatEvent) => {
    if (event.type === "status") {
      setStreamStage(event.text);
      setMessages((current) => {
        const lastMessage = current[current.length - 1];
        if (lastMessage?.role === "assistant" && lastMessage.status === "streaming") {
          return current.map((message, index) =>
            index === current.length - 1
              ? { ...message, stageText: event.text }
              : message,
          );
        }
        return [...current, { ...createAssistantMessage(), stageText: event.text }];
      });
      return;
    }

    if (event.type === "token") {
      setMessages((current) => {
        const last = current[current.length - 1];
        if (!last || last.role !== "assistant" || last.status !== "streaming") {
          return [
            ...current,
            { ...createAssistantMessage(), content: event.text },
          ];
        }

        return current.map((message, index) =>
          index === current.length - 1
            ? { ...message, content: message.content + event.text }
            : message,
        );
      });
      return;
    }

    if (event.type === "legacy_markdown") {
      setMessages((current) =>
        current.map((message, index) =>
          index === current.length - 1
            ? { ...message, content: message.content + event.markdown }
            : message,
        ),
      );
      return;
    }

    if (event.type === "sources") {
      setMessages((current) =>
        current.map((message, index) =>
          index === current.length - 1
            ? { ...message, assets: [...message.assets, ...event.assets] }
            : message,
        ),
      );
      return;
    }

    if (event.type === "usage") {
      options.onUsage?.({
        input: event.inputTokens,
        output: event.outputTokens,
        total: event.totalTokens,
        questionsRemaining: event.questionsRemaining,
      });
      return;
    }

    if (event.type === "done") {
      setStreamStage(null);
      setMessages((current) =>
        current.map((message, index) =>
          index === current.length - 1 ? { ...message, status: "done" } : message,
        ),
      );
      return;
    }

    const text = event.type === "cancelled" ? event.text ?? "Cancelled." : event.text;
    setStreamStage(null);
    setMessages((current) => [
      ...current,
      {
        id: globalThis.crypto?.randomUUID?.() ?? `assistant-error-${Date.now()}`,
        role: "assistant",
        content: text,
        status: "error",
        assets: [],
      },
    ]);
  }, [options]);

  const sendMessage = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isStreaming) {
      return;
    }

    setMessages((current) => [
      ...current,
      {
        id: globalThis.crypto?.randomUUID?.() ?? `user-${Date.now()}`,
        role: "user",
        content: trimmed,
        status: "done",
        assets: [],
      },
    ]);
    setIsStreaming(true);
    setStreamStage("Atlas AI is reasoning through the request.");

    const controller = new AbortController();
    try {
      const nextThread = await ensureThread();
      await atlasApi.streamMessage(nextThread, trimmed, {
        signal: controller.signal,
        onEvent: applyEvent,
      });
      options.onConversationSettled?.();
    } catch (error) {
      applyEvent({
        type: "error",
        text: error instanceof Error ? error.message : "Unable to complete the request.",
      });
    } finally {
      setIsStreaming(false);
      setStreamStage(null);
    }
  }, [applyEvent, ensureThread, isStreaming, options]);

  return useMemo(
    () => ({
      messages,
      isStreaming,
      streamStage,
      sendMessage,
      thread,
      resetConversation,
    }),
    [isStreaming, messages, resetConversation, sendMessage, streamStage, thread],
  );
}
