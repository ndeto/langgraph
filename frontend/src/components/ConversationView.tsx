import { useEffect, useRef } from "react";

import { Composer } from "./Composer";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

import type { ConversationMessage } from "../lib/types";

type ConversationViewProps = {
  messages: ConversationMessage[];
  streamStage: string | null;
  disabled: boolean;
  onSend: (message: string) => Promise<void> | void;
  onPickFile: (file: File) => void;
  onTogglePanel: () => void;
  panelOpen: boolean;
};

export function ConversationView({
  messages,
  streamStage,
  disabled,
  onSend,
  onPickFile,
  onTogglePanel,
  panelOpen,
}: ConversationViewProps) {
  const conversationEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, streamStage]);

  function transformMarkdownUrl(url: string) {
    if (url.startsWith("data:image/")) {
      return url;
    }

    return defaultUrlTransform(url);
  }

  function assetSource(assetId: string) {
    if (assetId.startsWith("data:image/")) {
      return assetId;
    }

    return `/api/v1/assets/${assetId}`;
  }

  return (
    <div className="conversation-shell">
      <section className="conversation-panel" aria-label="Conversation">
        <div className="conversation-toolbar">
          <button className="panel-toggle" type="button" onClick={onTogglePanel}>
            <span className="sr-only">{panelOpen ? "Hide panel" : "Open panel"}</span>
            <span className="panel-toggle-lines" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
          </button>
        </div>

        <div className="conversation-list">
          {messages.length === 0 ? (
            <div className="empty-state empty-state-chat">
              <h3>Start with a question or attach a PDF first.</h3>
            </div>
          ) : null}

          {messages.map((message) => {
            const assistantBody = message.content || "";
            const showAssistantMessage =
              message.role !== "assistant" ||
              assistantBody.trim().length > 0 ||
              message.assets.length > 0;

            if (!showAssistantMessage) {
              return null;
            }

            return (
              <article key={message.id} className={`message message-${message.role}`}>
                <span className="message-label">
                  {message.role === "user" ? "You" : "Atlas AI"}
                </span>
                {message.role === "assistant" ? (
                  <div className="message-content">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      urlTransform={transformMarkdownUrl}
                    >
                      {assistantBody}
                    </ReactMarkdown>
                    {message.assets.length ? (
                      <div className="asset-grid">
                        {message.assets.map((asset) => (
                          <img
                            key={asset.assetId}
                            src={assetSource(asset.assetId)}
                            alt="Retrieved source"
                            loading="lazy"
                          />
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <div className="message-content">{message.content}</div>
                )}
              </article>
            );
          })}

          {streamStage ? (
            <div className="agent-status" role="status">
              <span className="status-pulse" />
              <span className="status-text">{streamStage}</span>
            </div>
          ) : null}

          <div ref={conversationEndRef} aria-hidden="true" />
        </div>

        <Composer disabled={disabled} onSend={onSend} onPickFile={onPickFile} />
      </section>

      <p className="conversation-note">
        This is an anonymous session. Documents are deleted after they are indexed,
        and all data expires within 24 hours. Send any concerns to{" "}
        <a href="mailto:martin@beav3r.ai">martin@beav3r.ai</a>.
      </p>
    </div>
  );
}
