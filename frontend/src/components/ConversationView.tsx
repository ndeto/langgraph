import { useEffect, useRef } from "react";

import { Composer } from "./Composer";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

import type { ConversationMessage } from "../lib/types";

type ConversationViewProps = {
  messages: ConversationMessage[];
  streamStage: string | null;
  uploadNotice: { tone: "working" | "ready" | "error"; text: string } | null;
  disabled: boolean;
  onSend: (message: string) => Promise<void> | void;
  onPickFile: (file: File) => void;
  onTogglePanel: () => void;
  panelOpen: boolean;
};

export function ConversationView({
  messages,
  streamStage,
  uploadNotice,
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
  }, [messages, streamStage, uploadNotice]);

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

          {uploadNotice ? (
            <div className={`document-notice document-notice-${uploadNotice.tone}`} role="status">
              <span className="status-pulse" aria-hidden="true" />
              <span>{uploadNotice.text}</span>
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
                      <section className="related-images" aria-label="Related images">
                        <h3>Related Images</h3>
                        <p>Depends on the quality of the uploaded document.</p>
                        <div className="asset-grid">
                          {message.assets.map((asset) => (
                            <img
                              key={asset.assetId}
                              src={assetSource(asset.assetId)}
                              alt="Related document image"
                              loading="lazy"
                            />
                          ))}
                        </div>
                      </section>
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
        <a href="mailto:martin@beav3r.ai">martin@beav3r.ai</a>. Read the{" "}
        <a href="https://medium.com/@ndeto/building-ai-agents-graph-loops-memory-retrieval-rag-bf2490930834">
          article
        </a>{" "}
        or view the{" "}
        <a href="https://github.com/ndeto/langgraph">source code</a>.
      </p>
    </div>
  );
}
