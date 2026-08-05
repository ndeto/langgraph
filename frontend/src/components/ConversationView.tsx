import { Composer } from "./Composer";
import ReactMarkdown from "react-markdown";
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
  return (
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

        {messages.map((message) => (
          <article key={message.id} className={`message message-${message.role}`}>
            <span className="message-label">
              {message.role === "user" ? "You" : "Atlas AI"}
            </span>
            {message.role === "assistant" ? (
              <div className="message-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content || message.stageText || ""}
                </ReactMarkdown>
                {message.assets.length ? (
                  <div className="asset-grid">
                    {message.assets.map((asset) => (
                      <img
                        key={asset.assetId}
                        src={`/api/v1/assets/${asset.assetId}`}
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
        ))}

        {streamStage ? (
          <div className="agent-status" role="status">
            <span className="status-pulse" />
            <span className="status-text">{streamStage}</span>
          </div>
        ) : null}
      </div>

      <Composer disabled={disabled} onSend={onSend} onPickFile={onPickFile} />
    </section>
  );
}
