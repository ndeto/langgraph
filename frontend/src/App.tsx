import { useEffect, useMemo, useState } from "react";

import { ConversationView } from "./components/ConversationView";
import { DocumentStageCard } from "./components/DocumentStageCard";
import { HeroHeader } from "./components/HeroHeader";
import { useConversation } from "./hooks/useConversation";
import { useDocumentUpload } from "./hooks/useDocumentUpload";
import { useSession } from "./hooks/useSession";
import type { DocumentSummary } from "./lib/types";

function App() {
  const session = useSession();
  const [localDocument, setLocalDocument] = useState<DocumentSummary | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const sessionData = session.state.status === "ready" ? session.state.data : null;
  const documentForChat = localDocument ?? sessionData?.activeDocument ?? null;
  const conversation = useConversation({
    activeThread: sessionData?.activeThread ?? null,
    documentId: null,
    onUsage(usage) {
      session.applyUsage({
        input: usage.input,
        output: usage.output,
        total: usage.total,
      });
      if (typeof usage.questionsRemaining === "number") {
        void session.refresh();
      }
    },
    onConversationSettled() {
      void session.refresh();
    },
  });
  const upload = useDocumentUpload({
    activeDocument: documentForChat,
    onUploadComplete(document) {
      setLocalDocument(document);
      void session.refresh();
    },
  });

  const activeDocument = useMemo<DocumentSummary | null>(
    () => upload.state.document ?? localDocument ?? sessionData?.activeDocument ?? null,
    [localDocument, sessionData?.activeDocument, upload.state.document],
  );
  const showPanel = isPanelOpen;

  function handlePickFile(file: File) {
    setIsPanelOpen(true);
    void upload.beginUpload(file);
  }

  async function handleRotateSession() {
    if (
      globalThis.confirm(
        "Start a fresh session? Current uploads, chat, and usage will be queued for deletion.",
      ) === false
    ) {
      return;
    }

    try {
      const nextSession = await session.rotate();
      setLocalDocument(nextSession.activeDocument);
      upload.resetUpload(nextSession.activeDocument);
      conversation.resetConversation();
      setIsPanelOpen(false);
    } catch (error) {
      globalThis.alert(
        error instanceof Error ? error.message : "Unable to start a new session.",
      );
    }
  }

  useEffect(() => {
    function handleKeyboardShortcut(event: KeyboardEvent) {
      if (
        !event.repeat &&
        !session.isRotating &&
        event.altKey &&
        event.shiftKey &&
        event.code === "KeyN"
      ) {
        event.preventDefault();
        void handleRotateSession();
      }
    }

    globalThis.addEventListener("keydown", handleKeyboardShortcut);
    return () => globalThis.removeEventListener("keydown", handleKeyboardShortcut);
  });

  if (session.state.status === "loading") {
    return (
      <main className="app-shell loading-shell">
        <div className="loading-card">Restoring anonymous session…</div>
      </main>
    );
  }

  if (session.state.status === "error") {
    return (
      <main className="app-shell loading-shell">
        <div className="loading-card">
          <h2>Session unavailable</h2>
          <p>{session.state.message}</p>
          <button className="attachment-button" type="button" onClick={() => void session.refresh()}>
            Retry
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="app-shell">
      {showPanel ? (
        <button
          className={`panel-backdrop${isPanelOpen ? " panel-backdrop-visible" : ""}`}
          type="button"
          aria-label="Close panel"
          onClick={() => setIsPanelOpen(false)}
        />
      ) : null}

      <div className="page-frame">
        <section className="main-stage main-stage-editorial">
          <HeroHeader hasMessages={conversation.messages.length > 0} />
          <ConversationView
            messages={conversation.messages}
            streamStage={conversation.streamStage}
            disabled={conversation.isStreaming}
            onSend={conversation.sendMessage}
            onPickFile={handlePickFile}
            onTogglePanel={() => setIsPanelOpen((current) => !current)}
            panelOpen={showPanel}
          />
        </section>
      </div>

      <aside className={`dock-stage${showPanel ? " dock-stage-open" : ""}`}>
        <DocumentStageCard
          session={session.state.data}
          activeDocument={activeDocument}
          upload={upload.state}
          open={showPanel}
          pinned={isPanelOpen}
          onClose={() => setIsPanelOpen(false)}
        />
      </aside>
    </main>
  );
}

export default App;
