import { useMemo, useState } from "react";

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
  const [isPanelPinned, setIsPanelPinned] = useState(false);
  const sessionData = session.state.status === "ready" ? session.state.data : null;
  const documentForChat = localDocument ?? sessionData?.activeDocument ?? null;
  const conversation = useConversation({
    activeThread: sessionData?.activeThread ?? null,
    documentId: documentForChat?.id ?? null,
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
      conversation.resetConversation();
      void session.refresh();
    },
  });

  const activeDocument = useMemo<DocumentSummary | null>(
    () => upload.state.document ?? localDocument ?? sessionData?.activeDocument ?? null,
    [localDocument, sessionData?.activeDocument, upload.state.document],
  );
  const isIngesting = ["validating", "uploading", "processing"].includes(upload.state.status);
  const showPanel = isPanelPinned || isIngesting;

  function handlePickFile(file: File) {
    const shouldReplace =
      activeDocument &&
      globalThis.confirm(
        `Replace ${activeDocument.name}? A successful upload should start a fresh document-bound thread.`,
      ) === false;

    if (shouldReplace) {
      return;
    }

    void upload.beginUpload(file);
  }

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
      <div className={`page-frame${showPanel ? " page-frame-with-panel" : ""}`}>
        <section className="main-stage main-stage-editorial">
          <HeroHeader hasMessages={conversation.messages.length > 0} />
          <ConversationView
            messages={conversation.messages}
            streamStage={conversation.streamStage}
            disabled={conversation.isStreaming}
            onSend={conversation.sendMessage}
            onPickFile={handlePickFile}
            onTogglePanel={() => setIsPanelPinned((current) => !current)}
            panelOpen={showPanel}
          />
        </section>

        <aside className={`dock-stage${showPanel ? " dock-stage-open" : ""}`}>
          <DocumentStageCard
            session={session.state.data}
            activeDocument={activeDocument}
            upload={upload.state}
            open={showPanel}
            pinned={isPanelPinned}
            onClose={() => setIsPanelPinned(false)}
          />
        </aside>
      </div>
    </main>
  );
}

export default App;
