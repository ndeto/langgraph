import { QuotaStrip } from "./QuotaStrip";

import type { DocumentSummary, SessionData, UploadState } from "../lib/types";

type DocumentStageCardProps = {
  session: SessionData;
  activeDocument: DocumentSummary | null;
  upload: UploadState;
  open: boolean;
  pinned: boolean;
  onClose: () => void;
};

export function DocumentStageCard({
  session,
  activeDocument,
  upload,
  open,
  pinned,
  onClose,
}: DocumentStageCardProps) {
  const latestLog = upload.logs[upload.logs.length - 1];
  const expiresAt = new Date(session.expiresAt).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <section className={`document-stage${open ? " document-stage-open" : ""}`}>
      <div className="card-head">
        <div>
          <p className="panel-label">Panel</p>
          <h2>Ingestion</h2>
        </div>
        {pinned ? (
          <button className="panel-close" type="button" onClick={onClose}>
            <span className="sr-only">Close panel</span>
          </button>
        ) : null}
      </div>

      <QuotaStrip session={session} />

      <div className="session-chip-row">
        <span className="session-chip mono">Session {session.userId.slice(0, 8)}</span>
        <span className="session-chip">Expires {expiresAt}</span>
        {activeDocument ? <span className="session-chip">{activeDocument.name}</span> : null}
      </div>

      <article className="panel-card logs-card">
        <div className="compact-status-head">
          <div>
            <p className="panel-label">Ingestion state</p>
            <p className="panel-value">{upload.status}</p>
          </div>
          {activeDocument?.chunks ? (
            <span className="session-chip">{activeDocument.chunks} chunks</span>
          ) : null}
        </div>
        <p className="panel-subtle">
          {upload.errorMessage ?? latestLog ?? "Waiting for upload."}
        </p>
        <div className="logs-box">
          {upload.logs.length ? upload.logs.join("\n") : "Processing logs appear here."}
        </div>
      </article>
    </section>
  );
}
