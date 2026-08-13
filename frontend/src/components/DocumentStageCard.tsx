import { useEffect, useMemo, useRef } from "react";

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
  const logsRef = useRef<HTMLDivElement>(null);
  const latestLog = upload.logs[upload.logs.length - 1];
  const activeDocumentName = activeDocument
    ? session.uploadedDocuments.find((document) => document.id === activeDocument.id)?.name ??
      activeDocument.name
    : "None";
  const isProcessing = ["validating", "uploading", "processing"].includes(upload.status);
  const statusLabel = useMemo(() => {
    if (upload.status === "ready") {
      return "Done!";
    }
    if (upload.status === "error") {
      return "Error";
    }
    if (isProcessing) {
      return "Processing";
    }
    return "Idle";
  }, [isProcessing, upload.status]);
  const expiresAt = new Date(session.expiresAt).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  useEffect(() => {
    const node = logsRef.current;
    if (!node) {
      return;
    }
    node.scrollTop = node.scrollHeight;
  }, [upload.logs, latestLog]);

  return (
    <section className={`document-stage${open ? " document-stage-open" : ""}`}>
      <div className="card-head">
        <div>
          <p className="panel-label">Panel</p>
          <h2>Ingestion</h2>
        </div>
        <div className="panel-head-actions">
          {pinned ? (
            <button className="panel-close" type="button" onClick={onClose}>
              <span className="sr-only">Close panel</span>
            </button>
          ) : null}
        </div>
      </div>

      <QuotaStrip session={session} />

      <div className="panel-divider" />

      <dl className="panel-info-list">
        <div className="panel-info-row">
          <dt className="panel-info-label">Anon Session ID</dt>
          <dd className="panel-info-value mono">{session.userId.slice(0, 8)}</dd>
        </div>
        <div className="panel-info-row">
          <dt className="panel-info-label">Expires</dt>
          <dd className="panel-info-value">{expiresAt}</dd>
        </div>
        <div className="panel-info-row">
          <dt className="panel-info-label">Active document</dt>
          <dd className="panel-info-value">{activeDocumentName}</dd>
        </div>
      </dl>

      {session.uploadedDocuments.length ? (
        <>
          <div className="panel-divider" />
          <div className="panel-section">
            <p className="panel-label">Uploaded documents</p>
            <ul className="panel-document-list">
              {session.uploadedDocuments.map((document) => (
                <li key={document.id} className="panel-document-item">
                  <div className="panel-document-copy">
                    <span className="panel-document-name">{document.name}</span>
                    <span className="panel-document-status">{document.status}</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </>
      ) : null}

      <div className="panel-divider" />

      <section className="panel-section">
        <div className="panel-status-row">
          <div>
            <p className="panel-label">Ingestion state</p>
            <div className={`panel-status-display${isProcessing ? " panel-status-display-active" : ""}`}>
              {isProcessing ? <span className="panel-status-pulse" aria-hidden="true" /> : null}
              <p className="panel-value">{statusLabel}</p>
            </div>
          </div>
          {activeDocument?.chunks ? (
            <span className="panel-inline-meta">{activeDocument.chunks} chunks</span>
          ) : null}
        </div>
        <p className="panel-subtle">
          {upload.errorMessage ??
            latestLog ??
            (upload.status === "ready" ? "Document ready." : "Waiting for upload.")}
        </p>
        <div ref={logsRef} className="logs-box">
          {upload.logs.length ? upload.logs.join("\n") : "Processing logs appear here."}
        </div>
      </section>
    </section>
  );
}
