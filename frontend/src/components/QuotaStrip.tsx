import type { SessionData } from "../lib/types";

type QuotaStripProps = {
  session: SessionData;
};

export function QuotaStrip({ session }: QuotaStripProps) {
  return (
    <section className="quota-strip" aria-label="Usage summary">
      <p className="quota-inline">
        <span className="quota-label">Questions</span>{" "}
        <span className="quota-meta">
          {session.quota.questions.remaining} left • {session.quota.questions.used}/
          {session.quota.questions.limit}
        </span>
      </p>
      <p className="quota-inline">
        <span className="quota-label">Uploads</span>{" "}
        <span className="quota-meta">
          {session.quota.uploads.remaining} left • {session.quota.uploads.used}/
          {session.quota.uploads.limit}
        </span>
      </p>
      <p className="quota-inline">
        <span className="quota-label">Tokens</span>{" "}
        <span className="quota-meta">
          {session.quota.tokens.total} total • {session.quota.tokens.input} in /{" "}
          {session.quota.tokens.output} out
        </span>
      </p>
    </section>
  );
}
