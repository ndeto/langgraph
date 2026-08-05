import aaiLogo from "../assets/aai-logo.svg";

type HeroHeaderProps = {
  hasMessages: boolean;
};

export function HeroHeader({ hasMessages }: HeroHeaderProps) {
  return (
    <header className={`hero-intro${hasMessages ? " hero-intro-compact" : ""}`}>
      <img className="hero-logo" src={aaiLogo} alt="" aria-hidden="true" />
      <div>
        <p className="eyebrow">Retrieval, memory, and reasoning</p>
        <h1>Atlas AI</h1>
        <p className="subtitle">
          Upload a PDF, follow the ingestion trail, and question the agent.
        </p>
      </div>
    </header>
  );
}
