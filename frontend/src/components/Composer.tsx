import { ChangeEvent, FormEvent, KeyboardEvent, useRef, useState } from "react";

type ComposerProps = {
  disabled: boolean;
  onSend: (message: string) => Promise<void> | void;
  onPickFile: (file: File) => void;
};

export function Composer({ disabled, onSend, onPickFile }: ComposerProps) {
  const [value, setValue] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextValue = value.trim();
    if (!nextValue || disabled) {
      return;
    }
    setValue("");
    await onSend(nextValue);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }

    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }

  function handleFileSelect(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.item(0);
    if (file) {
      onPickFile(file);
    }
    event.target.value = "";
  }

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        className="sr-only"
        onChange={handleFileSelect}
      />
      <label className="sr-only" htmlFor="chat-input">
        Message Atlas AI
      </label>
      <textarea
        id="chat-input"
        className="chat-input"
        rows={2}
        placeholder="Start with a question or attach a PDF first."
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />
      <div className="composer-meta">
        <div className="composer-actions">
          <button
            className="attachment-button"
            type="button"
            onClick={() => fileInputRef.current?.click()}
          >
            Attach File
          </button>
          <button className="send-button" type="submit" disabled={disabled}>
            {disabled ? "Working" : "Send"}
          </button>
        </div>
      </div>
    </form>
  );
}
