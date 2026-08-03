const panel = document.getElementById("rag-panel");
const panelToggle = document.getElementById("panel-toggle");
const panelClose = document.getElementById("panel-close");
const attachmentButton = document.getElementById("attachment-button");
const pdfInput = document.getElementById("pdf-input");
const selectedFile = document.getElementById("selected-file");
const processingLogs = document.getElementById("processing-logs");
const chatView = document.getElementById("chat-view");
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const newThreadButton = document.getElementById("new-thread");
const threadLabel = document.getElementById("thread-label");
const elementsCount = document.getElementById("elements-count");
const chunksCount = document.getElementById("chunks-count");
const docsCount = document.getElementById("docs-count");
const markdownRenderer = globalThis.markdownit
  ? globalThis.markdownit({ html: false, linkify: true, breaks: true })
  : null;
const MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024;
const PDF_MIME_TYPES = new Set(["application/pdf"]);

let threadId = createThreadId();
let activeController = null;
let activeUploadController = null;
let isStreaming = false;
let scrollFrame = null;

function createThreadId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }

  return `thread-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function updateThreadLabel() {
  threadLabel.textContent = `Thread ${threadId.slice(0, 8)}`;
  threadLabel.title = threadId;
}

function setPanelOpen(isOpen) {
  panel.classList.toggle("is-open", isOpen);
  panel.setAttribute("aria-hidden", String(!isOpen));
  panelToggle.setAttribute("aria-expanded", String(isOpen));
}

function setStreaming(nextValue) {
  isStreaming = nextValue;
  sendButton.disabled = nextValue;
  sendButton.textContent = nextValue ? "Working" : "Send";
}

function setUploadStreaming(nextValue) {
  attachmentButton.disabled = nextValue;
  attachmentButton.textContent = nextValue ? "Uploading..." : "Attach PDF";
  pdfInput.disabled = nextValue;
}

function scrollConversation() {
  if (scrollFrame !== null) {
    cancelAnimationFrame(scrollFrame);
  }

  scrollFrame = requestAnimationFrame(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
    scrollFrame = null;
  });
}

function showConversation() {
  chatMessages.hidden = false;
  chatView.classList.add("has-messages");
}

function addMessage(role, text = "") {
  const message = document.createElement("article");
  const label = document.createElement("span");
  const content = document.createElement("div");

  message.className = `message message-${role}`;
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "Atlas AI";
  content.className = "message-content";
  content.textContent = text;

  message.append(label, content);
  chatMessages.append(message);
  scrollConversation();

  return content;
}

function renderAssistantMarkdown(element, markdown) {
  if (!markdownRenderer) {
    element.textContent = markdown;
    return;
  }

  element.innerHTML = markdownRenderer.render(markdown);
  element.querySelectorAll("a").forEach((link) => {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });
}

function addStatus(text) {
  const status = document.createElement("div");
  const pulse = document.createElement("span");
  const statusText = document.createElement("span");

  status.className = "agent-status";
  status.setAttribute("role", "status");
  pulse.className = "status-pulse";
  statusText.className = "status-text";
  statusText.textContent = text;
  status.append(pulse, statusText);
  chatMessages.append(status);
  scrollConversation();

  return status;
}

function statusMessage(text) {
  return text.replace(/^\[Atlas AI\]\s*/, "");
}

function setPanelStats({ elements, chunks, docs }) {
  elementsCount.textContent = elements ?? "-";
  chunksCount.textContent = chunks ?? "-";
  docsCount.textContent = docs ?? "-";
}

function resetIngestionPanel() {
  selectedFile.textContent = "No document selected";
  setPanelStats({ elements: "-", chunks: "-", docs: "-" });
  processingLogs.textContent = "Processing logs will appear here.";
}

function appendProcessingLog(text) {
  const nextLine = String(text || "").trim();
  if (!nextLine) {
    return;
  }

  if (processingLogs.textContent === "Processing logs will appear here.") {
    processingLogs.textContent = nextLine;
  } else {
    processingLogs.textContent += `\n${nextLine}`;
  }

  processingLogs.scrollTop = processingLogs.scrollHeight;
}

async function consumeNdjson(response, onEvent) {
  if (!response.body) {
    throw new Error("This browser does not support streamed responses.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let lineEnd = buffer.indexOf("\n");
    while (lineEnd !== -1) {
      const line = buffer.slice(0, lineEnd).trim();
      buffer = buffer.slice(lineEnd + 1);

      if (line) {
        onEvent(JSON.parse(line));
      }

      lineEnd = buffer.indexOf("\n");
    }

    if (done) {
      break;
    }
  }

  if (buffer.trim()) {
    onEvent(JSON.parse(buffer));
  }
}

async function submitMessage(userInput) {
  showConversation();
  addMessage("user", userInput);
  setStreaming(true);

  const requestController = new AbortController();
  activeController = requestController;
  let statusElement = null;
  let assistantContent = null;
  let assistantText = "";
  let markdownFrame = null;

  function flushAssistantRender() {
    if (!assistantContent) {
      return;
    }

    if (markdownFrame !== null) {
      cancelAnimationFrame(markdownFrame);
      markdownFrame = null;
    }

    renderAssistantMarkdown(assistantContent, assistantText);
    scrollConversation();
  }

  function scheduleAssistantRender() {
    if (markdownFrame !== null) {
      return;
    }

    markdownFrame = requestAnimationFrame(() => {
      markdownFrame = null;
      renderAssistantMarkdown(assistantContent, assistantText);
      scrollConversation();
    });
  }

  try {
    const response = await fetch("/invoke?stream_format=ndjson", {
      method: "POST",
      headers: {
        Accept: "application/x-ndjson",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ user_input: userInput, thread_id: threadId }),
      signal: requestController.signal,
    });

    if (!response.ok) {
      throw new Error(`Atlas AI returned ${response.status}.`);
    }

    await consumeNdjson(response, (event) => {
      if (event.type === "status") {
        if (!statusElement) {
          statusElement = addStatus(statusMessage(event.text));
        } else {
          statusElement.querySelector(".status-text").textContent = statusMessage(
            event.text,
          );
        }
        scrollConversation();
        return;
      }

      if (event.type === "token") {
        statusElement?.remove();
        statusElement = null;

        if (!assistantContent) {
          assistantContent = addMessage("assistant");
        }

        assistantText += event.text;
        scheduleAssistantRender();
        return;
      }

      if (event.type === "rag_images") {
        if (!assistantContent) {
          assistantContent = addMessage("assistant");
        }

        assistantText += event.markdown || "";
        scheduleAssistantRender();
      }
    });

    flushAssistantRender();
    statusElement?.remove();
    if (!assistantContent) {
      addMessage("assistant", "No response was returned.");
    }
  } catch (error) {
    statusElement?.remove();

    if (error.name !== "AbortError") {
      addMessage("assistant", `Unable to complete the request: ${error.message}`);
    }
  } finally {
    flushAssistantRender();
    if (activeController === requestController) {
      activeController = null;
      setStreaming(false);
      chatInput.focus();
    }
  }
}

async function readErrorMessage(response) {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") {
      return payload.detail;
    }
  } catch {}

  return `Upload failed with status ${response.status}.`;
}

function isPdfFile(file) {
  return (
    PDF_MIME_TYPES.has(file.type) ||
    file.name.toLowerCase().endsWith(".pdf")
  );
}

async function uploadPdf(file) {
  activeUploadController?.abort();
  const requestController = new AbortController();
  activeUploadController = requestController;

  setPanelOpen(true);
  selectedFile.textContent = `${file.name} (${Math.ceil(file.size / 1024)} KB)`;
  setPanelStats({ elements: "-", chunks: "-", docs: "-" });
  processingLogs.textContent = "Starting upload...";
  setUploadStreaming(true);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/ingest/pdf?stream_format=ndjson", {
      method: "POST",
      headers: {
        Accept: "application/x-ndjson",
      },
      body: formData,
      signal: requestController.signal,
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    await consumeNdjson(response, (event) => {
      if (event.type === "file") {
        selectedFile.textContent = event.file_name || file.name;
        return;
      }

      if (event.type === "stats") {
        setPanelStats({
          elements: event.elements,
          chunks: event.chunks,
          docs: event.docs,
        });
        return;
      }

      if (event.type === "log") {
        appendProcessingLog(event.text);
        return;
      }

      if (event.type === "error") {
        appendProcessingLog(`Error: ${event.text}`);
        return;
      }

      if (event.type === "done") {
        setPanelStats({
          elements: event.elements,
          chunks: event.chunks,
          docs: event.docs,
        });
        appendProcessingLog(event.text || "Ingestion complete");
      }
    });
  } catch (error) {
    if (error.name !== "AbortError") {
      appendProcessingLog(`Error: ${error.message}`);
    }
  } finally {
    if (activeUploadController === requestController) {
      activeUploadController = null;
      setUploadStreaming(false);
      pdfInput.value = "";
    }
  }
}

function startNewThread() {
  activeController?.abort();
  activeController = null;
  setStreaming(false);
  threadId = createThreadId();
  updateThreadLabel();
  chatMessages.replaceChildren();
  chatMessages.hidden = true;
  chatView.classList.remove("has-messages");
  chatInput.value = "";
  chatInput.focus();
}

panelToggle.addEventListener("click", () => {
  const isOpen = !panel.classList.contains("is-open");
  setPanelOpen(isOpen);
});

panelClose.addEventListener("click", () => {
  setPanelOpen(false);
});

attachmentButton.addEventListener("click", () => {
  pdfInput.click();
});

pdfInput.addEventListener("change", () => {
  const [file] = pdfInput.files || [];

  setPanelOpen(true);

  if (!file) {
    resetIngestionPanel();
    return;
  }

  if (!isPdfFile(file)) {
    selectedFile.textContent = file.name;
    setPanelStats({ elements: "-", chunks: "-", docs: "-" });
    processingLogs.textContent = "Only PDF files are supported.";
    pdfInput.value = "";
    return;
  }

  if (file.size > MAX_UPLOAD_SIZE_BYTES) {
    selectedFile.textContent = file.name;
    setPanelStats({ elements: "-", chunks: "-", docs: "-" });
    processingLogs.textContent = "PDF exceeds the 25 MB upload limit.";
    pdfInput.value = "";
    return;
  }

  uploadPdf(file);
});

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const userInput = chatInput.value.trim();

  if (!userInput || isStreaming) {
    return;
  }

  chatInput.value = "";
  submitMessage(userInput);
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

newThreadButton.addEventListener("click", startNewThread);

updateThreadLabel();
resetIngestionPanel();
