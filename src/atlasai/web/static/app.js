const panel = document.getElementById("rag-panel");
const panelToggle = document.getElementById("panel-toggle");
const panelClose = document.getElementById("panel-close");
const attachmentButton = document.getElementById("attachment-button");
const pdfInput = document.getElementById("pdf-input");
const selectedFile = document.getElementById("selected-file");
const processingLogs = document.getElementById("processing-logs");
const chatInput = document.getElementById("chat-input");

function setPanelOpen(isOpen) {
  panel.classList.toggle("is-open", isOpen);
  panel.setAttribute("aria-hidden", String(!isOpen));
  panelToggle.setAttribute("aria-expanded", String(isOpen));
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
    selectedFile.textContent = "No document selected";
    processingLogs.textContent = "Processing logs will appear here.";
    return;
  }

  selectedFile.textContent = file.name;
  processingLogs.textContent =
    "Upload wiring is intentionally deferred.\nThis panel is ready for future PDF ingestion events.";
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
  }
});
