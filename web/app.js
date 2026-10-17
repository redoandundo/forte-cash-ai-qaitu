const MAX_FILE_BYTES = 100 * 1024 * 1024;
const PROCESSING_DELAY_MS = 1800;

const body = document.body;
const uploadView = document.querySelector("#upload-view");
const workspace = document.querySelector("#workspace");
const uploadForm = document.querySelector("#upload-form");
const fileInput = document.querySelector("#video-file");
const dropzone = document.querySelector("#dropzone");
const uploadError = document.querySelector("#upload-error");
const apiKeyInput = document.querySelector("#api-key");
const videoPreview = document.querySelector("#video-preview");
const videoEmpty = document.querySelector("#video-empty");
const previewWarning = document.querySelector("#preview-warning");
const selectedFileName = document.querySelector("#selected-file-name");
const selectedFileMeta = document.querySelector("#selected-file-meta");
const videoTimeDisplay = document.querySelector("#video-time-display");
const startDemoButton = document.querySelector("#start-demo");
const removeFileButton = document.querySelector("#remove-file");
const retryDemoButton = document.querySelector("#retry-demo");
const stateButtons = [...document.querySelectorAll("[data-demo-state]")];
const statusKicker = document.querySelector("#status-kicker");
const statusTitle = document.querySelector("#status-title");
const statusDetail = document.querySelector("#status-detail");
const processingPanel = document.querySelector("#processing-panel");
const violationPanel = document.querySelector("#violation-panel");
const clearPanel = document.querySelector("#clear-panel");
const errorPanel = document.querySelector("#error-panel");
const eventTimecode = document.querySelector("#event-timecode");
const seekEventButton = document.querySelector("#seek-event");
const evidenceCanvas = document.querySelector("#evidence-canvas");
const evidenceImage = document.querySelector("#evidence-image");
const evidencePlaceholder = document.querySelector("#evidence-placeholder");

const statePanels = {
  processing: processingPanel,
  violation: violationPanel,
  no_violation: clearPanel,
  failed: errorPanel,
};

const stateCopy = {
  ready: {
    kicker: "Файл готов",
    title: "Можно начинать проверку",
    detail: "Видео выбрано и доступно для локального предпросмотра.",
  },
  processing: {
    kicker: "Идёт обработка",
    title: "Проверяем запись",
    detail: "Демо показывает будущий ход локального анализа без запуска модели.",
  },
  violation: {
    kicker: "Проверка завершена",
    title: "Есть эпизод для просмотра",
    detail: "Перейдите к таймкоду и проверьте наблюдаемое событие вручную.",
  },
  no_violation: {
    kicker: "Проверка завершена",
    title: "Дополнительных эпизодов нет",
    detail: "Признаки возможного нарушения в этой записи не обнаружены.",
  },
  failed: {
    kicker: "Проверка остановлена",
    title: "Возникла техническая ошибка",
    detail: "Результат не сформирован и не подменён отрицательным verdict.",
  },
};

let selectedFile = null;
let objectUrl = null;
let currentState = "empty";
let processingTimer = null;

apiKeyInput.value = sessionStorage.getItem("fortebank-api-key") || "";
apiKeyInput.addEventListener("change", () => {
  sessionStorage.setItem("fortebank-api-key", apiKeyInput.value);
});

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 Б";
  const units = ["Б", "КБ", "МБ", "ГБ"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / (1024 ** exponent);
  const precision = exponent === 0 || value >= 10 ? 0 : 1;
  return `${value.toFixed(precision)} ${units[exponent]}`;
}

function formatClock(seconds, includeHours = false) {
  const safeSeconds = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const secs = safeSeconds % 60;
  const parts = [minutes.toString().padStart(2, "0"), secs.toString().padStart(2, "0")];
  if (includeHours) parts.unshift(hours.toString().padStart(2, "0"));
  return parts.join(":");
}

function isMovFile(file) {
  return file && (file.name.toLowerCase().endsWith(".mov") || file.type === "video/quicktime");
}

function setUploadError(message = "") {
  uploadError.textContent = message;
  uploadError.hidden = !message;
}

function clearProcessingTimer() {
  if (processingTimer !== null) {
    window.clearTimeout(processingTimer);
    processingTimer = null;
  }
}

function revokePreviewUrl() {
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
}

function hideAllStatePanels() {
  for (const panel of Object.values(statePanels)) {
    panel.hidden = true;
  }
}

function getEventTime() {
  if (!Number.isFinite(videoPreview.duration) || videoPreview.duration <= 0) return 0;
  return Math.max(0, Math.min(videoPreview.duration * 0.42, videoPreview.duration - 0.1));
}

function updateTimecode() {
  const eventSeconds = getEventTime();
  eventTimecode.textContent = formatClock(eventSeconds);
  videoTimeDisplay.textContent = formatClock(videoPreview.currentTime, true);
}

function drawEvidenceFrame() {
  if (videoPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    evidenceImage.hidden = true;
    evidencePlaceholder.hidden = false;
    return;
  }

  const sourceWidth = videoPreview.videoWidth;
  const sourceHeight = videoPreview.videoHeight;
  if (!sourceWidth || !sourceHeight) return;

  const maxWidth = 920;
  const scale = Math.min(1, maxWidth / sourceWidth);
  evidenceCanvas.width = Math.max(1, Math.round(sourceWidth * scale));
  evidenceCanvas.height = Math.max(1, Math.round(sourceHeight * scale));

  const context = evidenceCanvas.getContext("2d");
  if (!context) return;

  try {
    context.drawImage(videoPreview, 0, 0, evidenceCanvas.width, evidenceCanvas.height);
    evidenceImage.src = evidenceCanvas.toDataURL("image/jpeg", 0.82);
    evidenceImage.hidden = false;
    evidencePlaceholder.hidden = true;
  } catch {
    evidenceImage.hidden = true;
    evidencePlaceholder.hidden = false;
  }
}

function prepareEvidence() {
  const targetTime = getEventTime();
  eventTimecode.textContent = formatClock(targetTime);

  if (videoPreview.readyState < HTMLMediaElement.HAVE_METADATA) {
    drawEvidenceFrame();
    return;
  }

  if (Math.abs(videoPreview.currentTime - targetTime) < 0.08) {
    drawEvidenceFrame();
    return;
  }

  videoPreview.pause();
  videoPreview.addEventListener("seeked", drawEvidenceFrame, { once: true });
  videoPreview.currentTime = targetTime;
}

function setDemoState(nextState, { preserveTimer = false } = {}) {
  if (!selectedFile || !stateCopy[nextState]) return;
  if (!preserveTimer) clearProcessingTimer();

  currentState = nextState;
  body.dataset.state = "workspace";
  body.dataset.demoState = nextState;
  hideAllStatePanels();

  for (const button of stateButtons) {
    const isActive = button.dataset.demoState === nextState;
    button.setAttribute("aria-pressed", String(isActive));
  }

  const copy = stateCopy[nextState];
  statusKicker.textContent = copy.kicker;
  statusTitle.textContent = copy.title;
  statusDetail.textContent = copy.detail;

  if (statePanels[nextState]) {
    statePanels[nextState].hidden = false;
  }

  const actionLabel = nextState === "ready"
    ? "Начать проверку"
    : nextState === "processing"
      ? "Проверка выполняется"
      : "Запустить ещё раз";
  startDemoButton.querySelector("span").textContent = actionLabel;
  startDemoButton.disabled = nextState === "processing";

  if (nextState === "violation") {
    prepareEvidence();
  }
}

function startDemoFlow() {
  if (!selectedFile) return;
  clearProcessingTimer();
  setDemoState("processing", { preserveTimer: true });
  processingTimer = window.setTimeout(() => {
    processingTimer = null;
    setDemoState("violation");
  }, PROCESSING_DELAY_MS);
}

function openWorkspace(file) {
  clearProcessingTimer();
  revokePreviewUrl();
  selectedFile = file;
  objectUrl = URL.createObjectURL(file);

  selectedFileName.textContent = file.name;
  selectedFileMeta.textContent = `${formatBytes(file.size)} · MOV · предпросмотр локально`;
  evidenceImage.hidden = true;
  evidenceImage.removeAttribute("src");
  evidencePlaceholder.hidden = false;
  previewWarning.hidden = true;
  videoEmpty.hidden = false;
  videoTimeDisplay.textContent = "00:00:00";

  videoPreview.src = objectUrl;
  videoPreview.load();

  uploadView.hidden = true;
  workspace.hidden = false;
  body.dataset.state = "workspace";
  setDemoState("ready");
  workspace.scrollIntoView({ behavior: "smooth", block: "start" });
}

function acceptFile(file) {
  setUploadError();
  if (!file) return;
  if (!isMovFile(file)) {
    setUploadError("Выберите видео в формате .mov.");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    setUploadError("Файл превышает лимит 100 МБ.");
    return;
  }
  openWorkspace(file);
}

function resetToUpload() {
  clearProcessingTimer();
  videoPreview.pause();
  videoPreview.removeAttribute("src");
  videoPreview.load();
  revokePreviewUrl();
  selectedFile = null;
  currentState = "empty";
  fileInput.value = "";
  workspace.hidden = true;
  uploadView.hidden = false;
  body.dataset.state = "empty";
  delete body.dataset.demoState;
  hideAllStatePanels();
  setUploadError();
  uploadView.scrollIntoView({ behavior: "smooth", block: "start" });
}

uploadForm.addEventListener("submit", (event) => {
  event.preventDefault();
});

fileInput.addEventListener("change", () => {
  acceptFile(fileInput.files?.[0]);
});

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "dragend"]) {
  dropzone.addEventListener(eventName, () => {
    dropzone.classList.remove("dragging");
  });
}

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  acceptFile(event.dataTransfer?.files?.[0]);
});

videoPreview.addEventListener("loadedmetadata", () => {
  videoEmpty.hidden = true;
  selectedFileMeta.textContent = `${formatBytes(selectedFile?.size ?? 0)} · ${formatClock(videoPreview.duration)} · MOV`;
  updateTimecode();
});

videoPreview.addEventListener("loadeddata", () => {
  videoEmpty.hidden = true;
  previewWarning.hidden = true;
  if (currentState === "violation") prepareEvidence();
});

videoPreview.addEventListener("timeupdate", updateTimecode);

videoPreview.addEventListener("error", () => {
  videoEmpty.hidden = false;
  previewWarning.hidden = false;
});

startDemoButton.addEventListener("click", startDemoFlow);
retryDemoButton.addEventListener("click", startDemoFlow);
removeFileButton.addEventListener("click", resetToUpload);

for (const button of stateButtons) {
  button.addEventListener("click", () => {
    setDemoState(button.dataset.demoState);
  });
}

seekEventButton.addEventListener("click", () => {
  if (videoPreview.readyState < HTMLMediaElement.HAVE_METADATA) return;
  videoPreview.currentTime = getEventTime();
  videoPreview.play().catch(() => {});
});

window.addEventListener("beforeunload", revokePreviewUrl);
