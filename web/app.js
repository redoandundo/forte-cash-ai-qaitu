const MAX_FILE_BYTES = 100 * 1024 * 1024;
const POLL_INTERVAL_MS = 500;
const JOB_TIMEOUT_MS = 5 * 60 * 1000;

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
const startButton = document.querySelector("#start-analysis");
const removeFileButton = document.querySelector("#remove-file");
const retryButton = document.querySelector("#retry-analysis");
const statusKicker = document.querySelector("#status-kicker");
const statusTitle = document.querySelector("#status-title");
const statusDetail = document.querySelector("#status-detail");
const processingPanel = document.querySelector("#processing-panel");
const violationPanel = document.querySelector("#violation-panel");
const clearPanel = document.querySelector("#clear-panel");
const errorPanel = document.querySelector("#error-panel");
const progressValue = document.querySelector("#progress-value");
const progressBar = document.querySelector("#progress-bar");
const stageItems = [...document.querySelectorAll("#stage-list [data-stage]")];
const runtimeBadge = document.querySelector("#runtime-badge");
const jobCaption = document.querySelector("#job-caption");
const resultCount = document.querySelector("#result-count");
const violationSummary = document.querySelector("#violation-summary");
const clearSummary = document.querySelector("#clear-summary");
const errorDetail = document.querySelector("#error-detail");
const eventList = document.querySelector("#event-list");
const eventMarkers = document.querySelector("#event-markers");

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
    detail: "Видео выбрано. Сервис выполнит декодирование, нарезку окон, inference и подготовку кадров.",
  },
  processing: {
    kicker: "Идёт локальная обработка",
    title: "Проверяем запись",
    detail: "Этапы ниже приходят из реального inference pipeline.",
  },
  violation: {
    kicker: "Проверка завершена",
    title: "Есть эпизоды для просмотра",
    detail: "Нажмите таймкод или отметку на видео и проверьте событие вручную.",
  },
  no_violation: {
    kicker: "Проверка завершена",
    title: "Дополнительных эпизодов нет",
    detail: "Локальная модель не отметила интервалы выше заданного порога.",
  },
  failed: {
    kicker: "Проверка остановлена",
    title: "Возникла техническая ошибка",
    detail: "Результат не сформирован и не подменён отрицательным verdict.",
  },
};

const stageOrder = [
  "upload_received",
  "video_decode",
  "temporal_windows",
  "model_inference",
  "aggregation",
  "evidence",
  "summary",
];

let selectedFile = null;
let objectUrl = null;
let activeController = null;
let currentViolations = [];

apiKeyInput.value = sessionStorage.getItem("fortebank-api-key") || "demo-secret";
apiKeyInput.addEventListener("change", () => {
  sessionStorage.setItem("fortebank-api-key", apiKeyInput.value.trim());
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

function episodeWord(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return "эпизод";
  if ([2, 3, 4].includes(mod10) && ![12, 13, 14].includes(mod100)) return "эпизода";
  return "эпизодов";
}

function isMovFile(file) {
  return file && (file.name.toLowerCase().endsWith(".mov") || file.type === "video/quicktime");
}

function setUploadError(message = "") {
  uploadError.textContent = message;
  uploadError.hidden = !message;
}

function revokePreviewUrl() {
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
}

function cancelActiveRequest() {
  if (activeController) {
    activeController.abort();
    activeController = null;
  }
}

function hideAllStatePanels() {
  for (const panel of Object.values(statePanels)) panel.hidden = true;
}

function setUiState(nextState) {
  const copy = stateCopy[nextState];
  if (!copy) return;
  body.dataset.state = "workspace";
  body.dataset.demoState = nextState;
  hideAllStatePanels();
  statusKicker.textContent = copy.kicker;
  statusTitle.textContent = copy.title;
  statusDetail.textContent = copy.detail;
  if (statePanels[nextState]) statePanels[nextState].hidden = false;

  startButton.disabled = nextState === "processing";
  startButton.querySelector("span").textContent = nextState === "processing"
    ? "Проверка выполняется"
    : nextState === "ready"
      ? "Начать проверку"
      : "Запустить ещё раз";
}

function updateProgress(progress) {
  const percent = Math.max(0, Math.min(100, Number(progress.progress_percent) || 0));
  progressValue.textContent = `${percent}%`;
  progressBar.style.width = `${percent}%`;

  const currentIndex = stageOrder.indexOf(progress.stage);
  for (const item of stageItems) {
    const itemIndex = stageOrder.indexOf(item.dataset.stage);
    item.classList.toggle("complete", progress.status === "done" || itemIndex < currentIndex);
    item.classList.toggle("active", progress.status === "processing" && itemIndex === currentIndex);
  }

  const label = progress.runtime_mode || "Локальный runtime";
  runtimeBadge.textContent = label.toLowerCase().includes("demo") ? "Demo model" : "ONNX";
  jobCaption.textContent = label;
}

function seekVideo(seconds) {
  if (videoPreview.readyState < HTMLMediaElement.HAVE_METADATA) return;
  const target = Math.max(0, Math.min(Number(seconds) || 0, videoPreview.duration || 0));
  videoPreview.currentTime = target;
  videoPreview.play().catch(() => {});
}

function makeEvidenceGallery(frames, eventIndex) {
  const gallery = document.createElement("div");
  gallery.className = "evidence-gallery";
  frames.forEach((frame, frameIndex) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "evidence-shot";
    button.setAttribute("aria-label", `Перейти к кадру ${frameIndex + 1} эпизода ${eventIndex + 1}`);
    const image = document.createElement("img");
    image.src = `data:image/jpeg;base64,${frame.image_base64}`;
    image.alt = `JPEG-доказательство ${frameIndex + 1}, таймкод ${formatClock(frame.timestamp_sec)}`;
    const label = document.createElement("span");
    label.className = "evidence-label";
    label.textContent = `JPEG ${String(frameIndex + 1).padStart(2, "0")} · ${formatClock(frame.timestamp_sec)}`;
    button.append(image, label);
    button.addEventListener("click", () => seekVideo(frame.timestamp_sec));
    gallery.append(button);
  });
  return gallery;
}

function renderMarkers(violations) {
  eventMarkers.replaceChildren();
  if (!Number.isFinite(videoPreview.duration) || videoPreview.duration <= 0) return;
  violations.forEach((violation, index) => {
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = "event-marker";
    marker.style.left = `${Math.max(1, Math.min(99, (violation.timecode_sec / videoPreview.duration) * 100))}%`;
    marker.title = `Эпизод ${index + 1}: ${formatClock(violation.timecode_sec)}`;
    marker.setAttribute("aria-label", marker.title);
    marker.addEventListener("click", () => seekVideo(violation.timecode_sec));
    eventMarkers.append(marker);
  });
}

function renderViolationResult(job, progress) {
  currentViolations = job.violations || [];
  const count = currentViolations.length;
  resultCount.textContent = `${count} ${episodeWord(count)}`;
  violationSummary.textContent = progress.summary || "Отмеченные эпизоды требуют ручной проверки по видео и кадрам.";
  eventList.replaceChildren();

  currentViolations.forEach((violation, index) => {
    const article = document.createElement("article");
    article.className = "event-card";
    article.append(makeEvidenceGallery(violation.evidence_frames || [], index));

    const copy = document.createElement("div");
    copy.className = "event-copy";
    const timeButton = document.createElement("button");
    timeButton.type = "button";
    timeButton.className = "timecode-button";
    timeButton.textContent = `▶  ${formatClock(violation.timecode_sec)}`;
    timeButton.addEventListener("click", () => seekVideo(violation.timecode_sec));
    const heading = document.createElement("h3");
    heading.textContent = violation.type;
    const description = document.createElement("p");
    description.textContent = violation.description;
    const review = document.createElement("span");
    review.className = "review-label";
    review.textContent = "Требует проверки сотрудником";
    copy.append(timeButton, heading, description, review);
    article.append(copy);
    eventList.append(article);
  });

  renderMarkers(currentViolations);
  setUiState("violation");
}

function renderCompleted(job, progress) {
  updateProgress({ ...progress, status: "done", stage: "summary", progress_percent: 100 });
  if (job.verdict === "violation" && job.violations?.length) {
    renderViolationResult(job, progress);
    return;
  }
  currentViolations = [];
  eventMarkers.replaceChildren();
  clearSummary.textContent = progress.summary || "В этой записи не найдено эпизодов для дополнительной ручной проверки.";
  setUiState("no_violation");
}

async function responseJson(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

function errorMessage(status, payload) {
  if (status === 401) return "API-ключ не принят. Проверьте ключ в технических настройках.";
  if (status === 413) return "Файл превышает лимит 100 МБ.";
  if (status === 415) return "Сервис принимает только видео в формате MOV.";
  return typeof payload.detail === "string" ? payload.detail : "Локальный сервис не вернул результат.";
}

async function getJson(url, apiKey, signal) {
  const response = await fetch(url, { headers: { "X-API-Key": apiKey }, signal });
  const payload = await responseJson(response);
  if (!response.ok) throw new Error(errorMessage(response.status, payload));
  return payload;
}

async function pollJob(jobId, apiKey, signal) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < JOB_TIMEOUT_MS) {
    const [job, progress] = await Promise.all([
      getJson(`/api/v1/jobs/${encodeURIComponent(jobId)}`, apiKey, signal),
      getJson(`/internal/api/v1/jobs/${encodeURIComponent(jobId)}/progress`, apiKey, signal),
    ]);
    updateProgress(progress);
    if (job.status === "done") {
      renderCompleted(job, progress);
      return;
    }
    if (job.status === "failed") throw new Error(job.error || progress.error || "Обработка завершилась технической ошибкой.");
    await new Promise((resolve) => window.setTimeout(resolve, POLL_INTERVAL_MS));
  }
  throw new Error("Превышен лимит обработки 5 минут. Результат не сформирован.");
}

async function startAnalysis() {
  if (!selectedFile) return;
  const apiKey = apiKeyInput.value.trim();
  if (!apiKey) {
    errorDetail.textContent = "Укажите API-ключ в технических настройках.";
    setUiState("failed");
    return;
  }

  cancelActiveRequest();
  activeController = new AbortController();
  const { signal } = activeController;
  setUploadError();
  eventList.replaceChildren();
  eventMarkers.replaceChildren();
  jobCaption.textContent = "создаём задачу";
  runtimeBadge.textContent = "Local";
  updateProgress({ status: "processing", stage: "upload_received", progress_percent: 5, runtime_mode: "Preparing runtime" });
  setUiState("processing");

  try {
    sessionStorage.setItem("fortebank-api-key", apiKey);
    const formData = new FormData();
    formData.append("file", selectedFile, selectedFile.name);
    const response = await fetch("/api/v1/analyze", {
      method: "POST",
      headers: { "X-API-Key": apiKey },
      body: formData,
      signal,
    });
    const payload = await responseJson(response);
    if (!response.ok) throw new Error(errorMessage(response.status, payload));
    jobCaption.textContent = `job ${payload.job_id.slice(0, 8)}`;
    await pollJob(payload.job_id, apiKey, signal);
  } catch (error) {
    if (error.name === "AbortError") return;
    errorDetail.textContent = error instanceof Error ? error.message : "Не удалось обработать файл.";
    setUiState("failed");
  } finally {
    if (activeController?.signal === signal) activeController = null;
  }
}

function openWorkspace(file) {
  cancelActiveRequest();
  revokePreviewUrl();
  selectedFile = file;
  objectUrl = URL.createObjectURL(file);
  currentViolations = [];
  selectedFileName.textContent = file.name;
  selectedFileMeta.textContent = `${formatBytes(file.size)} · MOV · локальный предпросмотр`;
  previewWarning.hidden = true;
  videoEmpty.hidden = false;
  videoTimeDisplay.textContent = "00:00:00";
  eventList.replaceChildren();
  eventMarkers.replaceChildren();
  jobCaption.textContent = "ожидает запуска";
  runtimeBadge.textContent = "Local";
  videoPreview.src = objectUrl;
  videoPreview.load();
  uploadView.hidden = true;
  workspace.hidden = false;
  setUiState("ready");
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
  cancelActiveRequest();
  videoPreview.pause();
  videoPreview.removeAttribute("src");
  videoPreview.load();
  revokePreviewUrl();
  selectedFile = null;
  currentViolations = [];
  fileInput.value = "";
  workspace.hidden = true;
  uploadView.hidden = false;
  body.dataset.state = "empty";
  delete body.dataset.demoState;
  hideAllStatePanels();
  setUploadError();
  uploadView.scrollIntoView({ behavior: "smooth", block: "start" });
}

uploadForm.addEventListener("submit", (event) => event.preventDefault());
fileInput.addEventListener("change", () => acceptFile(fileInput.files?.[0]));

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "dragend"]) {
  dropzone.addEventListener(eventName, () => dropzone.classList.remove("dragging"));
}

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  acceptFile(event.dataTransfer?.files?.[0]);
});

videoPreview.addEventListener("loadedmetadata", () => {
  videoEmpty.hidden = true;
  selectedFileMeta.textContent = `${formatBytes(selectedFile?.size ?? 0)} · ${formatClock(videoPreview.duration)} · MOV`;
  renderMarkers(currentViolations);
});

videoPreview.addEventListener("loadeddata", () => {
  videoEmpty.hidden = true;
  previewWarning.hidden = true;
});

videoPreview.addEventListener("timeupdate", () => {
  videoTimeDisplay.textContent = formatClock(videoPreview.currentTime, true);
});

videoPreview.addEventListener("error", () => {
  videoEmpty.hidden = false;
  previewWarning.hidden = false;
});

startButton.addEventListener("click", startAnalysis);
retryButton.addEventListener("click", startAnalysis);
removeFileButton.addEventListener("click", resetToUpload);
window.addEventListener("beforeunload", () => {
  cancelActiveRequest();
  revokePreviewUrl();
});
