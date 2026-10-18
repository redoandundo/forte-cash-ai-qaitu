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
const chooseFileButton = document.querySelector("#choose-file");
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
    detail: "Система автоматически сформировала вывод, основания и оценку риска для каждого эпизода.",
  },
  no_violation: {
    kicker: "Проверка завершена",
    title: "Дополнительных эпизодов нет",
    detail: "Анализ завершён без эпизодов, требующих дополнительной проверки.",
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
let activeEventIndex = -1;
let currentJobId = null;
let currentRuleCatalog = null;
let runtimeIdentity = { badge: "Local", caption: "проверяем runtime" };

// Remove credentials saved by older versions; the website now uses an HttpOnly session.
try { sessionStorage.removeItem("fortebank-api-key"); } catch { }

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 Б";
  const units = ["Б", "КБ", "МБ", "ГБ"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / (1024 ** exponent);
  const precision = exponent === 0 || value >= 10 ? 0 : 1;
  return `${value.toFixed(precision)} ${units[exponent]}`;
}

function formatClock(seconds, includeHours = false, fractionDigits = 0) {
  const safeSeconds = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const factor = 10 ** fractionDigits;
  const roundedSeconds = Math.round(safeSeconds * factor) / factor;
  const wholeSeconds = Math.floor(roundedSeconds);
  const hours = Math.floor(wholeSeconds / 3600);
  const minutes = Math.floor((wholeSeconds % 3600) / 60);
  const secs = wholeSeconds % 60;
  const fraction = fractionDigits > 0
    ? `.${Math.round((roundedSeconds - wholeSeconds) * factor).toString().padStart(fractionDigits, "0")}`
    : "";
  const parts = [minutes.toString().padStart(2, "0"), `${secs.toString().padStart(2, "0")}${fraction}`];
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
  const normalized = label.toLowerCase();
  runtimeBadge.textContent = normalized.includes("demo")
    ? "Demo model"
    : normalized.includes("onnx")
      ? "ONNX"
      : "Local";
  jobCaption.textContent = label;
}

function applyRuntimeIdentity() {
  runtimeBadge.textContent = runtimeIdentity.badge;
  jobCaption.textContent = runtimeIdentity.caption;
}

async function refreshRuntimeIdentity() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    const health = await responseJson(response);
    if (!response.ok || health.status !== "ok" || !health.model_ready) {
      runtimeIdentity = { badge: "Offline", caption: "модель не готова" };
    } else {
      const providers = Array.isArray(health.runtime_providers)
        ? health.runtime_providers.join(", ")
        : "runtime provider неизвестен";
      runtimeIdentity = {
        badge: health.inference_mode === "onnx"
          ? (health.semantic_mode === "qwen" ? "ONNX + Qwen" : "ONNX") : "Demo model",
        caption: `${health.model_version || health.runtime_mode} · ${providers}`
          + (health.semantic_mode === "qwen" ? ` · ${health.semantic_model_version || "Qwen"}` : ""),
      };
    }
  } catch {
    runtimeIdentity = { badge: "Offline", caption: "локальный API недоступен" };
  }
  applyRuntimeIdentity();
}

function seekVideo(seconds) {
  if (videoPreview.readyState < HTMLMediaElement.HAVE_METADATA) return;
  const target = Math.max(0, Math.min(Number(seconds) || 0, videoPreview.duration || 0));
  videoPreview.currentTime = target;
  videoPreview.play().catch(() => {});
}

function selectEvent(index, seconds, scroll = false) {
  activeEventIndex = index;
  [...eventList.children].forEach((card, cardIndex) => {
    card.classList.toggle("selected", cardIndex === index);
  });
  [...eventMarkers.children].forEach((marker, markerIndex) => {
    marker.setAttribute("aria-pressed", String(markerIndex === index));
  });
  if (scroll) eventList.children[index]?.scrollIntoView({ block: "nearest" });
  seekVideo(seconds);
}

function makeEvidenceGallery(frames, eventIndex, peakTime) {
  const gallery = document.createElement("div");
  gallery.className = "evidence-gallery";
  [...frames].sort((a, b) => a.timestamp_sec - b.timestamp_sec).forEach((frame, frameIndex) => {
    const offset = frame.timestamp_sec - peakTime;
    const role = Math.abs(offset) <= 0.05 ? "У пика" : offset < 0 ? "До пика" : "После пика";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "evidence-shot";
    button.setAttribute("aria-label", `Перейти к кадру ${frameIndex + 1} эпизода ${eventIndex + 1}`);
    const image = document.createElement("img");
    image.src = `data:image/jpeg;base64,${frame.image_base64}`;
    image.alt = `JPEG-доказательство ${frameIndex + 1}, таймкод ${formatClock(frame.timestamp_sec, false, 1)}`;
    const label = document.createElement("span");
    label.className = "evidence-label";
    label.textContent = `${role} · ${formatClock(frame.timestamp_sec, false, 1)}`;
    button.append(image, label);
    button.addEventListener("click", () => selectEvent(eventIndex, frame.timestamp_sec));
    gallery.append(button);
  });
  return gallery;
}

function makeChainSection(labelText, content, { list = false, className = "" } = {}) {
  const section = document.createElement("section");
  section.className = `chain-section ${className}`.trim();
  const label = document.createElement("p");
  label.className = "chain-label";
  label.textContent = labelText;
  section.append(label);
  if (list) {
    const items = document.createElement("ol");
    items.className = "chain-list";
    for (const item of content) {
      const listItem = document.createElement("li");
      listItem.textContent = item;
      items.append(listItem);
    }
    section.append(items);
  } else {
    const paragraph = document.createElement("p");
    paragraph.className = "chain-text";
    paragraph.textContent = content;
    section.append(paragraph);
  }
  return section;
}

function makeEvidenceChain(detail, violation) {
  const fallback = {
    thesis: `Вероятно, обнаружено событие: ${violation.type}.`,
    observations: [
      `Модельный сигнал выше порога локализован на ${formatClock(violation.timecode_sec, false, 1)}.`,
      `Система сформировала JPEG-кадров: ${(violation.evidence_frames || []).length}.`,
    ],
    reasoning: "Временная локализация и согласованные кадры подкрепляют автоматический вывод о подозрительном эпизоде.",
    risk: "Нетипичная активность в кассовой зоне может повышать операционный риск и затруднять прослеживаемость действий.",
    rule: {
      status: "not_established",
      label: "Нормативное основание не установлено",
      applicability: "Правило появится только после сопоставления с проверенным локальным каталогом.",
    },
  };
  const chain = detail || fallback;
  const wrapper = document.createElement("div");
  wrapper.className = "evidence-chain";
  wrapper.append(
    makeChainSection("1 · Тезис — что происходит", chain.thesis, { className: "chain-thesis" }),
    makeChainSection("2 · Какие признаки это подкрепляют", chain.observations || [], { list: true }),
    makeChainSection("3 · Логика вывода", chain.reasoning),
    makeChainSection("4 · Почему это важно", chain.risk, { className: "chain-risk" }),
  );

  const rules = Array.isArray(chain.rules) && chain.rules.length
    ? chain.rules : [chain.rule || fallback.rule];
  for (const [index, rule] of rules.entries()) {
    const ruleSection = makeChainSection(
      rules.length > 1 ? `5.${index + 1} · Предполагаемое правило` : "5 · Предполагаемое правило",
      rule.status === "matched"
        ? `${rule.label}. ${rule.document_name || ""} ${rule.section || ""}. ${rule.applicability || ""}`.trim()
        : `${rule.label}. ${rule.applicability || ""}`.trim(),
      { className: rule.status === "matched" ? "chain-rule matched" : "chain-rule unmatched" },
    );
    if (rule.status === "matched" && /^https?:\/\//.test(rule.source_url || "")) {
      const link = document.createElement("a");
      link.href = rule.source_url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = "Открыть нормативный источник";
      ruleSection.append(link);
    }
    wrapper.append(ruleSection);
  }
  return wrapper;
}

function makeRuleReview(article, violation, detail, eventIndex) {
  const panel = document.createElement("details");
  panel.className = "rule-review";
  const title = document.createElement("summary");
  title.textContent = "Сопоставить правило после просмотра";
  const note = document.createElement("p");
  note.textContent = "Отметьте только наблюдения, которые вы проверили по видео. Это ручная проверка, а не решение модели.";
  const select = document.createElement("select");
  select.setAttribute("aria-label", `Правило для эпизода ${eventIndex}`);
  const empty = document.createElement("option");
  empty.value = ""; empty.textContent = "Выберите правило"; select.append(empty);
  for (const rule of currentRuleCatalog.rules) {
    const option = document.createElement("option");
    option.value = rule.id; option.textContent = `${rule.label} · ${rule.section}`;
    select.append(option);
  }
  const facts = document.createElement("div");
  const limitation = document.createElement("p");
  const save = document.createElement("button");
  save.type = "button"; save.className = "secondary-button";
  save.textContent = "Подтвердить наблюдения и сопоставить"; save.disabled = true;
  const reset = document.createElement("button");
  reset.type = "button"; reset.className = "text-button"; reset.textContent = "Отменить ручное сопоставление";
  const message = document.createElement("p");
  message.setAttribute("role", "status");
  let boxes = [];
  select.addEventListener("change", () => {
    facts.replaceChildren(); boxes = []; save.disabled = true;
    const rule = currentRuleCatalog.rules.find(item => item.id === select.value);
    limitation.textContent = rule?.limitation || "";
    message.textContent = "Изменения применятся после подтверждения.";
    for (const fact of rule?.facts || []) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox"; input.value = fact.id;
      input.addEventListener("change", () => { save.disabled = !boxes.every(box => box.checked); });
      const text = document.createElement("span"); text.textContent = fact.label;
      boxes.push(input); label.append(input, text); facts.append(label);
    }
  });
  const jobId = currentJobId;
  async function submit(resetReview) {
    save.disabled = true; reset.disabled = true; select.disabled = true;
    boxes.forEach(box => { box.disabled = true; });
    try {
      const response = await fetch(`/web/api/v1/jobs/${encodeURIComponent(jobId)}/events/${eventIndex}/rule-review`, {
        method: "PUT", credentials: "same-origin", headers: {"X-Qaitu-Client": "web", "Content-Type": "application/json"},
        body: JSON.stringify({rule_id: resetReview ? null : select.value,
          confirmed_facts: resetReview ? [] : boxes.filter(box => box.checked).map(box => box.value)}),
      });
      const payload = await responseJson(response);
      if (!response.ok) throw new Error(errorMessage(response.status, payload));
      if (currentJobId !== jobId) return;
      article.querySelector(".evidence-chain").replaceWith(makeEvidenceChain(payload, violation));
      message.textContent = resetReview ? "Восстановлен исходный результат автоматического анализа." : "Сохранено: правило сопоставлено проверяющим. Автоматический verdict не изменён.";
      if (resetReview) { select.value = ""; facts.replaceChildren(); boxes = []; limitation.textContent = ""; }
    } catch (error) {
      message.textContent = error.message || "Не удалось сохранить сопоставление.";
    } finally {
      reset.disabled = false; select.disabled = false;
      boxes.forEach(box => { box.disabled = false; });
      save.disabled = !boxes.length || !boxes.every(box => box.checked);
    }
  }
  save.addEventListener("click", () => submit(false));
  reset.addEventListener("click", () => submit(true));
  panel.append(title, note, select, facts, limitation, save, reset, message);
  return panel;
}

function renderMarkers(violations) {
  eventMarkers.replaceChildren();
  if (!Number.isFinite(videoPreview.duration) || videoPreview.duration <= 0) return;
  violations.forEach((violation, index) => {
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = "event-marker";
    marker.style.left = `${Math.max(1, Math.min(99, (violation.timecode_sec / videoPreview.duration) * 100))}%`;
    marker.title = `Эпизод ${index + 1}: ${formatClock(violation.timecode_sec, false, 1)}`;
    marker.setAttribute("aria-label", marker.title);
    marker.setAttribute("aria-controls", `event-${index + 1}`);
    marker.setAttribute("aria-pressed", String(index === activeEventIndex));
    marker.addEventListener("click", () => selectEvent(index, violation.timecode_sec, true));
    eventMarkers.append(marker);
  });
}

function renderViolationResult(job, progress) {
  currentViolations = job.violations || [];
  activeEventIndex = -1;
  const count = currentViolations.length;
  resultCount.textContent = `${count} ${episodeWord(count)}`;
  violationSummary.textContent = progress.summary || "Система автоматически сформировала объяснение для отмеченных эпизодов.";
  eventList.replaceChildren();

  currentViolations.forEach((violation, index) => {
    const article = document.createElement("article");
    article.className = "event-card";
    article.id = `event-${index + 1}`;
    article.setAttribute("aria-label", `Эпизод ${index + 1}`);
    article.append(makeEvidenceGallery(violation.evidence_frames || [], index, violation.timecode_sec));

    const copy = document.createElement("div");
    copy.className = "event-copy";
    const timeButton = document.createElement("button");
    timeButton.type = "button";
    timeButton.className = "timecode-button";
    timeButton.textContent = `▶  ${formatClock(violation.timecode_sec, false, 1)}`;
    timeButton.addEventListener("click", () => selectEvent(index, violation.timecode_sec));
    const heading = document.createElement("h3");
    heading.textContent = `Эпизод ${index + 1} · ${violation.type}`;
    const frameCount = document.createElement("p");
    frameCount.className = "evidence-count";
    frameCount.textContent = `Кадры одного эпизода: ${(violation.evidence_frames || []).length}. Пик — отметка модели.`;
    const description = document.createElement("p");
    description.textContent = violation.description;
    const detail = (progress.analysis_details || []).find((item) => item.event_index === index + 1);
    description.hidden = Boolean(detail);
    copy.append(timeButton, heading, frameCount, description, makeEvidenceChain(detail, violation));
    if (currentJobId && currentRuleCatalog) copy.append(makeRuleReview(article, violation, detail, index + 1));
    article.append(copy);
    eventList.append(article);
  });

  renderMarkers(currentViolations);
  setUiState("violation");
}

function renderCompleted(job, progress) {
  if (!Array.isArray(job.violations)
      || !["violation", "no_violation"].includes(job.verdict)
      || (job.verdict === "violation") !== (job.violations.length > 0)) {
    throw new Error("Сервис вернул несогласованный результат. Повторите проверку.");
  }
  updateProgress({ ...progress, status: "done", stage: "summary", progress_percent: 100 });
  if (job.verdict === "violation" && job.violations?.length) {
    renderViolationResult(job, progress);
    return;
  }
  currentViolations = [];
  activeEventIndex = -1;
  eventList.replaceChildren();
  eventMarkers.replaceChildren();
  clearSummary.textContent = progress.summary || "Автоматический анализ не обнаружил в этой записи признаков возможного нарушения.";
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
  if (status === 401) return "Сессия истекла. Запустите проверку ещё раз.";
  if (status === 429) return "Сервис занят. Дождитесь завершения текущей проверки и повторите попытку.";
  if (status === 413) return "Файл превышает лимит 100 МБ.";
  if (status === 415) return "Сервис принимает только видео в формате MOV.";
  return typeof payload.detail === "string" ? payload.detail : "Локальный сервис не вернул результат.";
}

async function getJson(url, signal) {
  const response = await fetch(url, { credentials: "same-origin", headers: { "X-Qaitu-Client": "web" }, signal });
  const payload = await responseJson(response);
  if (!response.ok) {
    const error = new Error(errorMessage(response.status, payload));
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function pollJob(jobId, signal) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < JOB_TIMEOUT_MS) {
    const [job, progress] = await Promise.all([
      getJson(`/web/api/v1/jobs/${encodeURIComponent(jobId)}`, signal),
      getJson(`/web/api/v1/jobs/${encodeURIComponent(jobId)}/progress`, signal),
    ]);
    signal.throwIfAborted();
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

  cancelActiveRequest();
  currentJobId = null;
  activeController = new AbortController();
  const { signal } = activeController;
  setUploadError();
  currentViolations = [];
  activeEventIndex = -1;
  eventList.replaceChildren();
  eventMarkers.replaceChildren();
  jobCaption.textContent = "создаём задачу";
  runtimeBadge.textContent = "Local";
  updateProgress({ status: "processing", stage: "upload_received", progress_percent: 5, runtime_mode: "Preparing runtime" });
  setUiState("processing");

  try {
    const sessionResponse = await fetch("/web/api/v1/session", {
      method: "POST", credentials: "same-origin", headers: {"X-Qaitu-Client": "web"}, signal,
    });
    const session = await responseJson(sessionResponse);
    if (!sessionResponse.ok || !session.ready) throw new Error("Не удалось подключиться к сервису. Повторите попытку.");
    signal.throwIfAborted();
    currentRuleCatalog = await getJson("/web/api/v1/rules", signal);
    const formData = new FormData();
    formData.append("file", selectedFile, selectedFile.name);
    const response = await fetch("/web/api/v1/analyze", {
      method: "POST",
      credentials: "same-origin",
      headers: { "X-Qaitu-Client": "web" },
      body: formData,
      signal,
    });
    const payload = await responseJson(response);
    if (!response.ok) throw new Error(errorMessage(response.status, payload));
    signal.throwIfAborted();
    currentJobId = payload.job_id;
    jobCaption.textContent = `job ${payload.job_id.slice(0, 8)}`;
    await pollJob(payload.job_id, signal);
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
  currentJobId = null;
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
  applyRuntimeIdentity();
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
  currentJobId = null;
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
chooseFileButton.addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});

dropzone.addEventListener("click", (event) => {
  if (event.target.closest("button, input, details, summary, a")) return;
  fileInput.click();
});

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

refreshRuntimeIdentity();
