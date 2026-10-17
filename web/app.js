const form = document.querySelector("#analysis-form");
const fileInput = document.querySelector("#video-file");
const fileLabel = document.querySelector("#file-label");
const apiKeyInput = document.querySelector("#api-key");
const submitButton = document.querySelector("#submit-button");
const dropzone = document.querySelector("#dropzone");
const statusPanel = document.querySelector("#status-panel");
const statusTitle = document.querySelector("#status-title");
const statusDetail = document.querySelector("#status-detail");
const spinner = document.querySelector("#spinner");
const result = document.querySelector("#result");
const previewPanel = document.querySelector("#preview-panel");
const videoPreview = document.querySelector("#video-preview");
const timelineTrack = document.querySelector("#timeline-track");
const previewNote = document.querySelector("#preview-note");

let previewUrl = null;
let renderedJob = null;

apiKeyInput.value = sessionStorage.getItem("fortebank-api-key") || "";

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileLabel.textContent = file?.name || "Выберите файл или перетащите его сюда";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = file ? URL.createObjectURL(file) : null;
  videoPreview.src = previewUrl || "";
  previewPanel.hidden = !file;
  renderedJob = null;
  result.hidden = true;
  timelineTrack.replaceChildren();
  previewNote.textContent = "После анализа красные метки перенесут видео к найденным моментам.";
});

window.addEventListener("beforeunload", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});

videoPreview.addEventListener("error", () => {
  previewNote.textContent = "Браузер не воспроизводит кодек этого MOV. Кадры и таймкоды анализа останутся доступны ниже.";
});

videoPreview.addEventListener("loadedmetadata", () => renderTimeline(renderedJob));

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, () => dropzone.classList.add("dragging"));
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, () => dropzone.classList.remove("dragging"));
}

function headers() {
  return { "X-API-Key": apiKeyInput.value };
}

function showStatus(kind, title, detail) {
  statusPanel.hidden = false;
  spinner.className = `spinner ${kind}`;
  statusTitle.textContent = title;
  statusDetail.textContent = detail;
}

function seekTo(timestamp, index) {
  if (Number.isFinite(videoPreview.duration)) {
    videoPreview.currentTime = Math.min(timestamp, videoPreview.duration);
    videoPreview.play().catch(() => {});
  }
  document.querySelectorAll(".event, .timeline-marker").forEach((item) => item.classList.remove("selected"));
  document.querySelector(`.event[data-event-index="${index}"]`)?.classList.add("selected");
  document.querySelector(`.timeline-marker[data-event-index="${index}"]`)?.classList.add("selected");
}

function renderTimeline(job) {
  timelineTrack.replaceChildren();
  if (!job || !Number.isFinite(videoPreview.duration) || videoPreview.duration <= 0) return;
  job.violations.forEach((event, index) => {
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = "timeline-marker";
    marker.dataset.eventIndex = index;
    marker.style.left = `${Math.max(0, Math.min(100, event.timecode_sec / videoPreview.duration * 100))}%`;
    marker.title = `Открыть событие на ${event.timecode_sec.toFixed(1)} сек`;
    marker.setAttribute("aria-label", marker.title);
    marker.addEventListener("click", () => seekTo(event.timecode_sec, index));
    timelineTrack.append(marker);
  });
  previewNote.textContent = job.violations.length
    ? "Красные метки — кандидаты на ручную проверку. Нажмите на метку или карточку."
    : "В этой записи модель не нашла кандидатов выше порога.";
}

function renderResult(job) {
  renderedJob = job;
  renderTimeline(job);
  const clear = job.verdict === "no_violation";
  const events = job.violations.map((event, index) => {
    const frames = event.evidence_frames.map((frame) => `
      <button class="evidence-frame" type="button" data-seek="${frame.timestamp_sec}" data-event-index="${index}" aria-label="Перейти к ${frame.timestamp_sec.toFixed(1)} секунде">
        <img src="data:image/jpeg;base64,${frame.image_base64}" alt="Кадр фрагмента на ${frame.timestamp_sec.toFixed(1)} секунде" />
        <span>${frame.timestamp_sec.toFixed(1)} сек</span>
      </button>`).join("");
    return `
      <article class="event" data-event-index="${index}">
        <div class="evidence-gallery">${frames}</div>
        <div class="event-copy">
          <span class="timecode">Кандидат · ${event.timecode_sec.toFixed(1)} сек</span>
          <h3>${event.type}</h3>
          <p>${event.description}</p>
          <dl class="event-meta">
            <div><dt>Модельный score</dt><dd>${event.confidence.toFixed(4)}</dd></div>
            <div><dt>Правило</dt><dd>Не сопоставлено</dd></div>
            <div><dt>Следующий шаг</dt><dd>Просмотреть фрагмент</dd></div>
          </dl>
          <p class="score-note">Score служит для ранжирования и не является вероятностью нарушения.</p>
          <button class="open-event" type="button">Открыть фрагмент в видео</button>
        </div>
      </article>`;
  }).join("");

  result.innerHTML = `
    <section class="panel">
      <div class="verdict">
        <div><p class="step">Результат модели</p><h2>${clear ? "Признаки не обнаружены в этой записи" : "Найдены фрагменты для ручной проверки"}</h2></div>
        <span class="verdict-mark ${clear ? "clear" : "alert"}">${clear ? "Кандидатов нет" : `${job.violations.length} кандидат(а)`}</span>
      </div>
      <p class="result-explanation">${clear
        ? "Это означает, что ни один фрагмент не превысил порог модели. Результат не подтверждает отсутствие всех возможных нарушений."
        : "Метки показывают похожие на обучающие примеры визуальные фрагменты. Решение принимает проверяющий после просмотра контекста."}</p>
      ${events ? `<div class="event-grid">${events}</div>` : ""}
    </section>`;
  result.hidden = false;
  document.querySelectorAll(".event").forEach((card) => {
    const open = () => seekTo(job.violations[Number(card.dataset.eventIndex)].timecode_sec, card.dataset.eventIndex);
    card.addEventListener("click", (event) => {
      if (!event.target.closest(".evidence-frame")) open();
    });
  });
  document.querySelectorAll(".evidence-frame").forEach((frame) => {
    frame.addEventListener("click", () => seekTo(Number(frame.dataset.seek), frame.dataset.eventIndex));
  });
  result.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function poll(jobId) {
  const deadline = Date.now() + 5 * 60 * 1000;
  while (Date.now() < deadline) {
    const response = await fetch(`/api/v1/jobs/${jobId}`, { headers: headers() });
    if (!response.ok) throw new Error((await response.json()).detail || "Не удалось получить статус");
    const job = await response.json();
    if (job.status === "done") return job;
    if (job.status === "failed") throw new Error(job.error || "Анализ завершился ошибкой");
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error("Сервис не завершил анализ за 5 минут");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  sessionStorage.setItem("fortebank-api-key", apiKeyInput.value);
  submitButton.disabled = true;
  result.hidden = true;
  showStatus("", "Видео обрабатывается", "Загружаем файл и запускаем локальный анализ.");

  try {
    const payload = new FormData();
    payload.append("file", file);
    const response = await fetch("/api/v1/analyze", { method: "POST", headers: headers(), body: payload });
    if (!response.ok) throw new Error((await response.json()).detail || "Не удалось загрузить видео");
    const { job_id: jobId } = await response.json();
    const job = await poll(jobId);
    showStatus("done", "Анализ завершён", `Задание ${jobId} обработано.`);
    renderResult(job);
  } catch (error) {
    showStatus("failed", "Не удалось завершить анализ", error.message);
  } finally {
    submitButton.disabled = false;
  }
});
