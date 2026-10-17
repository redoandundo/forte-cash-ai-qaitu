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

apiKeyInput.value = sessionStorage.getItem("fortebank-api-key") || "";

fileInput.addEventListener("change", () => {
  fileLabel.textContent = fileInput.files[0]?.name || "Выберите файл или перетащите его сюда";
});

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

function renderResult(job) {
  const clear = job.verdict === "no_violation";
  const events = job.violations.map((event) => {
    const frame = event.evidence_frames[0];
    return `
      <article class="event">
        <img src="data:image/jpeg;base64,${frame.image_base64}" alt="Кадр-доказательство на ${frame.timestamp_sec.toFixed(1)} секунде" />
        <div class="event-copy">
          <span class="timecode">${event.timecode_sec.toFixed(1)} сек</span>
          <h3>${event.type}</h3>
          <p>${event.description}</p>
          <p class="confidence">Оценка baseline: ${(event.confidence * 100).toFixed(1)}%</p>
        </div>
      </article>`;
  }).join("");

  result.innerHTML = `
    <section class="panel">
      <div class="verdict">
        <div><p class="step">Результат</p><h2>${clear ? "Нарушение не обнаружено" : "Обнаружены признаки возможных нарушений"}</h2></div>
        <span class="verdict-mark ${clear ? "clear" : "alert"}">${clear ? "Проверка пройдена" : `${job.violations.length} событий`}</span>
      </div>
      ${events ? `<div class="event-grid">${events}</div>` : ""}
    </section>`;
  result.hidden = false;
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
