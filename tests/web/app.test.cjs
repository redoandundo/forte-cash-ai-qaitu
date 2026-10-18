const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor() {
    this.children = []; this.dataset = {}; this.style = {}; this.attributes = {};
    this.handlers = {}; this.textContent = ''; this.value = ''; this.hidden = false;
    this.classes = new Set();
    this.classList = {
      toggle: (name, enabled) => enabled ? this.classes.add(name) : this.classes.delete(name),
      add: name => this.classes.add(name), remove: name => this.classes.delete(name),
    };
  }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, handler) { this.handlers[name] = handler; }
  querySelector() { return this.child ||= new Element(); }
  scrollIntoView() { this.scrolled = true; }
  play() { return Promise.resolve(); }
}

function setup() {
  const elements = new Map();
  const document = {
    body: new Element(), createElement: () => new Element(), querySelectorAll: () => [],
    querySelector: selector => {
      if (!elements.has(selector)) elements.set(selector, new Element());
      return elements.get(selector);
    },
  };
  const context = vm.createContext({
    document, sessionStorage: { getItem: () => null, setItem() {} },
    window: { addEventListener() {}, setTimeout }, HTMLMediaElement: { HAVE_METADATA: 1 },
    fetch: async () => ({ ok: true, json: async () => ({status: 'ok', model_ready: false}) }),
    URL, AbortController, setTimeout,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../../web/app.js'), 'utf8'), context);
  const video = elements.get('#video-preview');
  video.duration = 60; video.readyState = 1;
  return { context, elements, video, run: script => vm.runInContext(script, context) };
}

function event(time, frames = [time - 1, time, time + 1]) {
  return { type: 'Тестовый эпизод', timecode_sec: time, description: 'Synthetic fixture',
    evidence_frames: frames.map(timestamp_sec => ({timestamp_sec, image_base64: 'fixture'})) };
}

test('runtime badge shows Qwen only when the API confirms it is enabled', async () => {
  const s = setup();
  s.context.fetch = async () => ({ok: true, json: async () => ({
    status: 'ok', model_ready: true, inference_mode: 'onnx', model_version: 'onnx-v4',
    runtime_providers: ['CPU'], semantic_mode: 'qwen', semantic_model_version: 'qwen-improved',
  })});
  await s.run('refreshRuntimeIdentity()');
  assert.equal(s.elements.get('#runtime-badge').textContent, 'ONNX + Qwen');
  assert.match(s.elements.get('#job-caption').textContent, /qwen-improved/);
  s.context.fetch = async () => ({ok: true, json: async () => ({
    status: 'ok', model_ready: true, inference_mode: 'onnx', model_version: 'onnx-v4',
  })});
  await s.run('refreshRuntimeIdentity()');
  assert.equal(s.elements.get('#runtime-badge').textContent, 'ONNX');
});

test('one episode renders all matched rules with their official sources', () => {
  const s = setup();
  s.context.detail = { thesis: 'Тестовый тезис', observations: ['Тестовый факт'], reasoning: 'Тест', risk: 'Тест',
    rules: [20, 22].map(section => ({status: 'matched', label: `Правило ${section}`, section: `пункт ${section}`,
      document_name: 'Проверенный каталог', applicability: 'Автоматическое сопоставление. Кадр: 15 сек.',
      source_url: 'https://adilet.zan.kz/rus/docs/V1900019680'})) };
  s.context.violation = event(15);
  const chain = s.run('makeEvidenceChain(detail, violation)');
  assert.equal(chain.children.length, 6);
  for (const section of chain.children.slice(4)) {
    const link = section.children.at(-1);
    assert.equal(link.href, 'https://adilet.zan.kz/rus/docs/V1900019680');
    assert.equal(link.textContent, 'Открыть нормативный источник');
  }
});

test('one episode has three chronological context frames, not three event markers', () => {
  const s = setup();
  s.context.job = { verdict: 'violation', violations: [event(10, [11, 9, 10])] };
  s.run('renderCompleted(job, {})');
  assert.equal(s.elements.get('#event-markers').children.length, 1);
  assert.equal(s.elements.get('#result-count').textContent, '1 эпизод');
  const shots = s.elements.get('#event-list').children[0].children[0].children;
  assert.deepEqual(Array.from(shots, shot => shot.children[1].textContent),
    ['До пика · 00:09.0', 'У пика · 00:10.0', 'После пика · 00:11.0']);
  assert.equal(s.context.job.violations[0].evidence_frames[0].timestamp_sec, 11);
});

test('marker, timecode and evidence clicks synchronize the selected episode and video', () => {
  const s = setup();
  s.context.job = { verdict: 'violation', violations: [event(10), event(30)] };
  s.run('renderCompleted(job, {})');
  const cards = s.elements.get('#event-list').children;
  const markers = s.elements.get('#event-markers').children;
  markers[1].handlers.click();
  assert.equal(s.video.currentTime, 30);
  assert.equal(cards[1].classes.has('selected'), true);
  assert.equal(cards[1].scrolled, true);
  assert.equal(markers[1].attributes['aria-pressed'], 'true');
  cards[0].children[1].children[0].handlers.click();
  assert.equal(s.video.currentTime, 10);
  assert.equal(cards[1].classes.has('selected'), false);
  cards[1].children[0].children[0].handlers.click();
  assert.equal(s.video.currentTime, 29);
  assert.equal(markers[0].attributes['aria-pressed'], 'false');
  s.run('renderMarkers(currentViolations)');
  assert.equal(s.elements.get('#event-markers').children[1].attributes['aria-pressed'], 'true');
});

test('a single available frame is not duplicated or mislabeled as the peak', () => {
  const s = setup();
  s.context.frames = event(10, [8]).evidence_frames;
  const gallery = s.run('makeEvidenceGallery(frames, 0, 10)');
  assert.equal(gallery.children.length, 1);
  assert.equal(gallery.children[0].children[1].textContent, 'До пика · 00:08.0');
});

test('a valid negative result clears previous cards and markers', () => {
  const s = setup();
  s.context.job = { verdict: 'violation', violations: [event(10)] };
  s.run('renderCompleted(job, {}); renderCompleted({verdict: "no_violation", violations: []}, {})');
  assert.equal(s.elements.get('#event-list').children.length, 0);
  assert.equal(s.elements.get('#event-markers').children.length, 0);
  assert.equal(s.elements.get('#clear-panel').hidden, false);
});

for (const job of [
  {}, { verdict: 'unknown', violations: [] },
  { verdict: 'violation', violations: [] },
  { verdict: 'no_violation', violations: [event(10)] },
]) {
  test(`invalid completed response is rejected: ${JSON.stringify(job)}`, () => {
    const s = setup(); s.context.job = job;
    assert.throws(() => s.run('renderCompleted(job, {})'), /несогласованный/);
    assert.notEqual(s.context.document.body.dataset.demoState, 'no_violation');
  });
}

test('selecting an episode still works when the preview codec is unsupported', () => {
  const s = setup(); s.video.readyState = 0;
  s.context.job = { verdict: 'violation', violations: [event(10)] };
  s.run('renderCompleted(job, {})');
  const card = s.elements.get('#event-list').children[0];
  card.children[1].children[0].handlers.click();
  assert.equal(card.classes.has('selected'), true);
  assert.equal(s.video.currentTime, undefined);
});

test('a cancelled request cannot paint its completed result onto a different video', async () => {
  const s = setup();
  s.context.controller = new AbortController();
  s.context.fetch = async () => {
    s.context.controller.abort();
    return {ok: true, json: async () => ({status: 'done', verdict: 'no_violation', violations: []})};
  };
  s.run('setUiState("ready")');
  await assert.rejects(s.run('pollJob("old-job", "test-key", controller.signal)'), {name: 'AbortError'});
  assert.equal(s.context.document.body.dataset.demoState, 'ready');
});
