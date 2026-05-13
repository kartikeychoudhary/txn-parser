// Stage 7 playground frontend.
// Calls POST /api/infer, renders side-by-side cards with diff highlighting
// for the "both" mode, and keeps the last 20 inputs in localStorage.

import { escapeHtml, highlightJson, extractJson } from '/utils.js';

const HISTORY_KEY = 'playground_history_v1';
const MAX_HISTORY = 20;

const els = {
  modelStatus: document.getElementById('model-status'),
  modelSelect: document.getElementById('model-select'),
  maxTokens: document.getElementById('max-tokens'),
  inputBox: document.getElementById('input-box'),
  runBtn: document.getElementById('run-btn'),
  outputArea: document.getElementById('output-area'),
  history: document.getElementById('history'),
  toast: document.getElementById('toast'),
};

// ---------- toast ----------

let toastTimer = null;
function toast(msg, opts = {}) {
  els.toast.textContent = msg;
  els.toast.classList.toggle('error', !!opts.error);
  els.toast.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove('show'), 2400);
}

// ---------- model status ----------

async function loadModelStatus() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const available = data.models || {};
    const teacher = available.teacher;
    const student = available.student;

    const bits = [];
    bits.push(renderStatusPill('Teacher', teacher));
    bits.push(renderStatusPill('Student', student));
    els.modelStatus.innerHTML = bits.join('');

    // Disable options for missing models.
    for (const opt of els.modelSelect.options) {
      let disabled = false;
      if (opt.value === 'teacher' && !teacher) disabled = true;
      if (opt.value === 'student' && !student) disabled = true;
      if (opt.value === 'both' && (!teacher || !student)) disabled = true;
      opt.disabled = disabled;
    }

    if (els.modelSelect.options[els.modelSelect.selectedIndex]?.disabled) {
      // Fall back to the first enabled option.
      for (const opt of els.modelSelect.options) {
        if (!opt.disabled) { els.modelSelect.value = opt.value; break; }
      }
    }
  } catch (e) {
    els.modelStatus.innerHTML = `<span class="muted">model status unavailable</span>`;
  }
}

function renderStatusPill(label, info) {
  if (!info) return `<span class="status-pill missing">${label}: not loaded</span>`;
  const cls = info.ready ? 'ready' : 'loading';
  const tail = info.ready ? '' : ' (loading…)';
  return `<span class="status-pill ${cls}">${label}${tail}</span>`;
}

// ---------- inference ----------

let inflight = null;

async function runInference() {
  const input = els.inputBox.value.trim();
  if (!input) {
    toast('input is empty', { error: true });
    return;
  }
  if (inflight) {
    toast('already running…');
    return;
  }
  const model = els.modelSelect.value;
  const maxTokens = Math.max(16, Math.min(2048, Number(els.maxTokens.value) || 512));

  els.runBtn.disabled = true;
  els.runBtn.textContent = 'Running…';
  renderLoading(model);

  const ctrl = new AbortController();
  inflight = ctrl;
  const t0 = performance.now();
  try {
    const res = await fetch('/api/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, input, max_tokens: maxTokens }),
      signal: ctrl.signal,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

    renderResults(model, data);
    pushHistory(input);
  } catch (e) {
    if (e.name !== 'AbortError') {
      toast(`inference failed: ${e.message}`, { error: true });
      els.outputArea.innerHTML = `<div class="placeholder error">${escapeHtml(e.message)}</div>`;
    }
  } finally {
    inflight = null;
    els.runBtn.disabled = false;
    els.runBtn.textContent = 'Run ▶';
    const elapsed = (performance.now() - t0).toFixed(0);
    console.log(`[infer] round-trip ${elapsed}ms`);
  }
}

function renderLoading(model) {
  const targets = model === 'both' ? ['teacher', 'student'] : [model];
  els.outputArea.innerHTML = targets
    .map((m) => `<article class="result-card"><header class="result-head">${m}</header><pre class="result-body">running…</pre></article>`)
    .join('');
}

function renderResults(modelSel, data) {
  const targets = modelSel === 'both' ? ['teacher', 'student'] : [modelSel];

  // For "both" mode, compute per-line diff on the pretty-printed JSON outputs.
  let diffLines = null;
  if (modelSel === 'both') {
    const tObj = data.teacher && !data.teacher.error ? extractJson(data.teacher.output) : null;
    const sObj = data.student && !data.student.error ? extractJson(data.student.output) : null;
    if (tObj && sObj) {
      const tStr = JSON.stringify(tObj, null, 2);
      const sStr = JSON.stringify(sObj, null, 2);
      const tLines = tStr.split('\n');
      const sLines = sStr.split('\n');
      const max = Math.max(tLines.length, sLines.length);
      const tDiff = new Array(max).fill(false);
      const sDiff = new Array(max).fill(false);
      for (let i = 0; i < max; i++) {
        if (tLines[i] !== sLines[i]) {
          tDiff[i] = true;
          sDiff[i] = true;
        }
      }
      diffLines = { teacher: tDiff, student: sDiff };
    }
  }

  els.outputArea.innerHTML = targets.map((m) => renderCard(m, data[m], diffLines?.[m])).join('');
}

function renderCard(modelName, payload, diffMask) {
  if (!payload) {
    return `<article class="result-card"><header class="result-head">${modelName}</header>
            <pre class="result-body error">(no response)</pre></article>`;
  }
  if (payload.error) {
    return `<article class="result-card error"><header class="result-head">${modelName}</header>
            <pre class="result-body error">${escapeHtml(payload.error)}</pre></article>`;
  }
  const parsed = extractJson(payload.output);
  const validClass = parsed ? 'ok' : 'bad';
  const validLabel = parsed ? 'JSON ✓' : 'JSON ✗';
  const bodyHtml = parsed
    ? renderJsonWithDiff(parsed, diffMask)
    : `<span class="raw-text">${escapeHtml(payload.output || '(empty)')}</span>`;
  const meta = [
    payload.gguf ? `<code>${escapeHtml(payload.gguf)}</code>` : '',
    `${payload.latency_ms?.toFixed?.(1) ?? '—'} ms`,
    payload.tokens ? `${payload.tokens} tok` : '',
  ].filter(Boolean).join('  ·  ');
  return `
    <article class="result-card">
      <header class="result-head">
        <span class="model-name">${modelName}</span>
        <span class="badge ${validClass}">${validLabel}</span>
      </header>
      <pre class="result-body"><code>${bodyHtml}</code></pre>
      <footer class="result-foot">${meta}</footer>
    </article>
  `;
}

function renderJsonWithDiff(obj, diffMask) {
  const html = highlightJson(obj);
  if (!diffMask) return html;
  return html.split('\n').map((line, i) => {
    const cls = diffMask[i] ? 'line diff' : 'line';
    return `<span class="${cls}">${line || ' '}</span>`;
  }).join('\n');
}

// ---------- history ----------

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function saveHistory(items) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items)); }
  catch { /* quota / private mode */ }
}

function pushHistory(input) {
  const items = loadHistory().filter((it) => it.input !== input);
  items.unshift({ input, ts: Date.now() });
  while (items.length > MAX_HISTORY) items.pop();
  saveHistory(items);
  renderHistory();
}

function renderHistory() {
  const items = loadHistory();
  if (items.length === 0) {
    els.history.innerHTML = `<li class="history-empty muted">no recent inputs yet</li>`;
    return;
  }
  els.history.innerHTML = items.map((it, idx) => `
    <li class="history-item" data-index="${idx}">
      <span class="history-text">${escapeHtml(it.input)}</span>
      <button class="history-del" data-del="${idx}" title="remove">×</button>
    </li>
  `).join('');
}

els.history.addEventListener('click', (e) => {
  const delBtn = e.target.closest('[data-del]');
  if (delBtn) {
    const idx = Number(delBtn.dataset.del);
    const items = loadHistory();
    items.splice(idx, 1);
    saveHistory(items);
    renderHistory();
    e.stopPropagation();
    return;
  }
  const item = e.target.closest('.history-item');
  if (!item) return;
  const idx = Number(item.dataset.index);
  const items = loadHistory();
  if (items[idx]) {
    els.inputBox.value = items[idx].input;
    els.inputBox.focus();
  }
});

// ---------- wiring ----------

els.runBtn.addEventListener('click', runInference);
els.inputBox.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    runInference();
  }
});

loadModelStatus();
renderHistory();
// Re-poll model status every 5s while any worker is still loading.
const statusPoll = setInterval(async () => {
  const res = await fetch('/api/models').then((r) => r.json()).catch(() => null);
  if (!res) return;
  const allReady = Object.values(res.models || {}).every((m) => m.ready);
  loadModelStatus();
  if (allReady) clearInterval(statusPoll);
}, 5000);
