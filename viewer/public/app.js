// Stage 2: dataset viewer frontend.
// Single-page app over /api/examples, /api/stats, /api/flag.

import { escapeHtml, highlightJson } from '/utils.js';

const state = {
  file: 'train',
  page: 1,
  pageSize: 20,
  q: '',
};

const els = {
  tabs: document.querySelectorAll('.file-toggle .tab'),
  search: document.getElementById('search'),
  totalRecords: document.getElementById('total-records'),
  filteredRecords: document.getElementById('filtered-records'),
  flagCount: document.getElementById('flag-count'),
  catBars: document.getElementById('cat-bars'),
  typeSummary: document.getElementById('type-summary'),
  txnSummary: document.getElementById('txn-summary'),
  cards: document.getElementById('cards'),
  prev: document.getElementById('prev'),
  next: document.getElementById('next'),
  pageInfo: document.getElementById('page-info'),
  toast: document.getElementById('toast'),
  template: document.getElementById('card-template'),
};

// ---------- helpers ----------

let toastTimer = null;
function toast(msg, opts = {}) {
  els.toast.textContent = msg;
  els.toast.classList.toggle('error', !!opts.error);
  els.toast.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove('show'), 2200);
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// Debounce — used for the search box so we don't hammer the server on every keystroke.
function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    if (t) clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// ---------- rendering ----------

function renderPlaceholder(message) {
  els.cards.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'placeholder';
  div.innerHTML = message;
  els.cards.appendChild(div);
}

function renderCards(items) {
  els.cards.innerHTML = '';
  if (items.length === 0) {
    renderPlaceholder('No examples match the current filter.');
    return;
  }
  const frag = document.createDocumentFragment();
  for (const item of items) {
    const node = els.template.content.firstElementChild.cloneNode(true);
    if (item.flagged) node.classList.add('flagged');
    node.querySelector('.card-input').textContent = item.input;
    node.querySelector('.card-output code').innerHTML = highlightJson(item.output);
    const meta = node.querySelector('.card-meta');
    const sourceLabel = item.source ? item.source.replace(/\.jsonl$/, '') : '—';
    meta.textContent = `#${item.index} · ${sourceLabel}`;
    const flagBtn = node.querySelector('.flag-btn');
    if (item.flagged) {
      flagBtn.classList.add('flagged');
      flagBtn.textContent = 'flagged';
    }
    flagBtn.addEventListener('click', () => flagExample(item, node, flagBtn));
    frag.appendChild(node);
  }
  els.cards.appendChild(frag);
}

function renderStats(stats) {
  const cats = Object.entries(stats.categories).sort((a, b) => b[1] - a[1]);
  const max = cats.length > 0 ? cats[0][1] : 1;
  els.catBars.innerHTML = '';
  for (const [name, count] of cats) {
    const row = document.createElement('div');
    row.className = 'cat-bar';
    const pct = max > 0 ? (count / max) * 100 : 0;
    row.innerHTML = `
      <span class="cat-bar-label">${escapeHtml(name)}</span>
      <div class="cat-bar-track"><div class="cat-bar-fill" style="width:${pct.toFixed(1)}%"></div></div>
      <span class="cat-bar-count">${count}</span>
    `;
    els.catBars.appendChild(row);
  }
  const typeBits = Object.entries(stats.types).map(([k, v]) => `${k}: ${v}`).join('  ·  ');
  els.typeSummary.textContent = `types — ${typeBits || '—'}`;
  const txnBits = Object.entries(stats.txnCounts)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([k, v]) => `${k}×${v}`)
    .join('  ');
  els.txnSummary.textContent = `txns per input — ${txnBits || '—'} (total ${stats.totalTxns})`;
}

// ---------- actions ----------

async function flagExample(item, cardNode, btn) {
  const reason = window.prompt(
    `Flag example #${item.index} (${state.file}) as bad.\nReason (optional):`,
    '',
  );
  if (reason === null) return; // user cancelled
  try {
    await api('/api/flag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file: state.file, index: item.index, reason }),
    });
    cardNode.classList.add('flagged');
    btn.classList.add('flagged');
    btn.textContent = 'flagged';
    toast(`Flagged #${item.index}`);
    // Refresh flag count in topbar.
    loadExamples({ keepCards: true });
  } catch (e) {
    toast(`Flag failed: ${e.message}`, { error: true });
  }
}

async function loadExamples({ keepCards = false } = {}) {
  const params = new URLSearchParams({
    file: state.file,
    page: String(state.page),
    pageSize: String(state.pageSize),
  });
  if (state.q) params.set('q', state.q);

  try {
    const data = await api(`/api/examples?${params}`);
    els.totalRecords.textContent = `${data.totalFull} records`;
    els.filteredRecords.textContent = state.q
      ? `· ${data.total} filtered`
      : '';
    els.flagCount.textContent = data.flaggedCount > 0
      ? `· ${data.flaggedCount} flagged`
      : '';

    state.page = Math.min(state.page, data.pageCount);
    els.pageInfo.textContent = `page ${data.page} / ${data.pageCount}`;
    els.prev.disabled = data.page <= 1;
    els.next.disabled = data.page >= data.pageCount;

    if (!keepCards) renderCards(data.items);
  } catch (e) {
    renderPlaceholder(
      `<p>${escapeHtml(e.message)}</p><p>Run <code>python scripts/01_generate_dataset.py</code> then <code>python scripts/02_clean_dataset.py</code>, then refresh.</p>`,
    );
    els.totalRecords.textContent = '— records';
    els.filteredRecords.textContent = '';
    els.flagCount.textContent = '';
    els.pageInfo.textContent = 'page — / —';
    els.prev.disabled = true;
    els.next.disabled = true;
  }
}

async function loadStats() {
  try {
    const stats = await api(`/api/stats?file=${state.file}`);
    renderStats(stats);
  } catch {
    els.catBars.innerHTML = '';
    els.typeSummary.textContent = '';
    els.txnSummary.textContent = '';
  }
}

function refresh() {
  loadStats();
  loadExamples();
}

// ---------- wiring ----------

els.tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.classList.contains('active')) return;
    els.tabs.forEach(t => t.classList.toggle('active', t === tab));
    state.file = tab.dataset.file;
    state.page = 1;
    refresh();
  });
});

els.search.addEventListener(
  'input',
  debounce(() => {
    state.q = els.search.value.trim();
    state.page = 1;
    loadExamples();
  }, 200),
);

els.prev.addEventListener('click', () => {
  if (state.page > 1) {
    state.page -= 1;
    loadExamples();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
});

els.next.addEventListener('click', () => {
  state.page += 1;
  loadExamples();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

refresh();
