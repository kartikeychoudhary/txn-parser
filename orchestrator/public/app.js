// Single-page state store + render. No framework.

const state = {
  flows: [],
  jobs: new Map(),     // jobId -> meta + tail
  configs: [],         // list of available configs/*.json filenames
  selectedFlowId: null,
  selectedJobId: null,
  sockets: new Map(),  // jobId -> WebSocket
};

const $ = sel => document.querySelector(sel);

// ---------- Fetch helpers ----------

async function api(method, url, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(url, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: r.statusText }));
    throw new Error(err.error || `HTTP ${r.status}`);
  }
  return r.json();
}

// ---------- Initial load ----------

async function init() {
  const [{ flows }, { jobs }, { configs }] = await Promise.all([
    api("GET", "/api/flows"),
    api("GET", "/api/jobs"),
    api("GET", "/api/configs").catch(() => ({ configs: [] })),
  ]);
  state.flows = flows;
  state.configs = configs;
  for (const j of jobs) state.jobs.set(j.id, { ...j, tail: [] });

  renderFlows();
  renderJobList();

  for (const j of state.jobs.values()) {
    if (j.status === "running" || j.status === "stopping") {
      openSocket(j.id);
    }
  }
}

// ---------- Flow rendering ----------

function renderFlows() {
  const list = $("#flow-list");
  list.innerHTML = "";
  for (const f of state.flows) {
    const li = document.createElement("li");
    li.className = "flow-item" + (state.selectedFlowId === f.id ? " active" : "");
    li.innerHTML = `<div class="label"></div><div class="desc"></div>`;
    li.querySelector(".label").textContent = f.label;
    li.querySelector(".desc").textContent = f.description;
    li.addEventListener("click", () => selectFlow(f.id));
    list.appendChild(li);
  }
}

function selectFlow(id) {
  state.selectedFlowId = id;
  state.selectedJobId = null;
  renderFlows();
  renderJobList();
  renderFlowDetail();
}

function renderFlowDetail() {
  const flow = state.flows.find(f => f.id === state.selectedFlowId);
  $("#empty-state").classList.add("hidden");
  $("#job-view").classList.add("hidden");
  const panel = $("#flow-detail");
  if (!flow) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");

  $("#flow-label").textContent = flow.label;
  $("#flow-description").textContent = flow.description;

  const form = $("#flow-form");
  form.innerHTML = "";

  const basic = flow.args.filter(a => (a.group ?? "basic") === "basic");
  const advanced = flow.args.filter(a => a.group === "advanced");

  for (const arg of basic) form.appendChild(renderField(arg));

  if (advanced.length) {
    const det = document.createElement("details");
    det.className = "advanced-group";
    det.innerHTML = `<summary>Advanced (${advanced.length})</summary>`;
    const grid = document.createElement("div");
    grid.className = "args-form";
    for (const arg of advanced) grid.appendChild(renderField(arg));
    det.appendChild(grid);
    form.appendChild(det);
  }

  const extra = document.createElement("div");
  extra.className = "field field-extra";
  extra.innerHTML = `
    <label for="arg-extra">Additional args (whitespace-split, appended verbatim)</label>
    <textarea id="arg-extra" placeholder="--flag value …"></textarea>
  `;
  form.appendChild(extra);

  form.addEventListener("input", updateCommandPreview);
  form.addEventListener("change", e => {
    if (e.target.dataset?.type === "config") refreshConfigEditor();
  });

  // If this flow has a config arg, fetch the current contents into the editor.
  const cfgArg = flow.args.find(a => a.type === "config");
  if (cfgArg) refreshConfigEditor();

  updateCommandPreview();
}

function renderField(arg) {
  const wrap = document.createElement("div");
  const safeName = `arg-${arg.name}`;
  const helpHtml = arg.help ? `<span class="field-help">${escapeHtml(arg.help)}</span>` : "";

  if (arg.type === "flag") {
    wrap.className = "field flag";
    wrap.innerHTML = `
      <input type="checkbox" id="${safeName}" data-arg="${arg.name}" data-type="flag" ${arg.default ? "checked" : ""}/>
      <label for="${safeName}">--${arg.name}</label>
      ${helpHtml}
    `;
  } else if (arg.type === "choice") {
    wrap.className = "field";
    const opts = arg.choices.map(c => `<option value="${c}" ${c === arg.default ? "selected" : ""}>${c}</option>`).join("");
    wrap.innerHTML = `
      <label for="${safeName}">--${arg.name}${arg.required ? " *" : ""}</label>
      <select id="${safeName}" data-arg="${arg.name}" data-type="choice">${opts}</select>
      ${helpHtml}
    `;
  } else if (arg.type === "positional") {
    wrap.className = "field field-extra";
    wrap.innerHTML = `
      <label for="${safeName}">${arg.name} (positional${arg.required ? ", required" : ""})</label>
      <input type="text" id="${safeName}" data-arg="${arg.name}" data-type="positional" value="${escapeAttr(arg.default ?? "")}"/>
      ${helpHtml}
    `;
  } else if (arg.type === "config") {
    wrap.className = "field field-extra config-field";
    const opts = state.configs
      .map(n => `<option value="${escapeAttr(n)}" ${n === arg.default ? "selected" : ""}>${escapeHtml(n)}</option>`)
      .join("");
    wrap.innerHTML = `
      <label for="${safeName}">--${arg.name}${arg.required ? " *" : ""} (configs/&lt;name&gt;.json)</label>
      <select id="${safeName}" data-arg="${arg.name}" data-type="config">${opts || `<option value="${escapeAttr(arg.default ?? "")}">${escapeHtml(arg.default ?? "")}</option>`}</select>
      ${helpHtml}
      <details class="config-editor">
        <summary>View / edit config JSON</summary>
        <div class="config-editor-body">
          <textarea id="config-editor-text" spellcheck="false" rows="14" placeholder="Loading…"></textarea>
          <div class="config-editor-actions">
            <button type="button" id="config-save" class="btn btn-primary">Save</button>
            <button type="button" id="config-reload" class="btn">Reload</button>
            <span id="config-status" class="muted"></span>
          </div>
        </div>
      </details>
    `;
  } else {
    const inputType = (arg.type === "int" || arg.type === "float") ? "number" : "text";
    const step = arg.type === "float" ? "any" : "1";
    wrap.className = "field";
    wrap.innerHTML = `
      <label for="${safeName}">--${arg.name}${arg.required ? " *" : ""}</label>
      <input type="${inputType}" step="${step}" id="${safeName}" data-arg="${arg.name}" data-type="${arg.type}" value="${escapeAttr(arg.default ?? "")}" ${arg.required ? "required" : ""}/>
      ${helpHtml}
    `;
  }
  return wrap;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function escapeAttr(s) { return escapeHtml(s); }

// ---------- Config editor ----------

async function refreshConfigEditor() {
  const sel = document.querySelector('[data-type="config"]');
  if (!sel) return;
  const name = sel.value;
  const ta = document.getElementById("config-editor-text");
  const status = document.getElementById("config-status");
  if (!ta || !status) return;
  status.textContent = "Loading…";
  try {
    const r = await fetch(`/api/configs/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const text = await r.text();
    // Pretty-print so editing is sane
    try { ta.value = JSON.stringify(JSON.parse(text), null, 2); }
    catch { ta.value = text; }
    status.textContent = `Loaded ${name}`;
  } catch (e) {
    ta.value = "";
    status.textContent = `Failed: ${e.message}`;
  }
}

document.addEventListener("click", async e => {
  if (e.target.id === "config-save") {
    const sel = document.querySelector('[data-type="config"]');
    const ta = document.getElementById("config-editor-text");
    const status = document.getElementById("config-status");
    if (!sel || !ta || !status) return;
    let body;
    try { body = JSON.parse(ta.value); }
    catch (err) { status.textContent = `Invalid JSON: ${err.message}`; return; }
    status.textContent = "Saving…";
    try {
      await api("PUT", `/api/configs/${encodeURIComponent(sel.value)}`, body);
      status.textContent = `Saved ${sel.value}`;
    } catch (err) {
      status.textContent = `Save failed: ${err.message}`;
    }
  } else if (e.target.id === "config-reload") {
    refreshConfigEditor();
  }
});

function collectFormValues() {
  const args = {};
  let extraArgs = "";
  document.querySelectorAll("#flow-form [data-arg]").forEach(el => {
    const name = el.dataset.arg;
    const type = el.dataset.type;
    if (type === "flag") args[name] = el.checked;
    else if (type === "int") args[name] = el.value === "" ? "" : parseInt(el.value, 10);
    else if (type === "float") args[name] = el.value === "" ? "" : parseFloat(el.value);
    else args[name] = el.value;
  });
  const extraEl = document.getElementById("arg-extra");
  if (extraEl) extraArgs = extraEl.value;
  return { args, extraArgs };
}

function updateCommandPreview() {
  const flow = state.flows.find(f => f.id === state.selectedFlowId);
  if (!flow) return;
  const { args, extraArgs } = collectFormValues();
  const parts = [flow.script];
  for (const spec of flow.args) {
    const v = args[spec.name];
    if (spec.type === "flag") { if (v === true) parts.push(`--${spec.name}`); continue; }
    if (spec.type === "positional") continue;
    if (v === "" || v === null || v === undefined) continue;
    const rendered = spec.type === "config" ? `configs/${v}` : String(v);
    parts.push(`--${spec.name}`, rendered);
  }
  if (extraArgs && extraArgs.trim()) parts.push(...extraArgs.trim().split(/\s+/));
  for (const spec of flow.args) {
    if (spec.type === "positional" && args[spec.name]) parts.push(String(args[spec.name]));
  }
  $("#command-preview-text").textContent = "python " + parts.map(p => /\s/.test(p) ? `"${p}"` : p).join(" ");
}

const GPU_FLOWS = new Set([
  "stage_3_train_teacher", "stage_4_eval", "stage_5_legacy",
  "stage_6_train_student", "predict_one",
]);

$("#start-btn").addEventListener("click", async () => {
  const flow = state.flows.find(f => f.id === state.selectedFlowId);
  if (!flow) return;
  const { args, extraArgs } = collectFormValues();
  if (GPU_FLOWS.has(flow.id)) {
    const busy = [...state.jobs.values()].some(j =>
      GPU_FLOWS.has(j.flowId) && (j.status === "running" || j.status === "stopping")
    );
    if (busy && !confirm("Another GPU-using job is already running. Starting this one may OOM. Continue?")) return;
  }
  try {
    const { job } = await api("POST", "/api/jobs", { flowId: flow.id, args, extraArgs });
    state.jobs.set(job.id, { ...job, tail: [] });
    openSocket(job.id);
    selectJob(job.id);
    renderJobList();
  } catch (e) {
    alert("Failed to start: " + e.message);
  }
});

// ---------- Job list ----------

function relTime(iso) {
  if (!iso) return "";
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function elapsed(startISO, endISO) {
  const start = new Date(startISO).getTime();
  const end = endISO ? new Date(endISO).getTime() : Date.now();
  const s = Math.max(0, Math.floor((end - start) / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return h ? `${h}h ${m}m ${ss}s` : (m ? `${m}m ${ss}s` : `${ss}s`);
}

function renderJobList() {
  const list = $("#job-list");
  list.innerHTML = "";
  const jobs = [...state.jobs.values()].sort((a, b) => (a.startedAt < b.startedAt ? 1 : -1));
  for (const j of jobs) {
    const li = document.createElement("li");
    li.className = "job-item" + (state.selectedJobId === j.id ? " active" : "");
    const when = (j.status === "running" || j.status === "stopping")
      ? elapsed(j.startedAt)
      : relTime(j.endedAt || j.startedAt);
    li.innerHTML = `
      <div class="row">
        <span><span class="status-dot ${j.status}"></span><span class="label"></span></span>
        <span class="when"></span>
      </div>
    `;
    li.querySelector(".label").textContent = j.label;
    li.querySelector(".when").textContent = when;
    li.addEventListener("click", () => selectJob(j.id));
    list.appendChild(li);
  }
  refreshRunningCount();
}

function refreshRunningCount() {
  const n = [...state.jobs.values()].filter(j => j.status === "running" || j.status === "stopping").length;
  $("#running-count").textContent = `● ${n} running`;
}

// ---------- Job detail (console) ----------

let userScrolledUp = false;
const consoleEl = $("#console");
consoleEl.addEventListener("scroll", () => {
  const atBottom = consoleEl.scrollHeight - consoleEl.scrollTop - consoleEl.clientHeight < 8;
  userScrolledUp = !atBottom;
  $("#jump-live").classList.toggle("hidden", !userScrolledUp);
});
$("#jump-live").addEventListener("click", () => {
  consoleEl.scrollTop = consoleEl.scrollHeight;
  userScrolledUp = false;
  $("#jump-live").classList.add("hidden");
});

const DOM_LINE_CAP = 5000;

function appendLine(jobId, line) {
  if (state.selectedJobId !== jobId) return;
  const span = document.createElement("span");
  span.className = line.stream === "stderr" ? "err" : (line.stream === "log" ? "meta" : "out");
  span.textContent = (line.ts ? `[${line.ts}] ` : "") + line.text + "\n";
  consoleEl.appendChild(span);
  while (consoleEl.childNodes.length > DOM_LINE_CAP) consoleEl.removeChild(consoleEl.firstChild);
  if (!userScrolledUp) consoleEl.scrollTop = consoleEl.scrollHeight;
}

function renderJobDetail(jobId) {
  const job = state.jobs.get(jobId);
  if (!job) return;
  $("#empty-state").classList.add("hidden");
  $("#flow-detail").classList.add("hidden");
  $("#job-view").classList.remove("hidden");

  $("#job-label").textContent = job.label;
  $("#job-command").textContent = job.command;
  $("#job-started").textContent = new Date(job.startedAt).toLocaleString();
  $("#job-elapsed").textContent = elapsed(job.startedAt, job.endedAt);

  const chip = $("#job-status");
  chip.className = "status-chip " + job.status;
  chip.textContent = job.status;

  const stopBtn = $("#stop-btn");
  if (job.status === "running") {
    stopBtn.classList.remove("hidden", "force");
    stopBtn.textContent = "Stop";
  } else if (job.status === "stopping") {
    stopBtn.classList.remove("hidden");
    stopBtn.classList.add("force");
    stopBtn.textContent = "Force kill";
  } else {
    stopBtn.classList.add("hidden");
  }

  const terminal = ["succeeded", "failed", "killed", "orphaned"].includes(job.status);
  $("#delete-btn").classList.toggle("hidden", !terminal);

  $("#download-log").setAttribute("href", `/api/jobs/${job.id}/log`);

  consoleEl.innerHTML = "";
  for (const l of job.tail || []) appendLine(jobId, l);
  consoleEl.scrollTop = consoleEl.scrollHeight;
  userScrolledUp = false;
  $("#jump-live").classList.add("hidden");
}

function selectJob(id) {
  state.selectedJobId = id;
  state.selectedFlowId = null;
  renderFlows();
  renderJobList();
  renderJobDetail(id);
}

$("#stop-btn").addEventListener("click", async () => {
  const id = state.selectedJobId;
  if (!id) return;
  try { await api("POST", `/api/jobs/${id}/stop`); }
  catch (e) { alert("Stop failed: " + e.message); }
});

$("#delete-btn").addEventListener("click", async () => {
  const id = state.selectedJobId;
  if (!id) return;
  if (!confirm("Delete this run and its log?")) return;
  try { await api("DELETE", `/api/jobs/${id}`); }
  catch (e) { alert("Delete failed: " + e.message); return; }
  state.jobs.delete(id);
  state.selectedJobId = null;
  $("#job-view").classList.add("hidden");
  $("#empty-state").classList.remove("hidden");
  renderJobList();
});

// ---------- WebSocket subscription ----------

function openSocket(jobId) {
  if (state.sockets.has(jobId)) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/jobs/${jobId}`);
  state.sockets.set(jobId, ws);

  ws.addEventListener("message", ev => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "snapshot") {
      state.jobs.set(jobId, { ...msg.job, tail: msg.job.tail || [] });
      if (state.selectedJobId === jobId) renderJobDetail(jobId);
      renderJobList();
    } else if (msg.type === "line") {
      const job = state.jobs.get(jobId);
      if (!job) return;
      job.tail = job.tail || [];
      job.tail.push({ text: msg.text, stream: msg.stream, ts: msg.ts });
      if (job.tail.length > 5000) job.tail.shift();
      appendLine(jobId, msg);
    } else if (msg.type === "status") {
      const job = state.jobs.get(jobId) || { id: jobId };
      Object.assign(job, msg);
      if (msg.status) job.status = msg.status;
      state.jobs.set(jobId, job);
      if (state.selectedJobId === jobId) renderJobDetail(jobId);
      renderJobList();
      if (["succeeded", "failed", "killed", "orphaned"].includes(job.status)) {
        ws.close();
        state.sockets.delete(jobId);
      }
    }
  });

  ws.addEventListener("close", () => {
    state.sockets.delete(jobId);
    const job = state.jobs.get(jobId);
    if (job && (job.status === "running" || job.status === "stopping")) {
      setTimeout(() => openSocket(jobId), 1500);
    }
  });
}

// ---------- Periodic elapsed refresh ----------

setInterval(() => {
  if (state.selectedJobId) {
    const j = state.jobs.get(state.selectedJobId);
    if (j && (j.status === "running" || j.status === "stopping")) {
      $("#job-elapsed").textContent = elapsed(j.startedAt);
    }
  }
  renderJobList();
}, 1000);

window.addEventListener("DOMContentLoaded", init);
