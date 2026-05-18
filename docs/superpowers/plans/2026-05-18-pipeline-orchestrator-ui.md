# Pipeline Orchestrator UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a standalone web UI at `http://localhost:3100` that launches the project's pipeline scripts, streams their console output live, supports parallel jobs, exposes a two-stage force-quit, and persists run history to disk.

**Architecture:** New top-level `orchestrator/` directory holds a self-contained Node.js + Express + `ws` app. Three server modules: `flows.js` (curated flow catalogue + argv builder, pure), `jobs.js` (spawn/stream/persist/kill lifecycle), `server.js` (HTTP + WebSocket routes). Vanilla HTML/CSS/JS frontend served from `orchestrator/public/`. Each run gets a directory under `orchestrator/runs/<jobId>/` containing `meta.json` and `output.log`.

**Tech Stack:** Node.js 20+, Express 4, `ws`, `ulid`. Node's built-in test runner (`node --test`) — no jest/mocha needed. Frontend uses no framework or bundler.

**Spec:** `docs/superpowers/specs/2026-05-18-pipeline-orchestrator-ui-design.md`

---

## File Structure

```
orchestrator/
├── package.json                  # deps: express, ws, ulid
├── server.js                     # entry: HTTP + WebSocket, mounts routes, boot scan
├── flows.js                      # flow catalogue + buildArgv()
├── jobs.js                       # JobManager class: spawn, stream, persist, stop
├── public/
│   ├── index.html                # single-page UI skeleton
│   ├── app.js                    # state store, fetch, WS, render
│   └── style.css                 # dark theme
├── runs/                         # gitignored; created on demand
└── test/
    ├── flows.test.js             # argv builder per flow type
    ├── jobs.test.js              # spawn, stream-to-log, exit status, orphan recovery
    └── jobs.stop.test.js         # graceful then hard-kill paths
```

Also modified:
- `.gitignore` — add `orchestrator/runs/`.

---

## Task 1: Scaffold the orchestrator package

**Files:**
- Create: `orchestrator/package.json`
- Create: `orchestrator/.gitignore`
- Modify: `.gitignore` (root)
- Create: `orchestrator/server.js` (stub)

- [ ] **Step 1: Create `orchestrator/package.json`**

```json
{
  "name": "pipeline-orchestrator",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "engines": { "node": ">=20" },
  "scripts": {
    "start": "node server.js",
    "dev": "node --watch server.js",
    "test": "node --test test/"
  },
  "dependencies": {
    "express": "^4.21.1",
    "ulid": "^2.3.0",
    "ws": "^8.18.0"
  }
}
```

- [ ] **Step 2: Create `orchestrator/.gitignore`**

```
node_modules/
runs/
```

- [ ] **Step 3: Add `orchestrator/runs/` to root `.gitignore`**

Append to `C:\work\llm training\.gitignore`:

```
# Orchestrator UI run history
orchestrator/runs/
orchestrator/node_modules/
```

- [ ] **Step 4: Create stub `orchestrator/server.js`**

```js
import express from "express";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.get("/api/health", (_req, res) => res.json({ ok: true }));

const PORT = Number(process.env.PORT ?? 3100);
const HOST = process.env.HOST ?? "127.0.0.1";
app.listen(PORT, HOST, () => {
  console.log(`[orchestrator] listening on http://${HOST}:${PORT}`);
});
```

- [ ] **Step 5: Install deps and verify**

Run from `C:\work\llm training\orchestrator`:

```
npm install
```

Expected: creates `node_modules/`, `package-lock.json`. Then:

```
node --check server.js
```

Expected: exits 0 (no syntax errors).

- [ ] **Step 6: Commit**

```
git add orchestrator/package.json orchestrator/package-lock.json orchestrator/.gitignore orchestrator/server.js .gitignore
git commit -m "orchestrator: scaffold standalone Express app"
```

---

## Task 2: Flow catalogue + argv builder

**Files:**
- Create: `orchestrator/flows.js`
- Create: `orchestrator/test/flows.test.js`

- [ ] **Step 1: Write the failing test at `orchestrator/test/flows.test.js`**

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { FLOWS, getFlow, buildArgv } from "../flows.js";

test("catalogue exposes all expected flows", () => {
  const ids = FLOWS.map(f => f.id);
  assert.deepEqual(ids.sort(), [
    "export_gguf",
    "predict_one",
    "stage_1_generate_raw",
    "stage_2_clean_dataset",
    "stage_3_train_teacher",
    "stage_4_eval",
    "stage_5_legacy",
    "stage_5_multiprovider",
    "stage_6_train_student",
  ]);
});

test("getFlow returns null for unknown id", () => {
  assert.equal(getFlow("nope"), null);
});

test("buildArgv: int arg only included when set", () => {
  const flow = getFlow("stage_6_train_student");
  const argv = buildArgv(flow, { "batch-size": 16 }, "");
  assert.deepEqual(argv, ["scripts/06_train_student.py", "--batch-size", "16"]);
});

test("buildArgv: flag args are omitted when false, present when true", () => {
  const flow = getFlow("stage_6_train_student");
  const off = buildArgv(flow, { resume: false }, "");
  assert.deepEqual(off, ["scripts/06_train_student.py"]);
  const on = buildArgv(flow, { resume: true }, "");
  assert.deepEqual(on, ["scripts/06_train_student.py", "--resume"]);
});

test("buildArgv: string arg quoted-safe (split on spaces in shell, we pass as one argv)", () => {
  const flow = getFlow("predict_one");
  const argv = buildArgv(flow, { model: "models/student/gguf", input: "500 rs on beer" }, "");
  assert.deepEqual(argv, [
    "scripts/predict_one.py",
    "--model", "models/student/gguf",
    "500 rs on beer",
  ]);
});

test("buildArgv: extraArgs are appended (split on whitespace, empty preserved as empty)", () => {
  const flow = getFlow("stage_4_eval");
  const argv = buildArgv(flow, { model: "models/teacher/gguf" }, "--limit 50 --no-grammar");
  assert.deepEqual(argv, [
    "scripts/04_eval.py",
    "--model", "models/teacher/gguf",
    "--limit", "50", "--no-grammar",
  ]);
});

test("buildArgv: float arg formatted without forcing decimals when integer-valued", () => {
  const flow = getFlow("stage_6_train_student");
  const argv = buildArgv(flow, { epochs: 2 }, "");
  assert.deepEqual(argv, ["scripts/06_train_student.py", "--epochs", "2"]);
});

test("buildArgv: choice arg validated", () => {
  const flow = getFlow("stage_5_legacy");
  const argv = buildArgv(flow, { phase: "label" }, "");
  assert.deepEqual(argv, ["scripts/05_generate_distillation_data.py", "--phase", "label"]);
  assert.throws(() => buildArgv(flow, { phase: "bogus" }, ""), /invalid choice/);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `orchestrator/`:

```
npm test
```

Expected: fails — `flows.js` not found.

- [ ] **Step 3: Implement `orchestrator/flows.js`**

```js
// flows.js — curated pipeline flow catalogue.
// Each flow declares its script path and the args we expose in the UI.
// buildArgv translates UI form values into a Python argv array.

export const FLOWS = [
  {
    id: "stage_1_generate_raw",
    label: "Stage 1 — Generate raw dataset",
    description: "DeepSeek synthesizes raw transaction examples into data/raw/.",
    script: "scripts/01_generate_dataset.py",
    args: [
      { name: "batches",     type: "int",    default: 25 },
      { name: "start-batch", type: "int",    default: 1 },
      { name: "model",       type: "string", default: "deepseek-chat" },
      { name: "temperature", type: "float",  default: 1.0 },
      { name: "force",       type: "flag",   default: false },
    ],
  },
  {
    id: "stage_2_clean_dataset",
    label: "Stage 2 — Clean + dedupe + split",
    description: "Validate raw batches, dedupe, and split into train/eval.",
    script: "scripts/02_clean_dataset.py",
    args: [
      { name: "eval-frac", type: "float", default: 0.1 },
      { name: "seed",      type: "int",   default: 42 },
    ],
  },
  {
    id: "stage_3_train_teacher",
    label: "Stage 3 — Train teacher",
    description: "QLoRA fine-tune of Gemma 4 E2B on data/clean/.",
    script: "scripts/03_train_teacher.py",
    args: [
      { name: "batch-size",      type: "int",   default: 4 },
      { name: "grad-accum",      type: "int",   default: 4 },
      { name: "eval-batch-size", type: "int",   default: 8 },
      { name: "epochs",          type: "float", default: 3.0 },
      { name: "max-seq-length",  type: "int",   default: 1024 },
      { name: "max-steps",       type: "int",   default: -1 },
      { name: "resume",          type: "flag",  default: false },
      { name: "skip-gguf",       type: "flag",  default: false },
      { name: "force",           type: "flag",  default: false },
    ],
  },
  {
    id: "stage_4_eval",
    label: "Stage 4 — Evaluate",
    description: "Run a model against data/clean/eval.jsonl.",
    script: "scripts/04_eval.py",
    args: [
      { name: "model",      type: "string", default: "models/teacher/gguf", required: true },
      { name: "limit",      type: "int",    default: 0 },
      { name: "batch-size", type: "int",    default: 16 },
      { name: "no-grammar", type: "flag",   default: false },
    ],
  },
  {
    id: "stage_5_legacy",
    label: "Stage 5 (legacy) — Teacher-labeled distillation",
    description: "Teacher fp16 labels DeepSeek-generated synthetic inputs.",
    script: "scripts/05_generate_distillation_data.py",
    args: [
      { name: "phase",    type: "choice", choices: ["all", "inputs", "label", "eval"], default: "all" },
      { name: "n-inputs", type: "int",    default: 30000 },
      { name: "batch-size", type: "int",  default: 16 },
      { name: "limit",    type: "int",    default: 0 },
      { name: "retry-failed", type: "flag", default: false },
      { name: "retry-validation-failed", type: "flag", default: false },
    ],
  },
  {
    id: "stage_5_multiprovider",
    label: "Stage 5 — Multi-provider distillation",
    description: "DeepSeek + Gemini act as labelers via providers config.",
    script: "scripts/05_generate_distillation_data.py",
    args: [
      { name: "phase",           type: "choice", choices: ["inputs", "label"], default: "inputs" },
      { name: "provider-config", type: "string", default: "configs/example_providers.json", required: true },
      { name: "multi-provider",  type: "flag",   default: true },
      { name: "limit",           type: "int",    default: 0 },
      { name: "dry-run-quota",   type: "flag",   default: false },
    ],
  },
  {
    id: "stage_6_train_student",
    label: "Stage 6 — Train student",
    description: "QLoRA fine-tune of Gemma 3 270M on data/distill/train.jsonl.",
    script: "scripts/06_train_student.py",
    args: [
      { name: "batch-size",      type: "int",   default: 8 },
      { name: "grad-accum",      type: "int",   default: 2 },
      { name: "eval-batch-size", type: "int",   default: 8 },
      { name: "epochs",          type: "float", default: 2.0 },
      { name: "max-steps",       type: "int",   default: -1 },
      { name: "resume",          type: "flag",  default: false },
      { name: "skip-gguf",       type: "flag",  default: false },
      { name: "skip-comparison", type: "flag",  default: false },
    ],
  },
  {
    id: "predict_one",
    label: "Predict one",
    description: "Run a single prediction through a GGUF or adapter.",
    script: "scripts/predict_one.py",
    args: [
      { name: "model",      type: "string",     default: "models/student/gguf", required: true },
      { name: "no-grammar", type: "flag",       default: false },
      { name: "input",      type: "positional", default: "", required: true },
    ],
  },
  {
    id: "export_gguf",
    label: "Export GGUF",
    description: "Re-export a trained adapter to GGUF (no retraining).",
    script: "scripts/export_gguf.py",
    args: [
      { name: "role",   type: "choice", choices: ["teacher", "student"], default: "student", required: true },
      { name: "quants", type: "string", default: "q4_k_m" },
    ],
  },
];

export function getFlow(id) {
  return FLOWS.find(f => f.id === id) ?? null;
}

function formatNumber(n) {
  // Avoid forcing decimals on integer-valued floats: 2.0 -> "2", 2.5 -> "2.5"
  return Number.isInteger(n) ? String(n) : String(n);
}

export function buildArgv(flow, values, extraArgsString) {
  if (!flow) throw new Error("buildArgv: flow is required");
  const argv = [flow.script];
  const positionals = [];

  for (const spec of flow.args) {
    const raw = values?.[spec.name];
    const hasValue = raw !== undefined && raw !== null && raw !== "";

    if (spec.type === "flag") {
      if (raw === true) argv.push(`--${spec.name}`);
      continue;
    }
    if (spec.type === "positional") {
      if (hasValue) positionals.push(String(raw));
      else if (spec.required) throw new Error(`missing required positional: ${spec.name}`);
      continue;
    }
    if (!hasValue) {
      if (spec.required) throw new Error(`missing required arg: ${spec.name}`);
      continue;
    }
    if (spec.type === "choice") {
      if (!spec.choices.includes(raw)) {
        throw new Error(`invalid choice for ${spec.name}: ${raw}`);
      }
      argv.push(`--${spec.name}`, String(raw));
      continue;
    }
    if (spec.type === "int") {
      argv.push(`--${spec.name}`, String(parseInt(raw, 10)));
      continue;
    }
    if (spec.type === "float") {
      argv.push(`--${spec.name}`, formatNumber(Number(raw)));
      continue;
    }
    if (spec.type === "string") {
      argv.push(`--${spec.name}`, String(raw));
      continue;
    }
    throw new Error(`buildArgv: unknown arg type ${spec.type}`);
  }

  // Append extra args (whitespace-split, empty tokens dropped)
  if (extraArgsString && extraArgsString.trim()) {
    const tokens = extraArgsString.trim().split(/\s+/);
    for (const t of tokens) argv.push(t);
  }

  // Positionals come last
  for (const p of positionals) argv.push(p);

  return argv;
}
```

- [ ] **Step 4: Run tests to verify they pass**

```
npm test
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```
git add orchestrator/flows.js orchestrator/test/flows.test.js
git commit -m "orchestrator: flow catalogue with typed argv builder"
```

---

## Task 3: JobManager — spawn, stream, and persist

**Files:**
- Create: `orchestrator/jobs.js`
- Create: `orchestrator/test/jobs.test.js`

This task covers the happy path: start a job, capture stdout/stderr to a log file and in-memory tail, fire subscriber events, persist `meta.json` on exit. Stop/kill is in Task 4.

- [ ] **Step 1: Write the failing test at `orchestrator/test/jobs.test.js`**

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { JobManager } from "../jobs.js";

async function freshRunsDir() {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "orchestrator-runs-"));
  return dir;
}

function waitForEvent(jm, type, predicate) {
  return new Promise(resolve => {
    const handler = ev => {
      if (ev.type === type && (!predicate || predicate(ev))) {
        jm.off(handler);
        resolve(ev);
      }
    };
    jm.on(handler);
  });
}

test("start writes meta.json and streams stdout lines", async () => {
  const runsDir = await freshRunsDir();
  const jm = new JobManager({ runsDir, projectRoot: process.cwd() });

  const job = await jm.start({
    flowId: "smoke",
    label: "smoke",
    argv: ["-c", "for i in range(3):\n    print('line', i)"],
    pythonBin: "python",
  });

  // Collect line events
  const lines = [];
  jm.on(ev => { if (ev.jobId === job.id && ev.type === "line") lines.push(ev.text); });

  const exit = await waitForEvent(jm, "status", ev => ev.jobId === job.id && ["succeeded","failed","killed"].includes(ev.status));
  assert.equal(exit.status, "succeeded");
  assert.equal(exit.exitCode, 0);

  const meta = JSON.parse(await fs.readFile(path.join(runsDir, job.id, "meta.json"), "utf8"));
  assert.equal(meta.status, "succeeded");
  assert.equal(meta.exitCode, 0);
  assert.ok(meta.startedAt);
  assert.ok(meta.endedAt);

  const log = await fs.readFile(path.join(runsDir, job.id, "output.log"), "utf8");
  assert.match(log, /line 0/);
  assert.match(log, /line 2/);
  assert.ok(lines.some(l => l.includes("line 0")));
});

test("list returns jobs sorted newest-first", async () => {
  const runsDir = await freshRunsDir();
  const jm = new JobManager({ runsDir, projectRoot: process.cwd() });

  const a = await jm.start({ flowId: "smoke", label: "a", argv: ["-c", "print('a')"], pythonBin: "python" });
  await waitForEvent(jm, "status", ev => ev.jobId === a.id);
  const b = await jm.start({ flowId: "smoke", label: "b", argv: ["-c", "print('b')"], pythonBin: "python" });
  await waitForEvent(jm, "status", ev => ev.jobId === b.id);

  const jobs = jm.list();
  assert.equal(jobs[0].id, b.id);
  assert.equal(jobs[1].id, a.id);
});

test("loadFromDisk marks running jobs as orphaned", async () => {
  const runsDir = await freshRunsDir();
  const id = "ORPHANED01";
  await fs.mkdir(path.join(runsDir, id), { recursive: true });
  await fs.writeFile(path.join(runsDir, id, "meta.json"), JSON.stringify({
    id, flowId: "x", label: "x", argv: [], command: "",
    startedAt: new Date().toISOString(), endedAt: null,
    status: "running", exitCode: null, pid: 999999,
  }));
  await fs.writeFile(path.join(runsDir, id, "output.log"), "");

  const jm = new JobManager({ runsDir, projectRoot: process.cwd() });
  await jm.loadFromDisk();
  const job = jm.get(id);
  assert.equal(job.status, "orphaned");

  const meta = JSON.parse(await fs.readFile(path.join(runsDir, id, "meta.json"), "utf8"));
  assert.equal(meta.status, "orphaned");
});

test("get returns rolling tail", async () => {
  const runsDir = await freshRunsDir();
  const jm = new JobManager({ runsDir, projectRoot: process.cwd() });
  const job = await jm.start({
    flowId: "smoke", label: "tail",
    argv: ["-c", "for i in range(10):\n    print(i)"],
    pythonBin: "python",
  });
  await waitForEvent(jm, "status", ev => ev.jobId === job.id);

  const detail = jm.get(job.id);
  assert.ok(detail.tail.length >= 10);
  assert.ok(detail.tail.some(l => l.text.includes("9")));
});
```

- [ ] **Step 2: Run tests to verify they fail**

```
npm test
```

Expected: fails — `jobs.js` not found. (Note: requires Python on PATH; the tests use `python -c …`.)

- [ ] **Step 3: Implement `orchestrator/jobs.js`**

```js
// jobs.js — JobManager: spawn, stream, persist, kill.
// Pure event emitter pattern (no DOM/Node EventEmitter dependency); subscribers
// register a callback via .on(fn) and receive { type, jobId, ... } events.

import { spawn } from "node:child_process";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import { ulid } from "ulid";

const TAIL_MAX = 500;
const TERMINAL = new Set(["succeeded", "failed", "killed", "orphaned"]);

export class JobManager {
  constructor({ runsDir, projectRoot }) {
    if (!runsDir) throw new Error("JobManager: runsDir required");
    if (!projectRoot) throw new Error("JobManager: projectRoot required");
    this.runsDir = runsDir;
    this.projectRoot = projectRoot;
    this.jobs = new Map();     // jobId -> in-memory state
    this.subscribers = new Set(); // (event) => void
    fs.mkdirSync(runsDir, { recursive: true });
  }

  on(fn) { this.subscribers.add(fn); }
  off(fn) { this.subscribers.delete(fn); }
  _emit(event) { for (const fn of this.subscribers) { try { fn(event); } catch {} } }

  list() {
    return [...this.jobs.values()]
      .map(j => this._publicMeta(j))
      .sort((a, b) => (a.startedAt < b.startedAt ? 1 : -1));
  }

  get(id) {
    const j = this.jobs.get(id);
    if (!j) return null;
    return { ...this._publicMeta(j), tail: j.tail.slice() };
  }

  _publicMeta(j) {
    return {
      id: j.id, flowId: j.flowId, label: j.label,
      argv: j.argv, command: j.command,
      startedAt: j.startedAt, endedAt: j.endedAt,
      status: j.status, exitCode: j.exitCode, pid: j.pid,
    };
  }

  async loadFromDisk() {
    let entries;
    try { entries = await fsp.readdir(this.runsDir); } catch { return; }
    for (const id of entries) {
      const metaPath = path.join(this.runsDir, id, "meta.json");
      let meta;
      try { meta = JSON.parse(await fsp.readFile(metaPath, "utf8")); }
      catch { continue; }
      // Recover non-terminal jobs as orphaned (server restart lost the child)
      if (!TERMINAL.has(meta.status)) {
        meta.status = "orphaned";
        meta.endedAt = meta.endedAt ?? new Date().toISOString();
        await fsp.writeFile(metaPath, JSON.stringify(meta, null, 2));
      }
      // Load a small tail from the on-disk log
      const tail = await this._readLogTail(id, TAIL_MAX);
      this.jobs.set(id, { ...meta, tail, child: null, logStream: null, stopAttempts: 0, killTimer: null });
    }
  }

  async _readLogTail(id, max) {
    const logPath = path.join(this.runsDir, id, "output.log");
    let content;
    try { content = await fsp.readFile(logPath, "utf8"); } catch { return []; }
    const lines = content.split(/\r?\n/).filter(Boolean);
    return lines.slice(-max).map(text => ({ text, stream: "log", ts: null }));
  }

  async start({ flowId, label, argv, pythonBin }) {
    if (!Array.isArray(argv) || argv.length === 0) throw new Error("start: argv required");
    if (!pythonBin) throw new Error("start: pythonBin required");

    const id = ulid();
    const dir = path.join(this.runsDir, id);
    await fsp.mkdir(dir, { recursive: true });

    const startedAt = new Date().toISOString();
    const command = [pythonBin, ...argv].map(a => /\s/.test(a) ? `"${a}"` : a).join(" ");
    const job = {
      id, flowId, label, argv, command,
      startedAt, endedAt: null,
      status: "running", exitCode: null, pid: null,
      tail: [], child: null, logStream: null, stopAttempts: 0, killTimer: null,
    };
    this.jobs.set(id, job);

    const logPath = path.join(dir, "output.log");
    job.logStream = fs.createWriteStream(logPath, { flags: "a" });

    const env = { ...process.env, PYTHONUNBUFFERED: "1" };
    const child = spawn(pythonBin, argv, {
      cwd: this.projectRoot,
      env,
      windowsHide: true,
      detached: process.platform !== "win32",
      shell: false,
    });
    job.child = child;
    job.pid = child.pid;

    await this._writeMeta(job);

    const writeLine = (text, stream) => {
      const ts = new Date().toISOString().slice(11, 23); // HH:MM:SS.mmm
      const tag = stream === "stderr" ? "ERR" : "OUT";
      const formatted = `[${ts}] ${tag} ${text}`;
      job.logStream.write(formatted + "\n");
      const line = { text, stream, ts };
      job.tail.push(line);
      if (job.tail.length > TAIL_MAX) job.tail.shift();
      this._emit({ type: "line", jobId: id, ...line });
    };

    const rlOut = readline.createInterface({ input: child.stdout });
    rlOut.on("line", l => writeLine(l, "stdout"));
    const rlErr = readline.createInterface({ input: child.stderr });
    rlErr.on("line", l => writeLine(l, "stderr"));

    child.on("error", err => writeLine(`spawn error: ${err.message}`, "stderr"));

    child.on("exit", async (code, signal) => {
      if (job.killTimer) { clearTimeout(job.killTimer); job.killTimer = null; }
      // Status precedence: if user requested stop, mark killed; else infer from exit code.
      let status;
      if (job.stopAttempts > 0) status = "killed";
      else if (code === 0) status = "succeeded";
      else status = "failed";
      job.status = status;
      job.exitCode = code;
      job.endedAt = new Date().toISOString();
      try { job.logStream.end(); } catch {}
      await this._writeMeta(job);
      this._emit({ type: "status", jobId: id, status, exitCode: code, endedAt: job.endedAt });
    });

    this._emit({ type: "status", jobId: id, status: "running", startedAt });
    return this._publicMeta(job);
  }

  async _writeMeta(job) {
    const metaPath = path.join(this.runsDir, job.id, "meta.json");
    const meta = this._publicMeta(job);
    await fsp.writeFile(metaPath, JSON.stringify(meta, null, 2));
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```
npm test
```

Expected: all jobs.test.js cases pass alongside flows.test.js. If Python isn't on PATH, set `PYTHON_BIN` env var pointing at the conda python before running tests.

- [ ] **Step 5: Commit**

```
git add orchestrator/jobs.js orchestrator/test/jobs.test.js
git commit -m "orchestrator: JobManager with spawn, streaming, persistence"
```

---

## Task 4: JobManager — graceful then hard force quit

**Files:**
- Modify: `orchestrator/jobs.js` (add `stop()` method)
- Create: `orchestrator/test/jobs.stop.test.js`

- [ ] **Step 1: Write the failing test at `orchestrator/test/jobs.stop.test.js`**

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { JobManager } from "../jobs.js";

const SLEEP_SCRIPT = "import time\nfor _ in range(120):\n    print('tick', flush=True)\n    time.sleep(0.5)\n";

function waitForStatus(jm, jobId, status) {
  return new Promise(resolve => {
    const handler = ev => {
      if (ev.jobId === jobId && ev.type === "status" && ev.status === status) {
        jm.off(handler); resolve(ev);
      }
    };
    jm.on(handler);
  });
}

test("stop escalates from graceful to hard after timeout", async () => {
  const runsDir = await fs.mkdtemp(path.join(os.tmpdir(), "orchestrator-runs-"));
  const jm = new JobManager({ runsDir, projectRoot: process.cwd(), stopGraceMs: 1500 });
  const job = await jm.start({
    flowId: "sleep", label: "sleep", argv: ["-c", SLEEP_SCRIPT], pythonBin: "python",
  });

  await new Promise(r => setTimeout(r, 500));
  await jm.stop(job.id);
  const ev = await waitForStatus(jm, job.id, "killed");
  assert.equal(ev.status, "killed");
});

test("second stop call hard-kills immediately", async () => {
  const runsDir = await fs.mkdtemp(path.join(os.tmpdir(), "orchestrator-runs-"));
  const jm = new JobManager({ runsDir, projectRoot: process.cwd(), stopGraceMs: 60000 });
  const job = await jm.start({
    flowId: "sleep", label: "sleep", argv: ["-c", SLEEP_SCRIPT], pythonBin: "python",
  });

  await new Promise(r => setTimeout(r, 500));
  await jm.stop(job.id);          // graceful
  await new Promise(r => setTimeout(r, 200));
  await jm.stop(job.id);          // force
  const ev = await waitForStatus(jm, job.id, "killed");
  assert.equal(ev.status, "killed");
});

test("stop is a no-op for terminal jobs", async () => {
  const runsDir = await fs.mkdtemp(path.join(os.tmpdir(), "orchestrator-runs-"));
  const jm = new JobManager({ runsDir, projectRoot: process.cwd() });
  const job = await jm.start({
    flowId: "quick", label: "quick", argv: ["-c", "print('x')"], pythonBin: "python",
  });
  await waitForStatus(jm, job.id, "succeeded");
  // should not throw or change state
  const result = await jm.stop(job.id);
  assert.equal(result.status, "succeeded");
});
```

- [ ] **Step 2: Run tests to verify they fail**

```
npm test
```

Expected: jobs.stop.test.js fails — `stop` method does not exist.

- [ ] **Step 3: Extend `orchestrator/jobs.js`**

In the constructor add the configurable grace timeout:

```js
  constructor({ runsDir, projectRoot, stopGraceMs }) {
    if (!runsDir) throw new Error("JobManager: runsDir required");
    if (!projectRoot) throw new Error("JobManager: projectRoot required");
    this.runsDir = runsDir;
    this.projectRoot = projectRoot;
    this.stopGraceMs = stopGraceMs ?? 5000;
    this.jobs = new Map();
    this.subscribers = new Set();
    fs.mkdirSync(runsDir, { recursive: true });
  }
```

Add the `stop()` method at the end of the class:

```js
  async stop(id) {
    const job = this.jobs.get(id);
    if (!job) return null;
    if (TERMINAL.has(job.status)) return this._publicMeta(job);

    job.stopAttempts++;
    const isFirst = job.stopAttempts === 1;
    const child = job.child;
    if (!child || child.exitCode !== null) return this._publicMeta(job);

    if (isFirst) {
      job.status = "stopping";
      this._emit({ type: "status", jobId: id, status: "stopping" });
      await this._writeMeta(job);

      if (process.platform === "win32") {
        // taskkill /T walks the process tree, no /F so handlers can run.
        spawn("taskkill", ["/PID", String(child.pid), "/T"], { windowsHide: true });
      } else {
        try { process.kill(-child.pid, "SIGINT"); } catch (e) {
          try { child.kill("SIGINT"); } catch {}
        }
      }
      // Auto-escalate after grace period
      job.killTimer = setTimeout(() => {
        this._hardKill(job);
      }, this.stopGraceMs);
    } else {
      // Second call: hard kill now
      this._hardKill(job);
    }
    return this._publicMeta(job);
  }

  _hardKill(job) {
    if (job.killTimer) { clearTimeout(job.killTimer); job.killTimer = null; }
    const child = job.child;
    if (!child || child.exitCode !== null) return;
    if (process.platform === "win32") {
      spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], { windowsHide: true });
    } else {
      try { process.kill(-child.pid, "SIGKILL"); } catch {
        try { child.kill("SIGKILL"); } catch {}
      }
    }
  }
```

Also update the `child.on("exit")` handler to ensure status precedence is correct when `stopAttempts > 0` (existing logic already handles this).

- [ ] **Step 4: Run tests to verify they pass**

```
npm test
```

Expected: all tests pass. The first stop test may take up to ~2s due to graceMs=1500.

- [ ] **Step 5: Commit**

```
git add orchestrator/jobs.js orchestrator/test/jobs.stop.test.js
git commit -m "orchestrator: two-stage force quit (SIGINT/taskkill then SIGKILL/taskkill /F)"
```

---

## Task 5: HTTP API routes

**Files:**
- Modify: `orchestrator/server.js`

- [ ] **Step 1: Replace `orchestrator/server.js` with the full HTTP wiring**

```js
import express from "express";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs/promises";
import { FLOWS, getFlow, buildArgv } from "./flows.js";
import { JobManager } from "./jobs.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const runsDir = path.join(__dirname, "runs");

const pythonBin = process.env.PYTHON_BIN ?? (process.platform === "win32" ? "python" : "python3");
const jm = new JobManager({ runsDir, projectRoot });
await jm.loadFromDisk();

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

app.get("/api/health", (_req, res) => res.json({ ok: true }));

app.get("/api/flows", (_req, res) => res.json({ flows: FLOWS }));

app.get("/api/jobs", (req, res) => {
  const all = jm.list();
  const filtered = req.query.status === "running"
    ? all.filter(j => j.status === "running" || j.status === "stopping")
    : all;
  res.json({ jobs: filtered });
});

app.get("/api/jobs/:id", (req, res) => {
  const j = jm.get(req.params.id);
  if (!j) return res.status(404).json({ error: "not found" });
  const tail = Math.max(0, Math.min(5000, Number(req.query.tail ?? 500)));
  res.json({ ...j, tail: j.tail.slice(-tail) });
});

app.get("/api/jobs/:id/log", async (req, res) => {
  const logPath = path.join(runsDir, req.params.id, "output.log");
  try {
    const content = await fs.readFile(logPath, "utf8");
    res.type("text/plain").send(content);
  } catch {
    res.status(404).send("not found");
  }
});

app.post("/api/jobs", async (req, res) => {
  const { flowId, args = {}, extraArgs = "" } = req.body ?? {};
  const flow = getFlow(flowId);
  if (!flow) return res.status(400).json({ error: `unknown flowId: ${flowId}` });
  let argv;
  try { argv = buildArgv(flow, args, extraArgs); }
  catch (e) { return res.status(400).json({ error: e.message }); }

  const scriptAbs = path.join(projectRoot, argv[0]);
  try { await fs.access(scriptAbs); }
  catch { return res.status(400).json({ error: `script not found: ${argv[0]}` }); }

  const job = await jm.start({ flowId, label: flow.label, argv, pythonBin });
  res.status(201).json({ job });
});

app.post("/api/jobs/:id/stop", async (req, res) => {
  const meta = await jm.stop(req.params.id);
  if (!meta) return res.status(404).json({ error: "not found" });
  res.json({ job: meta });
});

app.delete("/api/jobs/:id", async (req, res) => {
  const j = jm.get(req.params.id);
  if (!j) return res.status(404).json({ error: "not found" });
  if (!["succeeded", "failed", "killed", "orphaned"].includes(j.status)) {
    return res.status(409).json({ error: "job not terminal" });
  }
  await fs.rm(path.join(runsDir, req.params.id), { recursive: true, force: true });
  jm.jobs.delete(req.params.id);
  res.json({ ok: true });
});

const PORT = Number(process.env.PORT ?? 3100);
const HOST = process.env.HOST ?? "127.0.0.1";
export const server = app.listen(PORT, HOST, () => {
  console.log(`[orchestrator] listening on http://${HOST}:${PORT}`);
});

export { app, jm };
```

- [ ] **Step 2: Manual smoke-test**

In one terminal:

```
cd "C:\work\llm training\orchestrator"
npm start
```

In another:

```powershell
curl http://127.0.0.1:3100/api/health
curl http://127.0.0.1:3100/api/flows
curl -X POST http://127.0.0.1:3100/api/jobs -H "Content-Type: application/json" -d '{"flowId":"stage_4_eval","args":{"model":"models/teacher/gguf","limit":1}}'
curl http://127.0.0.1:3100/api/jobs
```

Expected:
- `/api/health` → `{"ok":true}`
- `/api/flows` → catalogue
- `POST /api/jobs` → 201 with job meta (will fail-spawn if no python+model, that's fine — status becomes `failed`, you should still get a job object)
- `/api/jobs` → contains that job

Stop the server.

- [ ] **Step 3: Commit**

```
git add orchestrator/server.js
git commit -m "orchestrator: HTTP API for flows, jobs, log download, stop, delete"
```

---

## Task 6: WebSocket log streaming

**Files:**
- Modify: `orchestrator/server.js`

- [ ] **Step 1: Add WebSocket subscription at the end of `orchestrator/server.js`, before the export block**

```js
import { WebSocketServer } from "ws";

const wss = new WebSocketServer({ server, path: "/ws/jobs" });

wss.on("connection", (ws, req) => {
  // path format: /ws/jobs/<id>
  const url = new URL(req.url, "http://localhost");
  const parts = url.pathname.split("/").filter(Boolean);
  const jobId = parts[2];
  if (!jobId) { ws.close(1008, "jobId required"); return; }

  const job = jm.get(jobId);
  if (!job) { ws.close(1008, "unknown job"); return; }

  // Send the tail immediately so the UI can catch up
  ws.send(JSON.stringify({ type: "snapshot", job }));

  const handler = ev => {
    if (ev.jobId !== jobId) return;
    if (ws.readyState !== ws.OPEN) return;
    try { ws.send(JSON.stringify(ev)); } catch {}
  };
  jm.on(handler);

  ws.on("close", () => jm.off(handler));
  ws.on("error", () => jm.off(handler));
});
```

The `WebSocketServer` constructor must accept a custom path matcher. Since we want to match `/ws/jobs/<id>` (variable ID), replace the `path` option with manual upgrade handling:

Replace the `wss` block with:

```js
import { WebSocketServer } from "ws";

const wss = new WebSocketServer({ noServer: true });

server.on("upgrade", (req, socket, head) => {
  const url = new URL(req.url, "http://localhost");
  const m = url.pathname.match(/^\/ws\/jobs\/([^/]+)\/?$/);
  if (!m) { socket.destroy(); return; }
  const jobId = m[1];
  wss.handleUpgrade(req, socket, head, ws => {
    const job = jm.get(jobId);
    if (!job) { ws.close(1008, "unknown job"); return; }
    ws.send(JSON.stringify({ type: "snapshot", job }));

    const handler = ev => {
      if (ev.jobId !== jobId) return;
      if (ws.readyState !== ws.OPEN) return;
      try { ws.send(JSON.stringify(ev)); } catch {}
    };
    jm.on(handler);
    ws.on("close", () => jm.off(handler));
    ws.on("error", () => jm.off(handler));
  });
});
```

- [ ] **Step 2: Smoke-test with `wscat` or curl**

If you have `wscat` (`npm i -g wscat`):

```
wscat -c ws://127.0.0.1:3100/ws/jobs/<jobIdFromEarlier>
```

Expected: snapshot frame appears immediately. If you start a fresh fast-running job, you should also see line + status frames.

If wscat isn't available, this gets validated by the frontend in later tasks.

- [ ] **Step 3: Commit**

```
git add orchestrator/server.js orchestrator/package.json orchestrator/package-lock.json
git commit -m "orchestrator: WebSocket /ws/jobs/<id> streaming with snapshot + live events"
```

---

## Task 7: Frontend skeleton (HTML + dark CSS theme)

**Files:**
- Create: `orchestrator/public/index.html`
- Create: `orchestrator/public/style.css`

- [ ] **Step 1: Create `orchestrator/public/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Pipeline Orchestrator</title>
  <link rel="stylesheet" href="/style.css" />
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <span class="logo">🜲</span>
      <span class="title">Pipeline Orchestrator</span>
    </div>
    <div class="topbar-stats">
      <span class="pill pill-running" id="running-count">● 0 running</span>
    </div>
  </header>

  <main class="layout">
    <aside class="col col-flows">
      <h2 class="col-title">Flows</h2>
      <ul id="flow-list" class="flow-list"></ul>
    </aside>

    <section class="col col-main">
      <div id="flow-detail" class="flow-detail hidden">
        <header class="flow-header">
          <h2 id="flow-label"></h2>
          <p id="flow-description" class="muted"></p>
        </header>
        <form id="flow-form" class="args-form"></form>
        <div class="command-preview">
          <span class="muted">$</span>
          <code id="command-preview-text"></code>
        </div>
        <div class="form-actions">
          <button type="button" id="start-btn" class="btn btn-primary">Start</button>
        </div>
      </div>

      <div id="job-view" class="job-view hidden">
        <header class="job-header">
          <div class="job-title">
            <h2 id="job-label"></h2>
            <span id="job-status" class="status-chip"></span>
          </div>
          <div class="job-actions">
            <button type="button" id="stop-btn" class="btn btn-stop hidden">Stop</button>
            <a id="download-log" class="btn btn-ghost" href="#" download>Download log</a>
          </div>
        </header>
        <div class="job-meta">
          <span><strong>Command:</strong> <code id="job-command"></code></span>
          <span><strong>Started:</strong> <span id="job-started"></span></span>
          <span><strong>Elapsed:</strong> <span id="job-elapsed"></span></span>
        </div>
        <div class="console-wrap">
          <pre id="console" class="console" aria-live="polite"></pre>
          <button type="button" id="jump-live" class="jump-live hidden">Jump to live</button>
        </div>
      </div>

      <div id="empty-state" class="empty-state">
        Pick a flow on the left to start, or a past run on the right.
      </div>
    </section>

    <aside class="col col-jobs">
      <h2 class="col-title">Jobs</h2>
      <ul id="job-list" class="job-list"></ul>
    </aside>
  </main>

  <script src="/app.js" type="module"></script>
</body>
</html>
```

- [ ] **Step 2: Create `orchestrator/public/style.css`**

```css
:root {
  --bg-0: #0b1120;
  --bg-1: #0f172a;
  --bg-2: #111c33;
  --panel: rgba(255,255,255,0.04);
  --panel-strong: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.10);
  --text: #e2e8f0;
  --muted: #94a3b8;
  --accent: #22d3ee;
  --ok: #34d399;
  --warn: #fbbf24;
  --bad: #fb7185;
  --console-bg: #020617;
  --font-ui: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  font-family: var(--font-ui);
  color: var(--text);
  background:
    radial-gradient(1200px 600px at 80% -10%, rgba(34,211,238,0.08), transparent 60%),
    linear-gradient(180deg, var(--bg-0), var(--bg-1));
  min-height: 100%;
  -webkit-font-smoothing: antialiased;
}

.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 24px;
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(12px);
  position: sticky; top: 0; z-index: 5;
  background: rgba(11,17,32,0.7);
}
.brand { display: flex; align-items: center; gap: 10px; font-weight: 600; letter-spacing: 0.2px; }
.logo { font-size: 20px; }
.title { font-size: 15px; }
.pill {
  font-size: 12px; padding: 4px 10px; border-radius: 999px;
  background: var(--panel); border: 1px solid var(--border);
}
.pill-running { color: var(--accent); }

.layout {
  display: grid;
  grid-template-columns: 280px 1fr 320px;
  gap: 16px;
  padding: 16px;
  min-height: calc(100vh - 56px);
}
.col {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  backdrop-filter: blur(12px);
}
.col-title { margin: 0 0 12px; font-size: 12px; font-weight: 600; letter-spacing: 0.12em; color: var(--muted); text-transform: uppercase; }

.flow-list, .job-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 6px; }
.flow-item, .job-item {
  padding: 10px 12px;
  border-radius: 8px;
  background: rgba(255,255,255,0.02);
  border: 1px solid transparent;
  cursor: pointer;
  transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
}
.flow-item:hover, .job-item:hover { background: var(--panel-strong); }
.flow-item.active, .job-item.active { border-color: var(--accent); background: rgba(34,211,238,0.08); }
.flow-item .label, .job-item .label { font-weight: 500; }
.flow-item .desc { color: var(--muted); font-size: 12px; margin-top: 2px; }
.job-item .row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.job-item .when { color: var(--muted); font-size: 11px; }

.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.status-dot.running { background: var(--accent); box-shadow: 0 0 0 0 rgba(34,211,238,0.6); animation: pulse 1.4s infinite; }
.status-dot.stopping { background: var(--warn); }
.status-dot.succeeded { background: var(--ok); }
.status-dot.failed, .status-dot.killed, .status-dot.orphaned { background: var(--bad); }
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(34,211,238,0.5); }
  70%  { box-shadow: 0 0 0 8px rgba(34,211,238,0); }
  100% { box-shadow: 0 0 0 0 rgba(34,211,238,0); }
}

.flow-detail, .job-view { display: flex; flex-direction: column; gap: 14px; }
.flow-header h2, .job-header h2 { margin: 0; font-size: 18px; font-weight: 600; }
.muted { color: var(--muted); margin: 4px 0 0; font-size: 13px; }

.args-form { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
.field { display: flex; flex-direction: column; gap: 4px; }
.field label { font-size: 12px; color: var(--muted); }
.field input[type="text"], .field input[type="number"], .field select, .field textarea {
  background: var(--bg-2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px;
  font: 13px/1.4 var(--font-mono);
  width: 100%;
}
.field input[type="checkbox"] { width: 16px; height: 16px; }
.field.flag { flex-direction: row; align-items: center; gap: 8px; }
.field-extra { grid-column: 1 / -1; }
.field-extra textarea { min-height: 60px; }

.command-preview {
  font-family: var(--font-mono); font-size: 12px;
  background: var(--console-bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 12px; color: var(--text);
  overflow-x: auto; white-space: pre;
}

.btn {
  display: inline-flex; align-items: center; justify-content: center;
  padding: 8px 14px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--panel); color: var(--text); font: 500 13px var(--font-ui);
  cursor: pointer; transition: background 120ms ease, transform 120ms ease, border-color 120ms ease;
  text-decoration: none;
}
.btn:hover { background: var(--panel-strong); }
.btn-primary { background: linear-gradient(180deg, #0891b2, #0e7490); border-color: #0891b2; color: white; }
.btn-primary:hover { filter: brightness(1.1); }
.btn-stop { background: rgba(251,113,133,0.15); border-color: var(--bad); color: var(--bad); }
.btn-stop.force { background: var(--bad); color: white; }

.status-chip { display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border); }
.status-chip.running { color: var(--accent); border-color: rgba(34,211,238,0.4); }
.status-chip.stopping { color: var(--warn); border-color: rgba(251,191,36,0.4); }
.status-chip.succeeded { color: var(--ok); border-color: rgba(52,211,153,0.4); }
.status-chip.failed, .status-chip.killed, .status-chip.orphaned { color: var(--bad); border-color: rgba(251,113,133,0.4); }

.job-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.job-title { display: flex; align-items: center; gap: 12px; }
.job-actions { display: flex; align-items: center; gap: 8px; }
.job-meta { display: flex; gap: 18px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }
.job-meta code { font-family: var(--font-mono); color: var(--text); }

.console-wrap { position: relative; }
.console {
  background: var(--console-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  font: 12.5px/1.55 var(--font-mono);
  padding: 14px 16px;
  height: 60vh;
  overflow: auto;
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
}
.console .out { color: #cbd5e1; }
.console .err { color: #fda4af; }
.console .meta { color: var(--muted); }

.jump-live {
  position: absolute; right: 16px; bottom: 16px;
  background: var(--accent); color: #052e35; border: none; border-radius: 999px;
  padding: 6px 12px; font-size: 12px; font-weight: 600; cursor: pointer;
  box-shadow: 0 4px 12px rgba(34,211,238,0.3);
}

.hidden { display: none !important; }
.empty-state { color: var(--muted); padding: 40px; text-align: center; font-size: 14px; }
```

- [ ] **Step 3: Smoke-test in browser**

Start the server (`npm start`) and open `http://127.0.0.1:3100`. You should see the dark layout with empty columns and the "Pick a flow…" message. No JS errors in console.

- [ ] **Step 4: Commit**

```
git add orchestrator/public/index.html orchestrator/public/style.css
git commit -m "orchestrator: frontend skeleton + dark theme"
```

---

## Task 8: Frontend app.js — flow catalogue and args form

**Files:**
- Create: `orchestrator/public/app.js`

- [ ] **Step 1: Create `orchestrator/public/app.js`**

```js
// Single-page state store + render. No framework.

const state = {
  flows: [],
  jobs: new Map(),     // jobId -> meta + tail
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
  const [{ flows }, { jobs }] = await Promise.all([
    api("GET", "/api/flows"),
    api("GET", "/api/jobs"),
  ]);
  state.flows = flows;
  for (const j of jobs) state.jobs.set(j.id, { ...j, tail: [] });

  renderFlows();
  renderJobList();
  refreshRunningCount();

  // Subscribe to all non-terminal jobs so we keep them up to date
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
  for (const arg of flow.args) form.appendChild(renderField(arg));

  // Extra args textarea
  const extra = document.createElement("div");
  extra.className = "field field-extra";
  extra.innerHTML = `
    <label for="arg-extra">Additional args (whitespace-split, appended verbatim)</label>
    <textarea id="arg-extra" placeholder="--flag value …"></textarea>
  `;
  form.appendChild(extra);

  form.addEventListener("input", updateCommandPreview);
  updateCommandPreview();
}

function renderField(arg) {
  const wrap = document.createElement("div");
  const safeName = `arg-${arg.name}`;
  if (arg.type === "flag") {
    wrap.className = "field flag";
    wrap.innerHTML = `
      <input type="checkbox" id="${safeName}" data-arg="${arg.name}" data-type="flag" ${arg.default ? "checked" : ""}/>
      <label for="${safeName}">--${arg.name}</label>
    `;
  } else if (arg.type === "choice") {
    wrap.className = "field";
    const opts = arg.choices.map(c => `<option value="${c}" ${c === arg.default ? "selected" : ""}>${c}</option>`).join("");
    wrap.innerHTML = `
      <label for="${safeName}">--${arg.name}</label>
      <select id="${safeName}" data-arg="${arg.name}" data-type="choice">${opts}</select>
    `;
  } else if (arg.type === "positional") {
    wrap.className = "field field-extra";
    wrap.innerHTML = `
      <label for="${safeName}">${arg.name} (positional${arg.required ? ", required" : ""})</label>
      <input type="text" id="${safeName}" data-arg="${arg.name}" data-type="positional" value="${arg.default ?? ""}"/>
    `;
  } else {
    const inputType = (arg.type === "int" || arg.type === "float") ? "number" : "text";
    const step = arg.type === "float" ? "any" : "1";
    wrap.className = "field";
    wrap.innerHTML = `
      <label for="${safeName}">--${arg.name}${arg.required ? " *" : ""}</label>
      <input type="${inputType}" step="${step}" id="${safeName}" data-arg="${arg.name}" data-type="${arg.type}" value="${arg.default ?? ""}"/>
    `;
  }
  return wrap;
}

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
  // Mirror server's buildArgv logic for preview.
  const parts = [flow.script];
  for (const spec of flow.args) {
    const v = args[spec.name];
    if (spec.type === "flag") { if (v === true) parts.push(`--${spec.name}`); continue; }
    if (spec.type === "positional") continue;
    if (v === "" || v === null || v === undefined) continue;
    parts.push(`--${spec.name}`, String(v));
  }
  if (extraArgs && extraArgs.trim()) parts.push(...extraArgs.trim().split(/\s+/));
  for (const spec of flow.args) {
    if (spec.type === "positional" && args[spec.name]) parts.push(String(args[spec.name]));
  }
  $("#command-preview-text").textContent = "python " + parts.map(p => /\s/.test(p) ? `"${p}"` : p).join(" ");
}

$("#start-btn").addEventListener("click", async () => {
  const flow = state.flows.find(f => f.id === state.selectedFlowId);
  if (!flow) return;
  const { args, extraArgs } = collectFormValues();
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

// ---------- Job rendering & WS (filled in next task) ----------

window.addEventListener("DOMContentLoaded", init);

// Exposed for the next task's additions:
window.__app = { state, $, openSocket: null, selectJob: null, renderJobList: null, refreshRunningCount: null };
function openSocket() {}      // overridden in task 9
function selectJob() {}       // overridden in task 9
function renderJobList() {}   // overridden in task 9
function refreshRunningCount() {} // overridden in task 9
```

- [ ] **Step 2: Smoke-test in browser**

Reload `http://127.0.0.1:3100`. The Flows column populates with all 9 flows. Clicking one shows the args form and a live command preview that updates as you edit fields. The "Start" button POSTs (you can check the network tab; the job goes into limbo because the rendering helpers are stubbed — that's expected).

- [ ] **Step 3: Commit**

```
git add orchestrator/public/app.js
git commit -m "orchestrator: frontend flow catalogue + args form + command preview"
```

---

## Task 9: Frontend job list, console streaming, and force-quit button

**Files:**
- Modify: `orchestrator/public/app.js`

- [ ] **Step 1: Replace the four stub functions at the bottom of `orchestrator/public/app.js` with real implementations**

Delete the four stubs (`openSocket`, `selectJob`, `renderJobList`, `refreshRunningCount`) and the `window.__app` line at the bottom, then append:

```js
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
        <span class="when">${when}</span>
      </div>
    `;
    li.querySelector(".label").textContent = j.label;
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
      // reconnect after a delay
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
  // also refresh the "Xs ago" labels in the right column
  renderJobList();
}, 1000);
```

- [ ] **Step 2: End-to-end smoke test in the browser**

Restart server, open `http://127.0.0.1:3100`.

1. Click **Stage 4 — Evaluate**, set `--limit` to `1`, click **Start**. A new job appears on the right with a pulsing cyan dot. Console streams output (or shows the spawn failure if the model isn't present).
2. While running, click **Stop**. The chip flips to amber `stopping` and the button becomes a red `Force kill`. Within ~5s (or when you click it again), the job ends as `killed`.
3. Restart the server. The job is still present in the right column (history persisted). Clicking it loads the on-disk log into the console.

If any step fails, debug in this task; do not move on.

- [ ] **Step 3: Commit**

```
git add orchestrator/public/app.js
git commit -m "orchestrator: live console, job list, two-stage stop button, autoscroll"
```

---

## Task 10: Polish — DELETE button, GPU-contention warning, README

**Files:**
- Modify: `orchestrator/public/app.js`
- Modify: `orchestrator/public/index.html`
- Modify: `orchestrator/public/style.css`
- Create: `orchestrator/README.md`

- [ ] **Step 1: Add a Delete button for terminal jobs**

In `orchestrator/public/index.html`, inside `.job-actions` (next to `Download log`), add:

```html
<button type="button" id="delete-btn" class="btn btn-ghost hidden">Delete</button>
```

In `orchestrator/public/app.js`, inside `renderJobDetail`, after the stop-button block, add:

```js
  const delBtn = document.getElementById("delete-btn");
  const terminal = ["succeeded", "failed", "killed", "orphaned"].includes(job.status);
  delBtn.classList.toggle("hidden", !terminal);
```

And add the handler near `#stop-btn`'s:

```js
document.getElementById("delete-btn").addEventListener("click", async () => {
  const id = state.selectedJobId;
  if (!id) return;
  if (!confirm("Delete this run and its log?")) return;
  await api("DELETE", `/api/jobs/${id}`);
  state.jobs.delete(id);
  state.selectedJobId = null;
  $("#job-view").classList.add("hidden");
  $("#empty-state").classList.remove("hidden");
  renderJobList();
});
```

- [ ] **Step 2: GPU-contention soft warning**

The training/eval flows are GPU-heavy. In `app.js`, before posting the start request inside the `#start-btn` handler, add:

```js
  const gpuFlows = new Set(["stage_3_train_teacher", "stage_4_eval", "stage_5_legacy", "stage_6_train_student", "predict_one"]);
  const alreadyRunning = [...state.jobs.values()].some(j =>
    gpuFlows.has(j.flowId) && (j.status === "running" || j.status === "stopping")
  );
  if (gpuFlows.has(flow.id) && alreadyRunning) {
    if (!confirm("Another GPU-using job is already running. Starting this one may OOM. Continue?")) return;
  }
```

- [ ] **Step 3: Add `orchestrator/README.md`**

```markdown
# Pipeline Orchestrator UI

A standalone web UI to launch the project's pipeline scripts, watch their
console output live, and force-quit a run if needed. Runs alongside the
existing `viewer/` server.

## Run

```
cd orchestrator
npm install
npm start
# → http://127.0.0.1:3100
```

Environment:
- `PORT` — bind port (default `3100`).
- `HOST` — bind host (default `127.0.0.1`).
- `PYTHON_BIN` — Python interpreter to spawn (default `python` on Windows,
  `python3` elsewhere). Same convention as `viewer/`.

## Flows

The Flows column lists curated entries pulled from `USE.md`: stages 1-6
(legacy and multi-provider Stage 5 are separate entries), plus `predict_one`
and `export_gguf`. Editing the catalogue means editing `flows.js`.

Each flow has a free-form **Additional args** textarea — anything you type
there is whitespace-split and appended verbatim to the argv.

## Force-quit semantics

- First click — `Stop` — sends `SIGINT` (POSIX) or `taskkill /T` (Windows)
  to the process group. The chip flips to **stopping**.
- Second click within ~5s — `Force kill` — sends `SIGKILL` / `taskkill /T /F`.
  After 5s the server auto-escalates.

## Persistence

Every run gets a directory under `orchestrator/runs/<jobId>/`:
- `meta.json` — invocation metadata + final status + exit code.
- `output.log` — full stdout+stderr, each line prefixed by `[HH:MM:SS.mmm] OUT|ERR`.

Run history survives server restarts. Non-terminal jobs from a previous
process are marked `orphaned` because the PID can't be re-attached.
```

- [ ] **Step 4: Run tests + manual sanity check**

```
cd "C:\work\llm training\orchestrator"
npm test
```

All tests should still pass.

Manual: spin up server, repeat the Task 9 smoke test, additionally verify the Delete button removes the runs/<id>/ directory and that starting a second GPU flow while one is running triggers the confirm prompt.

- [ ] **Step 5: Commit**

```
git add orchestrator/public/index.html orchestrator/public/style.css orchestrator/public/app.js orchestrator/README.md
git commit -m "orchestrator: delete button, GPU-contention warning, README"
```

---

## Self-review pass

- **Spec coverage:**
  - Standalone `orchestrator/` app on port 3100 — Task 1.
  - Curated flow catalogue with editable args + extra-args escape hatch — Task 2 + Task 8.
  - Spawn + stream + persist — Task 3.
  - Two-stage force quit (SIGINT/taskkill, then SIGKILL/taskkill /F after 5s or second press) — Task 4.
  - HTTP API (`/api/flows`, `/api/jobs`, `/api/jobs/:id`, `/api/jobs/:id/log`, POST job, POST stop, DELETE) — Task 5 + Task 10.
  - WebSocket `/ws/jobs/<id>` with snapshot + line + status frames — Task 6.
  - Three-column dark UI with glass-morphism cards — Task 7.
  - Live console with sticky autoscroll + jump-to-live + DOM cap — Task 9.
  - Job list with status dot + pulsing animation, two-stage stop button — Task 9.
  - Orphan recovery at boot — Task 3 (`loadFromDisk`).
  - GPU contention soft warning — Task 10.
  - Run history persistence (`runs/<jobId>/`) — Task 3.
  - Free-form "Additional args" textarea — Task 8.
- **Placeholder scan:** No `TBD` / "implement later" / "similar to" markers. Every code step has full code.
- **Type consistency:** `buildArgv` signature `(flow, values, extraArgsString)` matches across `flows.js`, tests, server, and frontend preview. Event shapes `{type, jobId, ...}` consistent across `JobManager` emit, WebSocket frames, and frontend handlers. Status set `{"running","stopping","succeeded","failed","killed","orphaned"}` consistent across spec, server, frontend chip/dot CSS.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-18-pipeline-orchestrator-ui.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
