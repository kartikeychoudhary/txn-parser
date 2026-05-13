// Stage 2: dataset viewer backend.
// Serves the static frontend and exposes a small JSON API over
// data/clean/{train,eval}.jsonl plus a flag log at data/flags.json.

import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const DATA_DIR = path.join(REPO_ROOT, 'data', 'clean');
const FLAGS_FILE = path.join(REPO_ROOT, 'data', 'flags.json');
const MODELS_DIR = path.join(REPO_ROOT, 'models');
const WORKER_SCRIPT = path.join(__dirname, 'inference_worker.py');
const PYTHON_BIN = process.env.PYTHON_BIN || (process.platform === 'win32' ? 'python' : 'python3');
const PUBLIC_DIR = path.join(__dirname, 'public');
const PORT = Number(process.env.PORT) || 3000;
const INFER_TIMEOUT_MS = Number(process.env.INFER_TIMEOUT_MS) || 120_000;

const fileCache = new Map(); // absPath -> { mtimeMs, records }

function resolveDatasetPath(name) {
  if (name !== 'train' && name !== 'eval') return null;
  return path.join(DATA_DIR, `${name}.jsonl`);
}

function readJsonl(filePath) {
  const stat = fs.statSync(filePath);
  const cached = fileCache.get(filePath);
  if (cached && cached.mtimeMs === stat.mtimeMs) return cached.records;
  const text = fs.readFileSync(filePath, 'utf8');
  const records = [];
  let badLines = 0;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    try {
      records.push(JSON.parse(line));
    } catch {
      badLines += 1;
    }
  }
  if (badLines > 0) {
    console.warn(`[viewer] ${path.basename(filePath)}: skipped ${badLines} unparseable lines`);
  }
  fileCache.set(filePath, { mtimeMs: stat.mtimeMs, records });
  return records;
}

function readFlags() {
  try {
    const text = fs.readFileSync(FLAGS_FILE, 'utf8');
    const data = JSON.parse(text);
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

function writeFlags(flags) {
  fs.mkdirSync(path.dirname(FLAGS_FILE), { recursive: true });
  fs.writeFileSync(FLAGS_FILE, JSON.stringify(flags, null, 2), 'utf8');
}

const app = express();
app.use(express.json({ limit: '64kb' }));
app.use(express.static(PUBLIC_DIR));

app.get('/api/examples', (req, res) => {
  const file = String(req.query.file || '');
  const page = Math.max(1, parseInt(req.query.page, 10) || 1);
  const pageSize = Math.min(100, Math.max(1, parseInt(req.query.pageSize, 10) || 20));
  const q = String(req.query.q || '').trim().toLowerCase();

  const filePath = resolveDatasetPath(file);
  if (!filePath) {
    return res.status(400).json({ error: "file must be 'train' or 'eval'" });
  }
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({
      error: `${path.relative(REPO_ROOT, filePath)} not found — run Stage 1 first.`,
    });
  }

  const records = readJsonl(filePath);
  const flags = readFlags().filter(f => f.file === file);
  const flagIndexSet = new Set(flags.map(f => f.index));

  const tagged = records.map((r, i) => ({
    index: i,
    input: r.input,
    output: r.output,
    source: r._source ?? null,
    flagged: flagIndexSet.has(i),
  }));
  const filtered = q ? tagged.filter(r => r.input.toLowerCase().includes(q)) : tagged;

  const total = filtered.length;
  const start = (page - 1) * pageSize;
  const items = filtered.slice(start, start + pageSize);

  res.json({
    file,
    page,
    pageSize,
    pageCount: Math.max(1, Math.ceil(total / pageSize)),
    total,
    totalFull: records.length,
    flaggedCount: flags.length,
    items,
  });
});

app.get('/api/stats', (req, res) => {
  const file = String(req.query.file || '');
  const filePath = resolveDatasetPath(file);
  if (!filePath) {
    return res.status(400).json({ error: "file must be 'train' or 'eval'" });
  }
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({
      error: `${path.relative(REPO_ROOT, filePath)} not found — run Stage 1 first.`,
    });
  }

  const records = readJsonl(filePath);
  const categories = {};
  const types = {};
  const txnCounts = {};
  let totalTxns = 0;
  for (const r of records) {
    const txns = r.output?.transactions ?? [];
    txnCounts[txns.length] = (txnCounts[txns.length] || 0) + 1;
    for (const t of txns) {
      categories[t.category] = (categories[t.category] || 0) + 1;
      types[t.type] = (types[t.type] || 0) + 1;
      totalTxns += 1;
    }
  }
  res.json({ file, total: records.length, totalTxns, categories, types, txnCounts });
});

// ---------------------------------------------------------------------------
// Stage 7: inference workers
// ---------------------------------------------------------------------------

class InferenceWorker {
  constructor(name, ggufPath) {
    this.name = name;
    this.ggufPath = ggufPath;
    this.ready = false;
    this.pending = new Map(); // requestId -> { resolve, reject, timer }
    this.queue = [];          // requests buffered until 'ready'
    this.proc = null;
    this.start();
  }

  start() {
    const args = [WORKER_SCRIPT, '--model', this.ggufPath];
    if (process.env.LLAMA_N_GPU_LAYERS !== undefined) {
      args.push('--n-gpu-layers', String(process.env.LLAMA_N_GPU_LAYERS));
    }
    console.log(`[worker:${this.name}] spawning ${PYTHON_BIN} ${args.join(' ')}`);
    this.proc = spawn(PYTHON_BIN, args, { stdio: ['pipe', 'pipe', 'pipe'] });

    const rl = readline.createInterface({ input: this.proc.stdout });
    rl.on('line', (line) => this.handleLine(line));

    this.proc.stderr.on('data', (chunk) => {
      process.stderr.write(`[worker:${this.name}] ${chunk}`);
    });

    this.proc.on('error', (err) => {
      console.error(`[worker:${this.name}] spawn error: ${err.message}`);
      this.rejectAll(err);
    });

    this.proc.on('exit', (code, signal) => {
      console.log(`[worker:${this.name}] exited (code=${code}, signal=${signal})`);
      this.ready = false;
      this.rejectAll(new Error(`worker ${this.name} exited`));
      // Don't auto-restart — caller can restart the server.
    });
  }

  rejectAll(err) {
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer);
      reject(err);
    }
    this.pending.clear();
    for (const { reject, timer } of this.queue) {
      clearTimeout(timer);
      reject(err);
    }
    this.queue = [];
  }

  handleLine(line) {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      console.warn(`[worker:${this.name}] non-JSON stdout: ${line}`);
      return;
    }
    if (msg.ready) {
      this.ready = true;
      console.log(`[worker:${this.name}] ready (${msg.model})`);
      while (this.queue.length) {
        const pending = this.queue.shift();
        this.pending.set(pending.id, pending);
        this.write(pending.req);
      }
      return;
    }
    if (msg.id && this.pending.has(msg.id)) {
      const { resolve, timer } = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      clearTimeout(timer);
      resolve(msg);
    }
  }

  write(req) {
    if (this.proc && this.proc.stdin.writable) {
      this.proc.stdin.write(JSON.stringify(req) + '\n');
    }
  }

  infer(input, { maxTokens = 512 } = {}) {
    return new Promise((resolve, reject) => {
      const id = randomUUID();
      const req = { id, input, max_tokens: maxTokens };
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`inference timed out after ${INFER_TIMEOUT_MS}ms`));
        }
      }, INFER_TIMEOUT_MS);
      const pending = { id, req, resolve, reject, timer };
      if (this.ready) {
        this.pending.set(id, pending);
        this.write(req);
      } else {
        this.queue.push(pending);
      }
    });
  }

  stop() {
    if (this.proc) {
      try { this.proc.stdin.end(); } catch { /* ignore */ }
      this.proc.kill();
    }
  }

  status() {
    return { name: this.name, gguf: path.basename(this.ggufPath), ready: this.ready };
  }
}

const workers = new Map();

function tryStartWorker(name, dir) {
  if (!fs.existsSync(dir)) {
    console.log(`[worker:${name}] no model dir at ${dir} — skipping`);
    return;
  }
  const files = fs.readdirSync(dir)
    .filter((f) => f.toLowerCase().endsWith('.gguf'))
    .filter((f) => !f.toLowerCase().includes('mmproj'));  // skip vision projector sidecars
  if (files.length === 0) {
    console.log(`[worker:${name}] no .gguf in ${dir} — skipping`);
    return;
  }
  const gguf = path.join(dir, files[0]);
  workers.set(name, new InferenceWorker(name, gguf));
}

tryStartWorker('teacher', path.join(MODELS_DIR, 'teacher', 'gguf'));
tryStartWorker('student', path.join(MODELS_DIR, 'student', 'gguf'));

app.get('/api/models', (req, res) => {
  const out = {};
  for (const [name, w] of workers) out[name] = w.status();
  res.json({ models: out });
});

app.post('/api/infer', async (req, res) => {
  const { model, input, max_tokens } = req.body || {};
  if (typeof input !== 'string' || !input.trim()) {
    return res.status(400).json({ error: 'input must be a non-empty string' });
  }
  if (!['teacher', 'student', 'both'].includes(model)) {
    return res.status(400).json({ error: "model must be 'teacher', 'student', or 'both'" });
  }
  const targets = model === 'both' ? ['teacher', 'student'] : [model];
  const result = {};
  await Promise.all(targets.map(async (m) => {
    const worker = workers.get(m);
    if (!worker) {
      result[m] = { error: `${m} model not loaded — train it via Stage 3 (or 6) and restart the server` };
      return;
    }
    try {
      const r = await worker.infer(input, { maxTokens: Number(max_tokens) || 512 });
      result[m] = {
        output: r.output,
        latency_ms: r.latency_ms,
        tokens: r.tokens,
        error: r.error || null,
        gguf: path.basename(worker.ggufPath),
      };
    } catch (e) {
      result[m] = { error: e.message };
    }
  }));
  res.json(result);
});

function shutdown() {
  console.log('shutting down workers…');
  for (const w of workers.values()) w.stop();
  process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// ---------------------------------------------------------------------------
// Original Stage 2 routes
// ---------------------------------------------------------------------------

app.post('/api/flag', (req, res) => {
  const { file, index, reason } = req.body || {};
  if (file !== 'train' && file !== 'eval') {
    return res.status(400).json({ error: "file must be 'train' or 'eval'" });
  }
  if (!Number.isInteger(index) || index < 0) {
    return res.status(400).json({ error: 'index must be a non-negative integer' });
  }
  const flags = readFlags();
  flags.push({
    file,
    index,
    reason: typeof reason === 'string' ? reason.slice(0, 500) : '',
    timestamp: new Date().toISOString(),
  });
  writeFlags(flags);
  res.json({ ok: true, totalFlags: flags.length });
});

app.listen(PORT, () => {
  console.log(`[viewer] running at http://localhost:${PORT}`);
  console.log(`[viewer] dataset dir: ${DATA_DIR}`);
});
