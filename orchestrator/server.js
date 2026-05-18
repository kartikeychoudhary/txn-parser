import express from "express";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs/promises";
import { FLOWS, getFlow, buildArgv } from "./flows.js";
import { JobManager } from "./jobs.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const runsDir = path.join(__dirname, "runs");

// Resolve how to spawn Python so scripts always run in the project's conda env.
// Precedence: explicit PYTHON_BIN > already-activated matching env > `conda run`.
const condaEnv = process.env.CONDA_ENV ?? "llm-training";
let pythonBin;
let argvPrefix = [];
if (process.env.PYTHON_BIN) {
  pythonBin = process.env.PYTHON_BIN;
} else if (process.env.CONDA_DEFAULT_ENV === condaEnv) {
  pythonBin = process.platform === "win32" ? "python" : "python3";
} else {
  pythonBin = "conda";
  argvPrefix = ["run", "--no-capture-output", "-n", condaEnv, "python"];
}
console.log(`[orchestrator] python launch: ${pythonBin} ${argvPrefix.join(" ")}`.trim());

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
  // Gate on the in-memory job map to prevent path traversal via :id.
  if (!jm.get(req.params.id)) return res.status(404).type("text/plain").send("not found");
  const logPath = path.join(runsDir, req.params.id, "output.log");
  try {
    const content = await fs.readFile(logPath, "utf8");
    res.type("text/plain").send(content);
  } catch {
    res.status(404).type("text/plain").send("not found");
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

  const job = await jm.start({ flowId, label: flow.label, argv: [...argvPrefix, ...argv], pythonBin });
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

export { app, jm };
