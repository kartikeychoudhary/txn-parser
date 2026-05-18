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
