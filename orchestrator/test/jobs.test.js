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

test("start emits failed status when pythonBin does not exist", async () => {
  const runsDir = await freshRunsDir();
  const jm = new JobManager({ runsDir, projectRoot: process.cwd() });

  const job = await jm.start({
    flowId: "smoke", label: "missing",
    argv: ["-c", "print('never runs')"],
    pythonBin: "definitely-not-a-real-binary-xyz123",
  });

  const exit = await waitForEvent(jm, "status", ev => ev.jobId === job.id && ["failed","killed","succeeded"].includes(ev.status));
  assert.equal(exit.status, "failed");

  const meta = JSON.parse(await fs.readFile(path.join(runsDir, job.id, "meta.json"), "utf8"));
  assert.equal(meta.status, "failed");
  assert.ok(meta.endedAt);
});
