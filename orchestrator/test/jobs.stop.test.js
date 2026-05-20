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
