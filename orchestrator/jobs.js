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
    let child;
    try {
      child = spawn(pythonBin, argv, {
        cwd: this.projectRoot,
        env,
        windowsHide: true,
        detached: process.platform !== "win32",
        shell: false,
      });
    } catch (err) {
      try { job.logStream.end(); } catch {}
      job.status = "failed";
      job.exitCode = null;
      job.endedAt = new Date().toISOString();
      await this._writeMeta(job);
      this._emit({ type: "status", jobId: id, status: "failed", exitCode: null, endedAt: job.endedAt });
      throw err;
    }
    job.child = child;
    job.pid = child.pid;

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

    let finalized = false;
    const finalize = async (status, exitCode) => {
      if (finalized) return;
      finalized = true;
      if (job.killTimer) { clearTimeout(job.killTimer); job.killTimer = null; }
      job.status = status;
      job.exitCode = exitCode;
      job.endedAt = new Date().toISOString();
      try { job.logStream.end(); } catch {}
      await this._writeMeta(job);
      this._emit({ type: "status", jobId: id, status, exitCode, endedAt: job.endedAt });
    };

    child.on("error", err => {
      writeLine(`spawn error: ${err.message}`, "stderr");
      finalize("failed", null);
    });

    child.on("exit", code => {
      let status;
      if (job.stopAttempts > 0) status = "killed";
      else if (code === 0) status = "succeeded";
      else status = "failed";
      finalize(status, code);
    });

    await this._writeMeta(job);

    const rlOut = readline.createInterface({ input: child.stdout });
    rlOut.on("line", l => writeLine(l, "stdout"));
    const rlErr = readline.createInterface({ input: child.stderr });
    rlErr.on("line", l => writeLine(l, "stderr"));

    this._emit({ type: "status", jobId: id, status: "running", startedAt });
    return this._publicMeta(job);
  }

  async _writeMeta(job) {
    const metaPath = path.join(this.runsDir, job.id, "meta.json");
    const meta = this._publicMeta(job);
    await fsp.writeFile(metaPath, JSON.stringify(meta, null, 2));
  }

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
}
