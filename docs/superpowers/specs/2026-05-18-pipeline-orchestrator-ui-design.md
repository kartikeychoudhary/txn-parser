# Pipeline Orchestrator UI — Design

**Date:** 2026-05-18
**Status:** Draft for implementation

A standalone web UI that launches and monitors the project's pipeline scripts
(Stages 1–6, multi-provider scenarios, eval/predict/export utilities), streams
their stdout/stderr live to the browser, and lets the user force-quit a run
gracefully or hard.

## Motivation

Today the pipeline is driven from PowerShell. Re-running stages with
slightly different flags, watching long training jobs, and remembering past
invocations all happen in the user's shell history. The viewer at
`viewer/server.js` covers dataset inspection and the playground but offers
nothing for orchestrating stage scripts.

This design adds a focused, single-user, local-only UI for kicking off and
monitoring pipeline jobs. It runs alongside the existing viewer, not in
place of it.

## Goals

- Launch any of the curated pipeline flows from a web UI on
  `http://localhost:3100` with editable common flags.
- Stream the spawned process's stdout+stderr to the browser in real time.
- Support multiple jobs running in parallel; switching between consoles is
  one click.
- Two-stage force quit (graceful SIGINT → hard kill the whole process tree).
- Persist run history (command, timestamps, exit code, full log) to disk so
  it survives server restarts.
- Look good. Dark theme, modern typography, smooth state transitions.

## Non-goals

- Auth, multi-user, or remote access — local single-user only.
- Editing arbitrary CLI flags beyond the curated set. A free-form
  "Additional args" textarea is the escape hatch.
- Replacing or merging into `viewer/`.
- GPU telemetry, cost dashboards, or live `metrics.json` parsing (could be
  follow-on work).
- Auto-discovery of every script in `scripts/` — the catalogue is curated.

## Architecture

New top-level directory `orchestrator/` parallel to `viewer/`:

```
orchestrator/
├── server.js            # Express + ws, routes, boot-time runs/ scan
├── flows.js             # Flow catalogue (id, label, command builder, arg schema)
├── jobs.js              # Job manager: spawn, stream, persist, kill
├── public/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── runs/                # Persisted history (gitignored)
│   └── <jobId>/
│       ├── meta.json
│       └── output.log
└── package.json
```

`server.js` is the only entry point. It serves the static frontend, exposes
the JSON API, and upgrades WebSocket connections for log streaming.

`flows.js` is a pure module exporting the flow catalogue. Each flow declares
how to translate UI form values into argv for `python scripts/<name>.py …`.

`jobs.js` owns the lifecycle of every spawned child: PID, process group,
status state machine, log file handle, and the in-memory subscribers list
for WebSocket fan-out.

### Stack

- **Server:** Node.js 20+, Express, `ws` for WebSocket, no other runtime
  deps. Matches `viewer/` conventions.
- **Frontend:** Vanilla HTML/CSS/JS — no React/Vue/bundler. CSS uses
  modern features (custom properties, `backdrop-filter`, `:has()`,
  container queries) targeting evergreen browsers.
- **Fonts:** Inter (UI), JetBrains Mono (console). Self-hosted under
  `public/fonts/` so the app works offline.

### Why a separate process

Keeps the orchestrator's lifecycle independent from the inference workers
the viewer spawns. Crashing a Python label run shouldn't take down the
playground, and vice versa.

## Flow catalogue

Curated based on `USE.md`. Each entry has:

```js
{
  id: "stage_6_train_student",
  label: "Stage 6 — Train student",
  description: "Fine-tune Gemma 3 270M on data/distill/train.jsonl",
  script: "scripts/06_train_student.py",
  args: [
    { name: "batch-size",    type: "int",    default: 8 },
    { name: "grad-accum",    type: "int",    default: 2 },
    { name: "epochs",        type: "float",  default: 2.0 },
    { name: "resume",        type: "flag",   default: false },
    { name: "skip-gguf",     type: "flag",   default: false },
    { name: "skip-comparison", type: "flag", default: false },
  ],
  // free-form "Additional args" textarea is appended verbatim on all flows
}
```

Initial flows:

| id | Script | Notes |
|---|---|---|
| `stage_1_generate_raw` | `01_generate_dataset.py` | `--batches`, `--model` |
| `stage_2_clean_dataset` | `02_clean_dataset.py` | `--eval-frac`, `--seed` |
| `stage_3_train_teacher` | `03_train_teacher.py` | batch/grad/epochs/resume/skip-gguf/max-steps |
| `stage_4_eval` | `04_eval.py` | `--model`, `--limit`, `--no-grammar` |
| `stage_5_legacy` | `05_generate_distillation_data.py` | `--phase`, `--limit`, `--retry-failed`, `--retry-validation-failed` |
| `stage_5_multiprovider` | `05_generate_distillation_data.py` | `--phase`, `--provider-config`, `--multi-provider` |
| `stage_6_train_student` | `06_train_student.py` | batch/grad/epochs/resume/skip-gguf/skip-comparison |
| `predict_one` | `predict_one.py` | `--model`, positional input |
| `export_gguf` | `export_gguf.py` | `--role`, `--quants` |

The catalogue is a single static JS object — adding a flow means editing
`flows.js`, no migration needed.

## Job lifecycle

```
queued → running → (stopping →) succeeded | failed | killed | orphaned
```

- **queued** is transient — the moment the user clicks "Start", a job is
  created with status `queued`, then immediately spawned and moved to
  `running`. We keep the state so the UI never sees a job without a row.
- **stopping** is entered when the user clicks Stop the first time; the
  graceful signal has been sent but the child hasn't exited yet.
- **orphaned** is set at server boot for any `runs/<id>/meta.json` whose
  status was `running` or `stopping` (server crashed mid-job; PID is
  unverifiable across restarts).

### Spawn

```js
const child = spawn(pythonBin, [script, ...argv], {
  cwd: projectRoot,
  env: { ...process.env, PYTHONUNBUFFERED: "1" },
  // Windows: create new process group so we can CTRL_BREAK / taskkill /T
  windowsHide: true,
  detached: process.platform !== "win32", // POSIX: detach to make child the group leader
  shell: false,
});
```

`PYTHONUNBUFFERED=1` is critical — without it Python buffers stdout when
the descriptor isn't a TTY and the user sees nothing until the script
finishes.

`pythonBin` resolves from `PYTHON_BIN` env var, falling back to `python`
on Windows and `python3` elsewhere (same convention as `viewer/server.js`).

### Streaming

Each child's `stdout` and `stderr` are line-split (using `readline`
interfaces). Every line goes two places:
1. Appended to `runs/<jobId>/output.log` (single source of truth).
2. Fanned out to every open WebSocket subscribed to this job:
   `{type: "line", stream: "stdout"|"stderr", text, ts}`.

A small rolling tail (last 500 lines) is kept in memory so new
subscribers can catch up immediately without reading the log file.
Subscribers connected after the tail's start receive the rolling tail on
connect, then live lines.

On child exit, the server writes the final `meta.json`, closes the log
file, and emits `{type: "status", status: "succeeded"|"failed"|"killed", exitCode}`
to subscribers.

### Persistence

For job `J`:

`orchestrator/runs/J/meta.json`:
```json
{
  "id": "J",
  "flowId": "stage_6_train_student",
  "label": "Stage 6 — Train student",
  "argv": ["scripts/06_train_student.py", "--batch-size", "16"],
  "command": "python scripts/06_train_student.py --batch-size 16",
  "startedAt": "2026-05-18T12:01:33.412Z",
  "endedAt": "2026-05-18T13:04:12.880Z",
  "status": "succeeded",
  "exitCode": 0,
  "pid": 18472
}
```

`orchestrator/runs/J/output.log`: raw text. Each line is prefixed by a
small marker the server writes itself (not the child), e.g.
`[12:01:33.412] OUT step 10/2000 loss=…`. Stderr lines use `ERR`.

Job IDs are ULIDs (lexicographically sortable by time) so the runs/
directory listing doubles as a time-ordered index.

On boot, `server.js` scans `runs/*/meta.json`, rebuilds the in-memory
index, and marks any non-terminal status as `orphaned`.

### Force quit

Two-stage. Implemented in `jobs.js#stop(jobId)`:

1. **First call (graceful).** Status → `stopping`.
   - POSIX: `process.kill(-pgid, "SIGINT")` (negative PID kills the group).
   - Windows: spawn `taskkill /PID <pid> /T` (no `/F`). This terminates
     the process tree but lets handlers run. `CTRL_BREAK_EVENT` is an
     alternative but only works if the child installed a handler;
     `taskkill /T` is more reliable for our scripts.
   - Start a 5-second timer. If the child has not exited when it fires,
     escalate to step 2 automatically.

2. **Second call OR timer fires (hard).**
   - POSIX: `process.kill(-pgid, "SIGKILL")`.
   - Windows: `taskkill /PID <pid> /T /F`.
   - Mark status `killed` once `child.on("exit")` fires.

The UI knows which call it's on by inspecting the job's `status`: when
`status === "stopping"`, the button shows "Force kill" in red. When the
status flips to `killed`/`succeeded`/`failed`, the button disappears.

`stop()` is idempotent — repeated calls in `stopping` state collapse to
the hard kill.

## HTTP + WebSocket API

| Method | Path | Notes |
|---|---|---|
| `GET`  | `/api/flows` | Returns the catalogue as JSON. |
| `GET`  | `/api/jobs?status=running\|all` | Lists jobs. Default `all` returns recent + active. |
| `GET`  | `/api/jobs/:id` | Returns meta + last N log lines (default 500). `?tail=N`. |
| `GET`  | `/api/jobs/:id/log` | Full log as `text/plain` (download). |
| `POST` | `/api/jobs` | Body `{flowId, args, extraArgs}`. Returns the new job's meta. |
| `POST` | `/api/jobs/:id/stop` | Graceful-then-hard per status. Returns updated meta. |
| `DELETE` | `/api/jobs/:id` | Only allowed for terminal jobs. Removes the `runs/<id>/` directory. |
| `WS`   | `/ws/jobs/:id` | Streams `{type:"line",...}` and `{type:"status",...}` events. |

All requests are unauthenticated; the server binds to `127.0.0.1` by
default (configurable via `HOST` env var) so it isn't reachable from the
network without explicit opt-in.

## Frontend

Single-page app served from `public/`. Three-column layout:

- **Left:** Flow catalogue. Each flow is a clickable card; clicking
  selects it and reveals the args form in the center column.
- **Center:** Top half = selected flow's args form + "Start" button; the
  job's command preview updates as the user edits. Bottom half = active
  console for the currently selected job. Console is a styled `<div>`
  with monospace lines, sticky auto-scroll, and a "Jump to live" pill
  that appears when the user has scrolled up.
- **Right:** Job list. Running jobs at top with a pulsing dot, then
  terminal jobs sorted newest-first. Clicking a job swaps the console.
  Each job row has an inline status icon (`●` cyan / `✓` green / `✗`
  red / `⏹` amber) and an elapsed/ended-ago time.

A single sticky header shows the active running count.

### Visual style

- **Background:** linear-gradient from `#0b1120` (slate-950) to `#0f172a`
  (slate-900), with a subtle radial blur behind the active console.
- **Cards:** semi-transparent (`background: rgba(255,255,255,0.04)`),
  `backdrop-filter: blur(12px)`, soft border `rgba(255,255,255,0.08)`,
  rounded `12px`.
- **Accent palette:**
  - cyan `#22d3ee` — running
  - emerald `#34d399` — succeeded
  - rose `#fb7185` — failed / killed
  - amber `#fbbf24` — stopping
- **Console:** `#020617` background, JetBrains Mono, log-level coloring
  (stderr lines tinted rose, INFO/WARN/ERROR prefixes parsed and
  highlighted). Cursor block at the tail of live output.
- **Motion:** state changes fade for 150ms; running dot pulses at 1.4s
  intervals; the "Jump to live" pill slides in/out.

### State management

Single in-memory store keyed by `jobId`. On load:
1. `GET /api/flows` populates the catalogue.
2. `GET /api/jobs?status=all` populates the job list.
3. For every running job, open a WebSocket. The currently selected job's
   console renders live; others buffer their lines into the store so
   switching consoles is instant.

WebSocket reconnection on close uses exponential backoff (1s, 2s, 4s,
capped 10s). On reconnect we re-fetch the job's meta and ask for the
tail via `GET /api/jobs/:id?tail=500` to recover any missed lines.

## Error handling

- **Python not found / script missing:** server resolves the script path
  at job creation; if it doesn't exist, returns 400 before spawning.
- **Spawn error:** caught from the `child.on("error")` event; job moves
  to `failed`, `exitCode` set to `null`, error message appended as a
  stderr line.
- **Log file write failure:** logged to server console; the in-memory
  buffer continues so the UI keeps working. We do not crash the server.
- **WebSocket drops:** server treats subscriber lists as best-effort; on
  send error the subscriber is dropped silently.
- **Orphaned jobs at boot:** marked `orphaned` with a note in the UI
  ("Server restarted while this job was running — actual outcome unknown").
  The user can delete them.

## Testing

- **Unit (Node):** `flows.js` argv-builder for each flow type (int, float,
  string, flag, positional). Verify quoting and that omitted args don't
  appear.
- **Integration:** spin up the server, hit `POST /api/jobs` with a flow
  that runs `python -c "for i in range(5): print(i)"` (mocked flow id for
  tests), assert log file is written, WebSocket receives 5 line events,
  and final status is `succeeded` exit 0.
- **Force-quit:** flow that runs `python -c "import time; [time.sleep(1) for _ in range(60)]"`.
  Call stop, assert status flips to `stopping`, then to `killed` within 6s.
  Second test triggers the hard kill on the second stop call.
- **Boot-time orphan recovery:** write a fake `runs/X/meta.json` with
  `status: "running"`, start server, assert job appears with status
  `orphaned`.
- **Manual:** click through the UI on Stage 4 eval (short, safe), confirm
  console streams, force-quit kills cleanly, history persists after
  server restart.

## Risks and open questions

- **Windows process group reliability.** `taskkill /T` walks the process
  tree using Windows job objects; it should reliably catch grandchildren
  (e.g. Python → llama.cpp). Validated manually during implementation.
- **Console performance.** A long training run can emit tens of thousands
  of lines. The DOM cap should be ~5000 visible lines with older lines
  trimmed; the full log on disk remains intact and is downloadable.
- **Stage 5 `metrics.json`.** Not surfaced in v1. Follow-up could parse it
  and show per-provider acceptance / cost bars under the console.
- **Concurrent GPU runs.** UI allows them; if the user starts Stage 3 and
  Stage 6 simultaneously they will OOM. We will not lock this — surface
  a soft warning when starting a GPU-using flow while another is active.

## Out of scope (future work)

- Auth and remote-network access (not needed for local single-user).
- Editing per-flow args beyond the curated set (covered by the free-form
  "Additional args" field).
- Cron / scheduled runs.
- Streaming `metrics.json` as a live chart.
- Pinning specific runs or exporting/sharing a run bundle.
