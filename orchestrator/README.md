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
  `python3` elsewhere).

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

## Concurrency

Multiple jobs can run in parallel. The UI warns before launching a
second GPU-using flow (Stages 3-6 + predict_one) while one is already
active — local training/inference shares the GPU and will OOM if you
overlap them.

## API

- `GET /api/flows` — catalogue
- `GET /api/jobs?status=running|all` — list
- `GET /api/jobs/:id?tail=N` — meta + last N lines (clamped 0..5000)
- `GET /api/jobs/:id/log` — full log download
- `POST /api/jobs` — `{flowId, args, extraArgs}` → 201 `{job}`
- `POST /api/jobs/:id/stop` — graceful then hard (idempotent)
- `DELETE /api/jobs/:id` — terminal jobs only
- `WS /ws/jobs/:id` — `{type:"snapshot"|"line"|"status", ...}` frames

## Tests

```
npm test
```

Runs Node's built-in test runner against `test/*.test.js`. The job tests
spawn short Python processes; `python` must be on PATH.
