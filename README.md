# On-Device Transaction Parser — Fine-Tuning Pipeline

## Quick start (Linux / WSL)

One-shot setup — installs everything (torch cu128, training deps, CUDA-built
llama-cpp-python, node deps) and pulls the trained models from Hugging Face:

```bash
conda create -n llm-training python=3.11 -y && conda activate llm-training
bash setup.sh                # full setup
# bash setup.sh --no-models  # skip HF download
# bash setup.sh --cpu-only   # skip CUDA build for llama-cpp-python
```

The script is idempotent — re-running it short-circuits already-satisfied
steps. Manual setup instructions are below if you need finer control.

## Pretrained weights (Hugging Face)

The trained teacher + student artifacts (LoRA adapters + GGUFs) are mirrored at
[`kartikey31/txn-parser`](https://huggingface.co/kartikey31/txn-parser). The
`models/` directory is **not** tracked in this repo — pull it from HF before
running anything past Stage 2.

```bash
pip install -U "huggingface_hub[cli]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1   # PowerShell: $env:HF_HUB_ENABLE_HF_TRANSFER = "1"

huggingface-cli download kartikey31/txn-parser \
    --repo-type=model --local-dir models
```

Re-running the same command resumes — already-present files are skipped.

---


Trains a small language model that turns voice-transcribed transaction strings
("500 rs on beer 50 rs on candy") into a structured JSON array of transaction
DTOs. The final **student** model ships as a Q4_K_M GGUF (~300 MB) and runs
on-device on Android via `llama.cpp`.

The pipeline is two-stage distillation:

1. Fine-tune a **teacher** (Gemma 4 E2B) on a small human-supervised dataset.
2. Use the teacher to label a much larger synthetic dataset, then fine-tune
   the **student** (Gemma 3 270M) on it.

---

## Prerequisites

| | Required for |
|---|---|
| Python 3.11 | All stages |
| Node.js 20+ | Stages 2, 7 |
| NVIDIA GPU, 12+ GB VRAM | Stages 3, 5, 6 (training + teacher inference) |
| CUDA 12.8 toolkit (required for Blackwell sm_120) | Stages 3, 5, 6 |
| C/C++ build tools (MSVC, CMake) | Stage 3 GGUF export, Stage 7 `llama-cpp-python` |
| DeepSeek API key | Stages 1, 5 |

## One-time setup

We use **conda** for the Python environment (handles CUDA-aware torch cleanly on
Windows) but install pipeline packages with **pip** since Unsloth and
`llama-cpp-python` ship the freshest wheels there.

```powershell
# 1. Create + activate a dedicated env with Python 3.11
conda create -n llm-training python=3.11 -y
conda activate llm-training
python -m pip install --upgrade pip

# 2. Base deps (Stages 1, 2 backend, 4)
pip install -r requirements.txt

# 3. Training deps (Stages 3, 5, 6).
#    RTX 50-series / Blackwell is sm_120 — REQUIRES torch >= 2.6 built against CUDA 12.8.
#    torch 2.5 / cu124 silently lacks sm_120 kernels; do not use it on Blackwell.
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt

# 4. Inference deps (Stages 4, 7) — build llama-cpp-python with CUDA support.
#    Requires MSVC Build Tools (or gcc on Linux) + CUDA toolkit 12.8 on PATH.
$env:CMAKE_ARGS = "-DGGML_CUDA=on"   # PowerShell — on bash: export CMAKE_ARGS="-DGGML_CUDA=on"
pip install -r requirements-eval.txt --no-cache-dir

# 5. Node env (Stages 2 + 7)
cd viewer
npm install
cd ..
```

Each new shell needs `conda activate llm-training` before running any script.
Verify the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

> **Why cu128, not cu124.** Blackwell (sm_120, RTX 50-series) needs CUDA 12.8+ and
> torch ≥ 2.6. The older cu124 wheels predate sm_120 and will either crash with
> "no kernel image" or silently fall back to ptxas JIT (very slow).

## Environment variables

| Variable | Used by | Notes |
|---|---|---|
| `DEEPSEEK_API_KEY` | Stages 1, 5 | Required for any DeepSeek API call |
| `PYTHON_BIN` | Stage 7 | Python executable for the spawned inference worker. Default: `python` on Windows, `python3` elsewhere. Set if your conda `python` isn't on PATH. |
| `LLAMA_N_GPU_LAYERS` | Stage 7 | Layers to offload to GPU in `llama-cpp-python`. Default: all (set to `0` to force CPU). |
| `INFER_TIMEOUT_MS` | Stage 7 | Per-request timeout for inference. Default: `120000` (120 s). |
| `PORT` | Stages 2, 7 | Viewer/playground HTTP port. Default: `3000`. |

Set per-shell:

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
```

---

## Pipeline at a glance

| Stage | Script / Service | Inputs | Outputs |
|---|---|---|---|
| 1a | `scripts/01_generate_dataset.py` | `data_gen_prompt.md` | `data/raw/batch_NN.jsonl` |
| 1b | `scripts/02_clean_dataset.py` | `data/raw/*.jsonl` | `data/clean/{train,eval}.jsonl` |
| 2 | `viewer/server.js` | `data/clean/`, `data/flags.json` | Browser viewer at `:3000` |
| 3 | `scripts/03_train_teacher.py` | `data/clean/{train,eval}.jsonl` | `models/teacher/{adapters,gguf}/` |
| 4 | `scripts/04_eval.py` | A model + `data/clean/eval.jsonl` | `eval_results/<name>.jsonl` |
| 5 | `scripts/05_generate_distillation_data.py` | Trained teacher | `data/distill/train.jsonl` |
| 6 | `scripts/06_train_student.py` | `data/distill/train.jsonl` | `models/student/{adapters,gguf}/` |
| 7 | `viewer/server.js` `/playground` | Teacher + student GGUFs | Browser playground at `:3000/playground` |

Every script accepts `--help` and writes a timestamped log under `logs/`. Re-running any script is safe — outputs that already exist are skipped (pass `--force` to overwrite).

---

## Stage 1 — Dataset generation

Generate 25 batches via DeepSeek, then validate, deduplicate, and split.

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."

# 1a: 25 batches into data/raw/ (idempotent — skips existing batches)
python scripts/01_generate_dataset.py

# 1b: validate + dedup + 90/10 split into data/clean/
python scripts/02_clean_dataset.py
```

Useful flags:
- `01_generate_dataset.py --batches 5` — run only the first 5 batches (smoke test).
- `01_generate_dataset.py --model deepseek-reasoner` — swap the DeepSeek model ID.
- `01_generate_dataset.py --force` — regenerate a batch even if its file exists.
- `02_clean_dataset.py --eval-frac 0.1 --seed 42` — adjust the split.

## Stage 2 — Dataset viewer (browser)

Inspect `data/clean/*.jsonl` side-by-side, search, and flag bad examples.

```powershell
cd viewer
npm start
# → http://localhost:3000
```

The viewer hot-reads the JSONL files (mtime cache), so re-running Stage 1 doesn't need a server restart. Flags are appended to `data/flags.json` as an audit log.

Endpoints (if you want to script around it):
- `GET  /api/examples?file=train|eval&page=N&pageSize=20&q=…`
- `GET  /api/stats?file=train|eval`
- `POST /api/flag` — `{file, index, reason}`

## Stage 3 — Fine-tune the teacher (Gemma 4 E2B)

QLoRA via Unsloth on `unsloth/gemma-4-E2B-it`.

```powershell
python scripts/03_train_teacher.py
```

Spec defaults are baked in: LoRA r=16/α=32, all linear targets, 3 epochs, per-device batch=4 × grad-accum=4, cosine 2e-4, warmup 0.03, bf16, eval every 50 steps, `adamw_8bit`.

Outputs:
- `models/teacher/adapters/` — LoRA adapter
- `models/teacher/gguf/` — merged Q3_K_M GGUF
- `models/teacher/checkpoints/` — trainer checkpoints (last 2 kept)

Useful flags:
- `--skip-gguf` — train and save adapter, skip the slow GGUF export.
- `--resume` — continue from the latest checkpoint.
- `--force` — retrain even if the adapter already exists.
- `--max-steps 20` — smoke-test the whole pipeline on a handful of steps.
- `--max-seq-length 2048` — if your inputs+outputs get long.

## Stage 4 — Evaluate a model

Reusable eval script — runs against any model path and writes per-example results.

```powershell
# Teacher GGUF (after Stage 3)
python scripts/04_eval.py --model models/teacher/gguf

# Teacher fp16 via the LoRA adapter (better quality, used in Stage 5)
python scripts/04_eval.py --model models/teacher/adapters --name teacher-fp16

# Student (after Stage 6)
python scripts/04_eval.py --model models/student/gguf

# Smoke test on the first 50 examples
python scripts/04_eval.py --model models/teacher/gguf --limit 50
```

Reports % JSON-valid, % schema-valid, % exact match, and a confusion matrix for `category`. Per-example results land in `eval_results/<model_name>.jsonl`.

## Stage 5 — Teacher generates distillation data

Generate 30k synthetic inputs via DeepSeek, then label each one with the fine-tuned teacher (fp16, not the quantized version).

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."

# Default: run all three phases end to end (inputs -> label -> copy eval)
python scripts/05_generate_distillation_data.py

# Or run a single phase
python scripts/05_generate_distillation_data.py --phase inputs --n-inputs 30000
python scripts/05_generate_distillation_data.py --phase label
python scripts/05_generate_distillation_data.py --phase eval

# Smoke test: label only 50 inputs end-to-end
python scripts/05_generate_distillation_data.py --phase label --limit 50
```

Outputs:
- `data/distill/inputs_raw.jsonl` — raw synthetic inputs from DeepSeek (Phase 1 checkpoint, resume-friendly)
- `data/distill/train.jsonl` — teacher-labeled, schema-validated examples (Phase 2)
- `data/distill/failed.jsonl` — teacher outputs that failed JSON/schema checks (for inspection)
- `data/distill/eval.jsonl` — copy of `data/clean/eval.jsonl` (Phase 3)

All three phases are idempotent. Phase 2 uses the teacher's LoRA adapter loaded in **fp16** (not the quantized GGUF) — quality matters for distillation labels. Re-running picks up from existing on-disk state via input-string deduplication.

## Stage 6 — Fine-tune the student (Gemma 3 270M)

```powershell
python scripts/06_train_student.py

# Useful variants
python scripts/06_train_student.py --skip-gguf         # iterate without GGUF export
python scripts/06_train_student.py --skip-comparison   # train only, no eval after
python scripts/06_train_student.py --resume            # resume from latest checkpoint
python scripts/06_train_student.py --max-steps 20      # smoke test the whole flow
```

Shares the training loop with Stage 3 (`scripts/_training.py`). Student-specific defaults:
- Base model: `unsloth/gemma-3-270m-it` (override with `--model`)
- Training data: `data/distill/train.jsonl` (teacher-labeled, ~30k examples)
- LoRA r=32 / α=64 (higher capacity than teacher since the base is much smaller)
- 2 epochs (more data, fewer epochs to avoid overfit)
- Batch 8 × grad-accum 2 (the smaller model fits a larger batch)
- GGUF quant: **Q4_K_M** — this is the file that ships to Android

Outputs:
- `models/student/adapters/`
- `models/student/gguf/` — Q4_K_M, ~270 MB
- `models/student/checkpoints/`

After training, the script auto-runs `scripts/04_eval.py` against both teacher
and student GGUFs (on `data/clean/eval.jsonl`) and prints a side-by-side table
of JSON-valid / schema-valid / exact-match / mean-latency with Δ columns.
Per-model eval details land in `eval_results/<name>.jsonl`.

## Stage 7 — Playground (manual testing UI)

Same Express server as Stage 2, with an extra `/playground` page and a pair of long-running Python inference workers (one per model). Each worker holds the GGUF in memory via `llama-cpp-python`, so requests are warm.

> **Important:** activate the conda env in the *same shell* you run `npm start` from. The server spawns Python subprocesses with `PYTHON_BIN` (default `python`), and those need to find `llama-cpp-python` plus the `scripts/_lib.py` module.

```powershell
conda activate llm-training       # MUST be active in this shell
cd viewer
npm start
# → http://localhost:3000/playground
```

On startup the server scans `models/teacher/gguf/` and `models/student/gguf/`. If a directory is missing or has no `.gguf`, that worker is skipped and the corresponding option in the UI is disabled — so the playground works even if only one model is trained.

Pick **teacher / student / both**, paste an input, hit **Run** (or `Ctrl/Cmd + Enter`). In "both" mode the two outputs render side-by-side with **per-line diff highlighting** so divergences between teacher and student JSON jump out. Inference is deterministic (temp 0, top_p 1, default `max_tokens=512`). History of the last 20 inputs is kept in `localStorage`.

Endpoints (for scripting):
- `GET  /api/models` — `{ models: { teacher: {ready, gguf}, student: {…} } }`
- `POST /api/infer` — `{ model: "teacher"|"student"|"both", input, max_tokens? }`

Tuning env vars (see the table above): `PYTHON_BIN`, `LLAMA_N_GPU_LAYERS`, `INFER_TIMEOUT_MS`, `PORT`.

---

## Project layout

```
.
├── data_gen_prompt.md          # source prompt used in Stages 1 and 5
├── requirements.txt            # base deps (Stages 1, 2 backend, 4)
├── requirements-train.txt      # Unsloth + training stack (Stages 3, 5, 6)
├── data/
│   ├── raw/                    # DeepSeek batches (Stage 1a)
│   ├── clean/                  # train.jsonl, eval.jsonl (Stage 1b)
│   ├── distill/                # teacher-labeled data (Stage 5)
│   └── flags.json              # bad-example audit log (Stage 2)
├── scripts/
│   ├── 01_generate_dataset.py
│   ├── 02_clean_dataset.py
│   ├── 03_train_teacher.py
│   ├── 04_eval.py
│   ├── 05_generate_distillation_data.py
│   └── 06_train_student.py
├── viewer/
│   ├── server.js               # Express server (Stages 2 + 7)
│   ├── inference_worker.py     # long-running llama-cpp-python worker (Stage 7)
│   └── public/                 # index.html, playground.html, app.js, style.css
├── models/
│   ├── teacher/{adapters,gguf,checkpoints}/
│   └── student/{adapters,gguf,checkpoints}/
├── eval_results/               # per-model evaluation outputs (Stage 4)
└── logs/                       # one timestamped log per script run
```

## Targets

| Model | JSON-valid | Schema-valid | Exact match |
|---|---|---|---|
| Teacher | > 98% | > 97% | > 85% |
| Student | > 97% | > 95% | > 80% |

If a target is missed after training, propose specific fixes (more data, different LoRA rank, more epochs) rather than declaring success.
