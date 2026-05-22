# On-Device Transaction Parser — Fine-Tuning Pipeline

[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-kartikey31%2Ftxn--parser-yellow)](https://huggingface.co/kartikey31/txn-parser)

A two-stage distillation pipeline that turns voice-transcribed transaction
strings ("500 rs on beer 50 rs on candy") into structured JSON. Three
student models trained on the same 93k teacher-labeled dataset, published
side-by-side at [`kartikey31/txn-parser`](https://huggingface.co/kartikey31/txn-parser)
so downstream apps can pick the size/quality tradeoff that fits.

## Published student models

All variants live in subfolders of one repo:
[`huggingface.co/kartikey31/txn-parser`](https://huggingface.co/kartikey31/txn-parser/tree/main)

| Base model | Subfolder | Params | Status |
|---|---|---|---|
| `unsloth/gemma-3-270m-it` | [`gemma-3-270m/`](https://huggingface.co/kartikey31/txn-parser/tree/main/gemma-3-270m) | 270M | ✅ Published (5 quants) |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | [`smollm2-360m/`](https://huggingface.co/kartikey31/txn-parser/tree/main/smollm2-360m) | 360M | ✅ Published (5 quants) |
| `Qwen/Qwen3-0.6B` | [`qwen3-0.6b/`](https://huggingface.co/kartikey31/txn-parser/tree/main/qwen3-0.6b) | 600M | ✅ Published (5 quants) |

Each subfolder contains a LoRA adapter (`adapters/`), 5 merged GGUF quants
(`gguf/txn-parser-<short>-{F16,Q8_0,Q6_K,Q5_K_M,Q4_K_M}.gguf`), and a
model-card README with the exact system prompt to use at inference.

## Eval results (300-example held-out set, GBNF-constrained decoding)

Generated via `python scripts/eval_all_quants.py` — same eval set, same
grammar, same decoding settings for every model/quant pair.

### Summary

| Model | Quant | Size | JSON valid | Schema valid | Exact match | Amount exact | Mean ms | P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gemma-3-270m`  | **F16**    | 543 MB | 99.2%  | **99.2%**  | 51.7% | 85.8% | 1250 | 2019 |
| `gemma-3-270m`  | **Q8_0**   | 292 MB | 99.7%  | **99.7%**  | 54.7% | 88.0% | 1697 | 3221 |
| `gemma-3-270m`  | **Q6_K**   | 283 MB | 99.7%  | **99.7%**  | 53.7% | 88.0% | 1813 | 3418 |
| `gemma-3-270m`  | **Q5_K_M** | 260 MB | 99.7%  | **99.7%**  | 51.0% | 84.7% | 1788 | 3444 |
| `gemma-3-270m`  | **Q4_K_M** | 253 MB | 93.3%  | **93.3%**  | 48.3% | 80.7% | 2660 | 15185 |
| `smollm2-360m`  | **F16**    | 726 MB  | 100.0% | **100.0%** | 56.3% | 88.3% | 997  | 1646  |
| `smollm2-360m`  | **Q8_0**   | 386 MB  | 100.0% | **100.0%** | 56.3% | 88.3% | 995  | 1589  |
| `smollm2-360m`  | **Q6_K**   | 367 MB  | 100.0% | **100.0%** | 56.3% | 88.3% | 994  | 1615  |
| `smollm2-360m`  | **Q5_K_M** | 290 MB  | 100.0% | **100.0%** | 52.7% | 89.0% | 996  | 1599  |
| `smollm2-360m`  | **Q4_K_M** | 271 MB  | 100.0% | **100.0%** | 53.3% | 87.3% | 978  | 1595  |
| `qwen3-0.6b`    | **F16**    | 1198 MB | 100.0% | **100.0%** | 59.0% | 91.0% | 851  | 1373  |
| `qwen3-0.6b`    | **Q8_0**   | 639 MB  | 100.0% | **100.0%** | 59.3% | 91.3% | 851  | 1460  |
| `qwen3-0.6b`    | **Q6_K**   | 495 MB  | 100.0% | **100.0%** | 59.0% | 92.0% | 876  | 1419  |
| `qwen3-0.6b`    | **Q5_K_M** | 444 MB  | 100.0% | **100.0%** | 60.0% | 91.3% | 857  | 1415  |
| `qwen3-0.6b`    | **Q4_K_M** | 397 MB  | 100.0% | **100.0%** | 60.0% | 90.7% | 885  | 1373  |

**Headline:** **`qwen3-0.6b`** is the new accuracy leader — best
**exact match (60.0%)** and best **amount accuracy (92.0%)** across the
board, while also being the **fastest** family (~850 ms mean, well below
both gemma and smollm at every matching quant). SmolLM2-360M ties at
100% schema valid and stays smaller on disk. Gemma-3-270m Q4_K_M is the
only build that degrades meaningfully (93% schema + 15 s P95 from
grammar backtracking) — use Q5_K_M or higher in the gemma family.

Pick by deployment target:

| If you need… | Use | Why |
|---|---|---|
| **Best accuracy** | `qwen3-0.6b-Q4_K_M` (397 MB) | 100% schema, 60% exact, ~885 ms, smaller than F16 by 3× |
| **Smallest ship size** | `smollm2-360m-Q4_K_M` (271 MB) | 100% schema, 53% exact, ~1 s — 126 MB lighter than qwen |
| **Fastest mean latency** | `qwen3-0.6b-Q8_0` (639 MB) | 851 ms with full Q8 quality if disk isn't a constraint |

### Per-model detail

#### `gemma-3-270m`
Best schema_valid: **Q8_0** (99.7%, 1697 ms mean).

| Quant | Schema | Exact | Amt | TxnCount | Dup% | Super% | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F16    | 99.2% | 51.7% | 85.8% | 91.7% | 0.0% | 0.0% | 1250 | 2019  |
| Q8_0   | 99.7% | 54.7% | 88.0% | 93.0% | 0.0% | 0.3% | 1697 | 3221  |
| Q6_K   | 99.7% | 53.7% | 88.0% | 93.3% | 0.0% | 0.3% | 1813 | 3418  |
| Q5_K_M | 99.7% | 51.0% | 84.7% | 93.7% | 0.0% | 0.7% | 1788 | 3444  |
| Q4_K_M | 93.3% | 48.3% | 80.7% | 85.7% | 0.0% | 0.3% | 2660 | 15185 |

#### `smollm2-360m`
Best schema_valid: **Q6_K** (100.0%, 994 ms mean).

| Quant | Schema | Exact | Amt | TxnCount | Dup% | Super% | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F16    | 100.0% | 56.3% | 88.3% | 91.7% | 0.3% | 0.0% | 997 | 1646 |
| Q8_0   | 100.0% | 56.3% | 88.3% | 91.7% | 0.3% | 0.0% | 995 | 1589 |
| Q6_K   | 100.0% | 56.3% | 88.3% | 91.3% | 0.3% | 0.0% | 994 | 1615 |
| Q5_K_M | 100.0% | 52.7% | 89.0% | 92.0% | 0.3% | 0.0% | 996 | 1599 |
| Q4_K_M | 100.0% | 53.3% | 87.3% | 90.7% | 0.0% | 0.0% | 978 | 1595 |

#### `qwen3-0.6b`
Best schema_valid: **Q5_K_M** (100.0%, 857 ms mean). Tied with every other
qwen quant on schema validity, edges out by 0.7-1.0% on exact match.

| Quant | Schema | Exact | Amt | TxnCount | Dup% | Super% | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F16    | 100.0% | 59.0% | 91.0% | 93.7% | 0.0% | 0.0% | 851 | 1373 |
| Q8_0   | 100.0% | 59.3% | 91.3% | 94.0% | 0.0% | 0.0% | 851 | 1460 |
| Q6_K   | 100.0% | 59.0% | 92.0% | 93.7% | 0.0% | 0.0% | 876 | 1419 |
| Q5_K_M | 100.0% | 60.0% | 91.3% | 93.3% | 0.0% | 0.0% | 857 | 1415 |
| Q4_K_M | 100.0% | 60.0% | 90.7% | 93.7% | 0.0% | 0.0% | 885 | 1373 |

Quantization barely moves the needle for qwen — F16 → Q4_K_M loses only
0.3% amount accuracy while shrinking 3×. Safe to ship Q4_K_M.

Full JSON results in `eval_results/REPORT.json`; per-example predictions
in `eval_results/<model>-<quant>.jsonl`. Regenerate any time via
`python scripts/eval_all_quants.py` (re-uses cached evals — only re-runs
ones with missing or `--force`'d results).

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

All trained student variants live in subfolders of
[`kartikey31/txn-parser`](https://huggingface.co/kartikey31/txn-parser) on HF.
The `models/` directory is **not** tracked in this repo — pull from HF before
running anything past Stage 2.

### Recommended — `scripts/download_models.py`

```bash
# Everything (all 3 base models × 5 quants, lands in models/student-<short>/)
python scripts/download_models.py

# Just one base model
python scripts/download_models.py --model gemma-3-270m

# Just one quant of every model (fastest sanity check)
python scripts/download_models.py --quant Q4_K_M

# Skip the LoRA adapters (saves ~30 MB per model)
python scripts/download_models.py --no-adapters
```

The script mirrors the HF subfolder layout into
`models/student-<short>/{adapters,gguf,README.md}` so the rest of the
pipeline (`eval_all_quants.py`, `predict_one.py`, the viewer) finds
everything in the expected location. Re-running is idempotent — already-
present files are skipped.

### Manual fallback (`huggingface-cli`)

```bash
pip install -U "huggingface_hub[cli]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1   # PowerShell: $env:HF_HUB_ENABLE_HF_TRANSFER = "1"

# All models (mirrors the full repo)
huggingface-cli download kartikey31/txn-parser --repo-type=model --local-dir models

# Just one quant of one model
huggingface-cli download kartikey31/txn-parser \
    smollm2-360m/gguf/txn-parser-smollm2-360m-Q4_K_M.gguf \
    --local-dir .
```

---


## Pipeline overview

Two-stage distillation:

1. Fine-tune a **teacher** (Gemma 4 E2B, ~5B params) on a small human-supervised dataset
   (`data/clean/train.jsonl`, ~3k examples).
2. Use the fine-tuned teacher to label a much larger synthetic dataset
   (`data/distill/train.jsonl`, ~30k examples), then fine-tune the
   **student** (Gemma 3 270M) on those labels.

The student is the shippable model — small enough for on-device Android
inference, accurate enough for the JSON-output task because the teacher did
the heavy lifting of demonstrating the right structure across many phrasings.

## Quick predict (one input)

```bash
# Using the recommended on-device model (smollm2-360m Q4_K_M, 271 MB)
python scripts/predict_one.py \
    --model models/student-smollm2-360m/gguf/txn-parser-smollm2-360m-Q4_K_M.gguf \
    "500 rs on beer 50 rs on candy"

# Or any other variant — see `Published student models` above
python scripts/predict_one.py \
    --model models/student-gemma-3-270m/gguf/txn-parser-gemma-3-270m-Q5_K_M.gguf \
    "do sau rupay ka chai"
```

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

### Dev dependencies

Tests for the amount parser and validator use `pytest`:

```bash
pip install -r requirements-dev.txt
pytest tests/ --cov=amount_parser --cov=validator
```

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
- `--batch-size N --grad-accum M` — 5060 Ti 16GB: `4 × 4`. A100 80GB: try `16 × 1` or `32 × 1`. Effective batch = `N × M`.

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

# A100 80GB — push the adapter eval batch high
python scripts/04_eval.py --model models/teacher/adapters --batch-size 32
```

Reports % JSON-valid, % schema-valid, % exact match, and a confusion matrix for `category`. Per-example results land in `eval_results/<model_name>.jsonl`.

Stage 4 also reports four validator-derived aggregates: `amount_exact` (predicted
amounts equal expected as multisets), `txn_count_exact`, `duplicate_rate` (fraction
of examples with `duplicate_transactions_found`), and `superseded_amount_used_rate`.
Per-example rows in `eval_results/<name>.jsonl` carry the same fields plus
`validation_errors[]` for failed rows.

Useful flags:
- `--batch-size N` — examples per forward pass for the **transformers/adapter** backend. Default 16. A100 80GB: try 32-64. 5060 Ti 16GB: 8-16. The **GGUF** backend ignores this — `llama.cpp` doesn't natively batch chat completions.
- `--max-tokens N` — generation cap per example (default 512).
- `--n-gpu-layers N` / `--ngl` — GGUF only; `-1` = all (default), `0` = CPU.
- `--n-ctx N` — context window (default 2048).
- `--no-grammar` — GGUF only; disable GBNF grammar-constrained decoding for baseline comparisons (see below).

### GBNF grammar (default-on for GGUF inference)

By default, every GGUF-based inference path (Stage 4 eval, Stage 7
playground, and `scripts/predict_one.py`) constrains output via a GBNF
grammar derived from the validator's enum constants
(`_lib.CATEGORIES`/`TYPES`/`CURRENCIES`). Pass `--no-grammar` to disable
for baseline comparisons. The grammar is rebuilt from `_lib` at every
process start, so adding a category never drifts. See `scripts/grammar.py`
and the design at `docs/superpowers/specs/2026-05-17-grammar-constrained-decoding-design.md`.

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

# A100 80GB — batch hard, ~10x faster than the old single-example loop
python scripts/05_generate_distillation_data.py --phase label \
    --batch-size 32 --max-new-tokens 256
```

Outputs:
- `data/distill/inputs_raw.jsonl` — raw synthetic inputs from DeepSeek (Phase 1 checkpoint, resume-friendly)
- `data/distill/train.jsonl` — teacher-labeled, schema-validated examples (Phase 2)
- `data/distill/failed.jsonl` — teacher outputs that failed JSON/schema checks (for inspection)
- `data/distill/eval.jsonl` — copy of `data/clean/eval.jsonl` (Phase 3)

All three phases are idempotent. Phase 2 uses the teacher's LoRA adapter loaded in **fp16** (not the quantized GGUF) — quality matters for distillation labels. Re-running picks up from existing on-disk state via input-string deduplication.

Useful flags (Phase 2):
- `--batch-size N` — inputs per forward pass (transformers backend). Default 16. A100 80GB: 32-64. 5060 Ti 16GB: 8-16. **Single biggest perf knob** here — 28k labels go from ~20 hours at `batch=1` to ~45 min at `batch=32` on A100.
- `--max-new-tokens N` — default 384. Drop to 256 if outputs fit — saves wall time.
- `--limit N` — only label the first N pending inputs (smoke test).
- `--retry-failed` — re-attempt inputs previously written to `failed.jsonl`.
- `--backend gguf` — use the teacher GGUF (Q3_K_M) via `llama-cpp-python` instead of fp16. Lossier but useful if VRAM is tight or fp16 isn't an option. Sequential — batch_size is ignored.
- `--ngl N` / `--n-gpu-layers N` — GGUF backend only; `-1` = all layers on GPU.
- `--no-mmap`, `--mlock`, `--n-ctx`, `--n-batch` — passthrough to `llama-cpp-python`.

**Validator gate (Phase 2).** Phase 2 uses `scripts/validator.py` to gate teacher
labels: a label must pass the JSON schema AND the semantic checks (amount-in-input,
no-superseded-amount, currency hint, txn count, no unjustified duplicates). Rejected
rows land in `data/distill/failed.jsonl` with `reason ∈ {validation_failed,
json_parse_failed, teacher_error}` and structured `validation.errors[]` carrying
machine-readable codes. The `failed.jsonl` shape changed in this release — delete
or archive the old file before re-running. Re-attempt only semantic-validation
failures with `--retry-validation-failed`.

### Multi-provider scaffolding (Slice 1)

Stage 5 has a new optional path for multi-provider input/label generation, gated behind `--provider-config`. Slice 1 only supports dry-run config validation; real multi-provider execution lands in Slice 2/3.

```bash
# Validate a provider config and see quota allocations
python scripts/05_generate_distillation_data.py \
    --provider-config configs/test_providers.json \
    --dry-run-quota
```

The full config schema is documented in `docs/provider_config.md`. Two example configs ship:
- `configs/test_providers.json` — fake-only, used by smoke tests.
- `configs/example_providers.json` — realistic shape with DeepSeek / Gemini / local-teacher providers.

A standalone validator probe lets you inspect any JSONL of `(input, output)` pairs against the semantic validator:

```bash
python scripts/probe_validator.py \
    --input data/distill/train.jsonl \
    --output reports/validator_probe.jsonl
```

The probe reports per-code failure counts (e.g., AMOUNT_NOT_IN_INPUT, SUSPICIOUS_DUPLICATE) and writes an inspectable per-row report.

Without `--provider-config`, Stage 5 behaves exactly as before.

### Multi-provider input generation (Slice 2)

Slice 2 makes `--phase inputs --provider-config <path> --multi-provider` actually run with real DeepSeek + Gemini API calls. Set the API keys, point at a config, and run:

```bash
export DEEPSEEK_API_KEY=sk-...
export GOOGLE_API_KEY=...     # or GEMINI_API_KEY
python scripts/05_generate_distillation_data.py \
    --phase inputs \
    --provider-config configs/smoke_real_providers.json \
    --multi-provider
```

The generation loop is threaded per-provider (each provider's `threads` field in the config), with global dedupe against the existing `data/distill/inputs_raw.jsonl` and resume safety (re-running picks up at the unique-input count and only generates the remaining `target_inputs`). New rows include provider metadata: `_provider`, `_model`, `_batch_id`.

Quota is a soft scheduling hint — workers stop when the global accepted-unique total reaches `target_inputs`, not when a per-provider quota fills. Provider distribution may drift from configured weights based on latency and duplicate rate.

For tests and CI, set `DISTILL_DIR_OVERRIDE=<tmp_path>` to redirect writes away from `data/distill/`. This is a dev/test-only knob and is not surfaced in `--help`.

### Multi-provider output labeling (Slice 3)

Slice 3 makes `--phase label --provider-config <path> --multi-provider` actually run: real DeepSeek + Gemini label generation, a `LocalTeacherProvider` wrapping the fp16 Unsloth backend, validator-gated candidate scoring, and an optional repair-on-failure retry loop.

```bash
# Real API smoke (DeepSeek + Gemini)
export DEEPSEEK_API_KEY=sk-...
export GOOGLE_API_KEY=...     # or GEMINI_API_KEY
python scripts/05_generate_distillation_data.py \
    --phase label \
    --provider-config configs/smoke_real_providers.json \
    --multi-provider \
    --limit 10
```

For each input the orchestrator generates `label_attempts_per_input` candidates across configured providers, hard-rejects any with validator errors, scores survivors (clean validator pass is base 80; +10 count match, +5 no warnings, +5 priority bonus; max 100), and picks the highest-scoring. If all candidates fail and repair is enabled (`validation.retry_invalid_with_stricter_prompt: true` plus `max_repair_attempts > 0`), the orchestrator re-prompts the highest-priority provider with a stricter repair prompt containing the parsed amount candidates and validator failure summary.

Accepted rows in `data/distill/train.jsonl` carry `_source: "multi_provider_label"`, `_provider`, `_model`, `_validation_score`, `_attempts` metadata. Failed inputs land in `data/distill/failed.jsonl` with `reason ∈ {all_candidates_failed, repair_exhausted, provider_error}` and an `attempts[]` array of per-provider diagnostics.

Resume-safe: re-running skips inputs already in `train.jsonl` and (unless `--retry-failed`) inputs already in `failed.jsonl`. Use `--retry-validation-failed` to re-attempt only rows whose attempts include a validator-class failure.

`--phase eval` is legacy-only — omit `--multi-provider` for eval. `--phase all` exits via `parser.error`.

For local-teacher labeling (requires GPU + trained adapter at `models/teacher/adapters`), use `configs/smoke_local_teacher_providers.json`.

### Per-run metrics (Slice 4)

Every multi-provider Stage 5 run emits `data/distill/metrics.json` with per-
provider call counts, accepted/rejected candidates, latency (p50/p95), token
counts, and estimated USD cost. A compact summary is logged at the end of the
run:

```
=== Stage 5 metrics (phase=label, 100 inputs, 2m41s) ===
Calls: 213 (198 ok, 15 failed)  Accepted: 87  Failed rows: 13
Estimated cost: $0.0342 (rates from configs/prices.json)

Provider          Calls  Ok    Fail%   Acc   Cost      p50    p95    Top failure
gemini_flash      108    105   2.78%   52    $0.0283   812    1421   validation_failed (7)
deepseek_v4_pro   105    93    11.43%  35    $0.0059   1104   2210   provider_error (12)

Repair: 12 attempted, 5 accepted, 7 exhausted
```

Costs are estimates based on `configs/prices.json` at run time. The script
does not fetch live pricing — update `configs/prices.json` to match current
provider rates if you care about USD accuracy.

Metrics fire only for `--multi-provider --phase {inputs,label}`. Legacy
single-provider phases, `--phase eval`, `--phase all`, and `--dry-run-quota`
produce no metrics file.

## Stage 6 — Fine-tune the student (Gemma 3 270M)

```powershell
python scripts/06_train_student.py

# Useful variants
python scripts/06_train_student.py --skip-gguf         # iterate without GGUF export
python scripts/06_train_student.py --skip-comparison   # train only, no eval after
python scripts/06_train_student.py --resume            # resume from latest checkpoint
python scripts/06_train_student.py --max-steps 20      # smoke test the whole flow

# A100 80GB — student is tiny, push the batch hard.
# Eval batch must stay small even when train is huge: Trainer materializes
# fp32 logits over Gemma 3's 256k vocab, and a big eval batch OOMs at step
# `eval_steps` regardless of how much memory the train pass uses.
python scripts/06_train_student.py --batch-size 128 --grad-accum 1 --eval-batch-size 8
```

Shares the training loop with Stage 3 (`scripts/_training.py`). Student-specific defaults:
- Base model: `unsloth/gemma-3-270m-it` (override with `--model`)
- Training data: `data/distill/train.jsonl` (teacher-labeled, ~30k examples)
- LoRA r=32 / α=64 (higher capacity than teacher since the base is much smaller)
- 2 epochs (more data, fewer epochs to avoid overfit)
- Batch 8 × grad-accum 2 (the smaller model fits a larger batch — bump `--batch-size` on a bigger GPU)
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
├── setup.sh                    # one-shot installer + HF model download (Linux/WSL)
├── data_gen_prompt.md          # source prompt used in Stages 1 and 5
├── requirements.txt            # base deps (Stages 1, 2 backend)
├── requirements-train.txt      # Unsloth + training stack (Stages 3, 5, 6)
├── requirements-eval.txt       # llama-cpp-python (Stages 4, 7)
├── data/
│   ├── raw/                    # DeepSeek batches (Stage 1a)
│   ├── clean/                  # train.jsonl, eval.jsonl (Stage 1b)
│   ├── distill/                # teacher-labeled data (Stage 5)
│   └── flags.json              # bad-example audit log (Stage 2)
├── scripts/
│   ├── _lib.py                 # shared constants, schema, helpers
│   ├── _training.py            # shared SFT + LoRA loop (Stages 3 & 6)
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
├── models/                     # NOT tracked — pulled from kartikey31/txn-parser on HF
│   ├── teacher/{adapters,gguf,checkpoints}/
│   └── student-<base>/{adapters,gguf,README.md}/   # one dir per published base model
├── eval_results/               # per-model evaluation outputs (Stage 4)
└── logs/                       # one timestamped log per script run
```

## Targets

| Model | JSON-valid | Schema-valid | Exact match |
|---|---|---|---|
| Teacher | > 98% | > 97% | > 85% |
| Student | > 97% | > 95% | > 80% |

If a target is missed after training, propose specific fixes (more data, different LoRA rank, more epochs) rather than declaring success.
