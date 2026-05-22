# USE.md — Pipeline commands & scenarios

PowerShell syntax throughout. Switch backticks (`` ` ``) for backslashes if you're on bash.

API keys load automatically from `.env` (DEEPSEEK_API_KEY, GOOGLE_API_KEY or GEMINI_API_KEY). See `.env.example`.

Ctrl+C in Stage 5 stops gracefully; press again to force-quit. All Stage 5 phases are resumable — re-run the same command.

---

## 0. One-time setup

```powershell
# Create .env from template, fill in keys
Copy-Item .env.example .env
notepad .env
```

GPU sanity:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Blackwell (5060 Ti, 50-series) needs **torch built against CUDA 12.8** (cu128). cu124 silently lacks sm_120 kernels — training will run on CPU or fail. Check `torch.version.cuda` shows `12.8`.

---

## 1. End-to-end pipeline (start to finish)

```
Stage 1 → 2 → 3 → 4 → 5 → 6 → 4
raw      clean  teacher  eval  distill  student  eval
```

### Stage 1 — Generate raw dataset (DeepSeek)

```powershell
python scripts/01_generate_dataset.py
```

Outputs `data/raw/batch_01.jsonl` … `batch_25.jsonl` (25 batches default). Parameters:

| Flag | Default | Notes |
|---|---|---|
| `--batches` | 25 | How many batches to run from start |
| `--start-batch` | 1 | Resume from batch N (1..25) |
| `--model` | `deepseek-chat` | Or `deepseek-reasoner` |
| `--temperature` | 1.0 | Higher = more diversity |
| `--max-tokens` | 8000 | Per-call completion cap |
| `--max-retries` | 5 | Transient errors |
| `--force` | off | Regenerate existing batches |

### Stage 2 — Clean + dedupe + split

```powershell
python scripts/02_clean_dataset.py
```

Reads `data/raw/batch_*.jsonl`, writes `data/clean/{train,eval}.jsonl`.

| Flag | Default | Notes |
|---|---|---|
| `--eval-frac` | 0.1 | 10% held out for eval |
| `--seed` | 42 | Shuffle seed |

### Stage 3 — Train teacher (Gemma 4 E2B, QLoRA)

```powershell
python scripts/03_train_teacher.py
```

Outputs `models/teacher/{adapters,gguf,checkpoints}/`. **GPU profile tables below.**

| Flag | Default | Notes |
|---|---|---|
| `--batch-size` | 4 | Per-device train batch |
| `--grad-accum` | 4 | Effective batch = batch_size × grad_accum |
| `--eval-batch-size` | 8 | Keep small — fp32 logit materialization OOMs easily |
| `--epochs` | 3.0 | |
| `--max-seq-length` | 1024 | |
| `--lr` | 2e-4 | |
| `--lora-r` / `--lora-alpha` | 16 / 32 | |
| `--gguf-quant` | `q3_k_m` | Teacher ships small |
| `--skip-gguf` | off | Fast iteration during dev |
| `--resume` | off | Continue from last checkpoint |
| `--max-steps` | -1 | Set e.g. 20 for a smoke test |

### Stage 4 — Evaluate the teacher GGUF

```powershell
python scripts/04_eval.py --model models/teacher/gguf
```

Writes `eval_results/<model_stem>.jsonl`. Grammar-constrained decoding is **on by default** for GGUF.

| Flag | Default | Notes |
|---|---|---|
| `--model` | (required) | `.gguf` file, dir with one, or adapter dir |
| `--eval-file` | `data/clean/eval.jsonl` | |
| `--limit` | 0 | Eval first N (0 = all) |
| `--max-tokens` | — | Per example |
| `--n-ctx` | — | Context length |
| `--n-gpu-layers` | -1 | All on GPU; 0 = CPU |
| `--batch-size` | — | Transformers backend only |
| `--no-grammar` | off | Disable GBNF (debugging only) |

### Stage 5 — Generate distillation data (teacher labels OR multi-provider)

Two execution modes — **legacy** (teacher labels) and **multi-provider** (DeepSeek + Gemini). The multi-provider path is now the default; legacy is fallback.

#### Legacy: teacher labels (requires trained teacher + GPU)

```powershell
python scripts/05_generate_distillation_data.py --phase all
```

Three phases: `inputs` (DeepSeek generates synthetic inputs) → `label` (teacher fp16 labels them) → `eval` (copies `data/clean/eval.jsonl`).

| Flag | Default | Notes |
|---|---|---|
| `--phase` | `all` | Or `inputs` / `label` / `eval` |
| `--n-inputs` | 30000 | Target unique inputs |
| `--inputs-per-call` | 200 | DeepSeek batch |
| `--backend` | `transformers` | Or `gguf` (faster, lossy) |
| `--batch-size` | 16 | Phase 2 transformers; A100 32-64, 5060 Ti 8-16 |
| `--limit` | 0 | Label first N pending |
| `--retry-failed` | off | Retry from `failed.jsonl` |
| `--retry-validation-failed` | off | Retry only validator failures |

#### Multi-provider (no teacher needed — see scenarios below)

### Stage 6 — Train student (Gemma 3 270M, QLoRA)

```powershell
python scripts/06_train_student.py
```

Reads `data/distill/train.jsonl`. After training, auto-evals both teacher and student.

| Flag | Default | Notes |
|---|---|---|
| `--batch-size` | 8 | Per-device |
| `--grad-accum` | 2 | |
| `--eval-batch-size` | 8 | Student vocab is 256k — fp32 logits OOM fast |
| `--epochs` | 2.0 | More data → fewer epochs |
| `--lora-r` / `--lora-alpha` | 32 / 64 | Higher than teacher (smaller base) |
| `--gguf-quant` | `q4_k_m` | What ships to Android |
| `--skip-gguf` | off | |
| `--skip-comparison` | off | Skip auto-eval |
| `--resume` | off | |

---

## 2. Scenarios

### A. Generate distillation data directly from DeepSeek + Gemini (skip teacher)

Use this when you want to train the student **without** training a teacher first. Providers act as the teacher.

**A.1 — Dry-run config (no API calls)**

```powershell
python scripts/05_generate_distillation_data.py `
  --provider-config configs/example_providers.json `
  --dry-run-quota
```

**A.2 — Smoke (100 inputs + 100 labels)**

```powershell
# isolate from any existing data
Move-Item data/distill data/distill.bak -ErrorAction SilentlyContinue
New-Item -ItemType Directory data/distill | Out-Null

python scripts/05_generate_distillation_data.py `
  --phase inputs `
  --provider-config configs/smoke_real_providers.json `
  --multi-provider

python scripts/05_generate_distillation_data.py `
  --phase label `
  --provider-config configs/smoke_real_providers.json `
  --multi-provider
```

Inspect:

```powershell
(Get-Content data/distill/inputs_raw.jsonl | Measure-Object -Line).Lines
(Get-Content data/distill/train.jsonl      | Measure-Object -Line).Lines
Get-Content data/distill/metrics.json | Select-Object -First 60
```

**A.3 — Production (100k inputs)**

Create `configs/prod_providers.json` (see template at bottom). Then:

```powershell
python scripts/05_generate_distillation_data.py `
  --provider-config configs/prod_providers.json --dry-run-quota

python scripts/05_generate_distillation_data.py `
  --phase inputs --provider-config configs/prod_providers.json --multi-provider

python scripts/05_generate_distillation_data.py `
  --phase label  --provider-config configs/prod_providers.json --multi-provider
```

**A.4 — Then train student**

```powershell
# eval set comes from data/clean/eval.jsonl — copy it into distill/
python scripts/05_generate_distillation_data.py --phase eval

python scripts/06_train_student.py
```

#### Provider config tuning

Per-provider knobs in the config JSON:

| Field | Notes |
|---|---|
| `weight` | Quota share among providers (relative) |
| `threads` | Concurrency per provider — 4 is safe; 8-16 if you have headroom |
| `temperature` | 1.0 for input generation (diversity); 0.0 for labeling |
| `max_tokens` | 4000-8000 for inputs; 512 for labels |
| `max_retries` | 3-5 |
| `structured_output` | Gemini only — JSON-schema response API; label phase only |

Top-level `rate_limits.global_max_workers` caps the combined pool. Default 8, push to 16-24 for production.

`validation.max_repair_attempts > 0` enables the stricter-prompt repair loop on validation failures — adds cost but boosts yield by ~5-10%.

### B. Resume an interrupted run

Just re-run the same command. The script reads existing `inputs_raw.jsonl` / `train.jsonl` and only generates what's missing.

```powershell
# Identical command — picks up where Ctrl+C left it
python scripts/05_generate_distillation_data.py `
  --phase label --provider-config configs/prod_providers.json --multi-provider
```

### C. Retry validation failures with stricter prompts

```powershell
python scripts/05_generate_distillation_data.py `
  --phase label --retry-validation-failed
```

(Legacy path — multi-provider does this inline if `validation.retry_invalid_with_stricter_prompt: true`.)

### D. Evaluate any GGUF (teacher or student)

```powershell
python scripts/04_eval.py --model models/student/gguf --limit 100

# Disable grammar (debug only — see what raw model emits)
python scripts/04_eval.py --model models/student/gguf --no-grammar
```

### E. Re-export an adapter to GGUF without retraining

```powershell
python scripts/export_gguf.py --role teacher --quant q3_k_m
python scripts/export_gguf.py --role student --quant q4_k_m

# Multiple quants in one model load
python scripts/export_gguf.py --role student --quants bf16,q8_0,q5_k_m,q4_k_m
```

### F. Predict one input from the CLI

```powershell
python scripts/predict_one.py --model models/student/gguf "800 for a movie"

python scripts/predict_one.py --model models/student/gguf --no-grammar "spent 50 rs on chai"
```

### G. Interactive viewer (browser playground)

```powershell
cd viewer
npm install            # first time only
node server.js
# open http://localhost:3000
```

The viewer launches `inference_worker.py` which loads the student GGUF with grammar enabled by default.

---

## 3. GPU profiles

### 5060 Ti (16 GB VRAM, sm_120 — needs cu128)

| Stage | Knob | Value |
|---|---|---|
| Stage 3 (teacher) | `--batch-size` | 4–8 |
|  | `--grad-accum` | 4–8 (keep effective ≥16) |
|  | `--eval-batch-size` | 4–8 |
|  | `--max-seq-length` | 1024 |
| Stage 5 label (legacy) | `--backend transformers --batch-size` | 8–16 |
| Stage 5 label (legacy) | `--backend gguf` | sequential, lower VRAM |
| Stage 6 (student) | `--batch-size` | 8–16 |
|  | `--grad-accum` | 2–4 |
|  | `--eval-batch-size` | 4–8 |
| Stage 4 eval (transformers) | `--batch-size` | 8–16 |
| Stage 4 eval (gguf) | `--n-gpu-layers` | -1 (all) |

If you OOM at eval time, drop `--eval-batch-size` first — fp32 logit materialization is the usual culprit.

### A100 (80 GB VRAM)

| Stage | Knob | Value |
|---|---|---|
| Stage 3 (teacher) | `--batch-size` | 16–32 |
|  | `--grad-accum` | 1–2 |
|  | `--eval-batch-size` | 8–16 |
| Stage 5 label (legacy) | `--backend transformers --batch-size` | 32–64 |
| Stage 6 (student) | `--batch-size` | 32–64 |
|  | `--grad-accum` | 1 |
|  | `--eval-batch-size` | 8–16 |
| Stage 4 eval (transformers) | `--batch-size` | 32–64 |
| Stage 4 eval (gguf) | `--n-gpu-layers` | -1 |

Multi-provider Stage 5 (DeepSeek/Gemini) is network-bound — GPU doesn't matter. Tune `threads` per provider and `global_max_workers` instead.

### Multi-provider concurrency (any GPU)

| Knob | Suggested |
|---|---|
| `threads` per provider | 4 (safe), 8 (push), 16 (aggressive — watch for 429s) |
| `rate_limits.global_max_workers` | 8 (smoke), 16 (prod), 24 (aggressive) |
| `batch_size` (inputs phase) | 25–50 for diversity; ≤100 to keep retries cheap |
| `write_flush_every` | 25 (smoke), 250 (prod) |

---

## 4. Output locations cheat-sheet

| Path | Stage | Notes |
|---|---|---|
| `data/raw/batch_*.jsonl` | 1 | DeepSeek-generated batches |
| `data/clean/{train,eval}.jsonl` | 2 | Cleaned + split |
| `models/teacher/adapters/` | 3 | LoRA |
| `models/teacher/gguf/` | 3 | Merged + quantized |
| `data/distill/inputs_raw.jsonl` | 5 phase inputs | Synthetic input strings |
| `data/distill/train.jsonl` | 5 phase label | Teacher/provider labels |
| `data/distill/failed.jsonl` | 5 phase label | Validator/parser failures with reason |
| `data/distill/eval.jsonl` | 5 phase eval | Copy of clean eval |
| `data/distill/metrics.json` | 5 multi-provider | Per-run cost + latency + acceptance |
| `models/student/{adapters,gguf}/` | 6 | |
| `eval_results/<name>.jsonl` | 4 / 6 | Per-row eval output |
| `logs/<stage>_<timestamp>.log` | all | Per-run log |

---

## 5. Provider config template (`configs/prod_providers.json`)

```json
{
  "version": 1,
  "input_generation": {
    "enabled": true,
    "target_inputs": 100000,
    "batch_size": 50,
    "dedupe": true,
    "providers": [
      {"name": "deepseek_v4_pro", "type": "deepseek", "model": "deepseek-chat",
       "weight": 50, "threads": 8, "max_retries": 3,
       "temperature": 1.0, "max_tokens": 4000},
      {"name": "gemini_flash", "type": "gemini", "model": "gemini-2.5-flash",
       "weight": 50, "threads": 8, "max_retries": 3,
       "temperature": 1.0, "max_tokens": 4000}
    ]
  },
  "output_generation": {
    "enabled": true,
    "label_attempts_per_input": 2,
    "selection_policy": "first_valid_then_score",
    "providers": [
      {"name": "deepseek_v4_pro", "type": "deepseek", "model": "deepseek-chat",
       "weight": 50, "threads": 8, "temperature": 0.0, "max_tokens": 512},
      {"name": "gemini_flash", "type": "gemini", "model": "gemini-2.5-flash",
       "weight": 50, "threads": 8, "temperature": 0.0, "max_tokens": 512,
       "structured_output": true}
    ],
    "provider_priority": ["gemini_flash", "deepseek_v4_pro"]
  },
  "validation": {
    "schema": true, "semantic_validator": true, "reject_invalid": true,
    "retry_invalid_with_stricter_prompt": true, "max_repair_attempts": 1
  },
  "rate_limits": {"global_max_workers": 16, "write_flush_every": 250}
}
```

Prices for `metrics.json` cost estimates live in `configs/prices.json`. Lookup order: `provider_type:model` → `model` → `provider_type` → `default`. Missing file → $0 with a warning.
