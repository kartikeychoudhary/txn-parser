# v1.1.0 — Three published student models + multi-quant eval pipeline

The repo's first multi-model release. One training pipeline now produces
three student variants (gemma-3-270m, smollm2-360m, qwen3-0.6b), each
exported to 5 GGUF quantizations and published to a shared HF repo with
machine-generated model cards.

## Highlights

- **3 published students**, all on the same 93k teacher-labeled dataset
- **15 GGUF builds total** (F16 / Q8_0 / Q6_K / Q5_K_M / Q4_K_M per model)
- **One HF repo** with per-base-model subfolders:
  [`kartikey31/txn-parser`](https://huggingface.co/kartikey31/txn-parser)
- **Eval report** ([`eval_results/REPORT.md`](eval_results/REPORT.md))
  with the same eval set, grammar, and decoder for every (model, quant)
  pair — apples-to-apples comparison
- **Android deployment guide** in the README — per-model recommendation,
  llama.cpp param table, battery/responsiveness checklist
- **One-shot reproduction** via `scripts/train_and_publish.py` and
  `scripts/eval_all_quants.py`

## Eval headline

| Model | Best quant | Schema valid | Exact match | Amount exact | Mean latency |
|---|---|---:|---:|---:|---:|
| `gemma-3-270m` | Q5_K_M (260 MB) | 99.7% | 51.0% | 84.7% | 1788 ms |
| `smollm2-360m` | Q4_K_M (271 MB) | 100.0% | 53.3% | 87.3% | 978 ms |
| **`qwen3-0.6b`** | **Q4_K_M (397 MB)** | **100.0%** | **60.0%** | **90.7%** | **885 ms** |

Qwen3-0.6b sweeps accuracy and latency; smollm wins on disk size with
matching schema validity. Full per-quant breakdown in
[`eval_results/REPORT.md`](eval_results/REPORT.md).

## New scripts

| Script | Purpose |
|---|---|
| `scripts/train_and_publish.py` | One-shot multi-model trainer; per-base subfolder publish to HF; resume flags for mid-run failures |
| `scripts/export_gguf.py` | Multi-quant GGUF export from a single adapter load (already existed; now Unsloth-telemetry-safe) |
| `scripts/eval_all_quants.py` | Multi-model multi-quant eval with parallel workers and a Markdown report aggregator |
| `scripts/download_models.py` | Mirror the HF subfolder layout into local `models/student-<short>/` |
| `scripts/cleanup_hf_repo.py` | One-time cleanup that removes the legacy `student/` + `teacher/` folders on HF and uploads the new repo-root README |
| `scripts/dedupe_failed.py` | Collapse `failed.jsonl` retry-noise into one row per unique input |

## Breaking changes

### HF repo layout reorganized

Old layout (pre-v1.1):
```
kartikey31/txn-parser/
├── teacher/{adapters,gguf,checkpoints}/
└── student/{adapters,gguf,checkpoints}/
```

New layout (v1.1+):
```
kartikey31/txn-parser/
├── gemma-3-270m/{adapters,gguf,README.md}/
├── smollm2-360m/{adapters,gguf,README.md}/
├── qwen3-0.6b/{adapters,gguf,README.md}/
└── README.md
```

Any downstream code that pulled from `student/gguf/` paths needs to point
at `<short>/gguf/txn-parser-<short>-<QUANT>.gguf` instead. The legacy
`student/` and `teacher/` folders will be removed by
`scripts/cleanup_hf_repo.py --apply`.

### Local `models/` layout changed

- Before: `models/student/{adapters,gguf,checkpoints}/` (one model)
- After:  `models/student-<short>/{adapters,gguf,README.md}/` (one per base)

The old `models/student/` is no longer written by any script —
`train_and_publish.py` writes to `models/student/` *during* training,
then promotes to `models/student-<short>/` once done.

## Pipeline fixes shipped

- **Parser bugs (7 classes fixed):** decimals being split (150.25 → [150, 25]); Indian comma grouping (1,50,000 → [1, 50000]); cross-language additive ("1 lakh 50 thousand" not summed); colloquial "one eighty" = 180 not captured; hyphenated "ninety-five rupees" dropped by hyphen split; reject-keyword "pin" matching substring of "Shopping" and silently dropping all plain-digit candidates from those inputs
- **Validator + parser tests:** 72 → 118 cases
- **Grammar bug:** GBNF transaction rule was emitted across 7 lines; GBNF requires single-line rules → parser error, then llama.cpp segfault with NULL grammar pointer. Now a single-line rule.
- **Unsloth telemetry:** `get_statistics()` blocked on HF for 120 s and killed export-after-training. Disabled via env var + monkeypatch in both `_training.py` and `export_gguf.py`.

## Installation / upgrade

```bash
git pull
git checkout v1.1.0
bash setup.sh

# If you have a pre-v1.1 local checkout of the models, just delete and
# re-download from the new HF layout:
rm -rf models/student models/teacher   # only the legacy single-model dirs
python scripts/download_models.py      # pulls into models/student-<short>/
```

## Reproduce

Full pipeline from a fresh checkout, on an A100:

```bash
git clone https://github.com/kartikeychoudhary/txn-parser
cd txn-parser
bash setup.sh
export HF_TOKEN=hf_xxx   # write scope

python scripts/05_generate_distillation_data.py --phase eval --force-eval-copy
python scripts/train_and_publish.py             # ~3 hours on A100 80GB
python scripts/eval_all_quants.py --workers 4   # ~10 min, generates REPORT.md
```

## Next

- `qwen3-0.6b-Q4_K_M` is the current recommended model for accuracy-first
  deployments. The training data still has long-tail failures from
  parser-misses (~7k unique inputs in `failed.jsonl`); the next training
  pass will incorporate Sonnet-labeled rescues of those.
- Android sample app (Kotlin + JNI llama.cpp wrapper) for the README's
  deployment guide.
