#!/usr/bin/env python3
"""Train the student on multiple base models in sequence, export each as
a multi-quant GGUF bundle, and push to Hugging Face.

Designed for one-shot runs on rented A100 boxes — minimal interaction,
each model is pushed as soon as it succeeds so a failure mid-run doesn't
lose prior work.

Per base model:
  1. Wipe `models/student/`  (clean-slate retrain)
  2. Train via `scripts/06_train_student.py --skip-gguf --skip-comparison --force`
  3. Export GGUF quants in one model load via `scripts/export_gguf.py --quants ...`
  4. Generate a README.md model card
  5. Upload `adapters/` + `gguf/` + README to `<HF_NAMESPACE>/txn-parser-<suffix>`
     (creates the repo if missing; skips `checkpoints/` to stay under HF quotas)

Requires:
  - HF_TOKEN env var with write scope (or prior `huggingface-cli login`)
  - GPU with enough VRAM for the largest model in the list
  - data/distill/{train,eval}.jsonl present

Usage:
    python scripts/train_and_publish.py                    # all 3 default models
    python scripts/train_and_publish.py --only gemma       # one model
    python scripts/train_and_publish.py --skip-push        # train + export, skip HF
    python scripts/train_and_publish.py --hf-namespace foo # override HF org
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDENT_DIR = REPO_ROOT / "models" / "student"
ADAPTERS_DIR = STUDENT_DIR / "adapters"
GGUF_DIR = STUDENT_DIR / "gguf"
LOGS_DIR = REPO_ROOT / "logs"
TRAIN_FILE = REPO_ROOT / "data" / "distill" / "train.jsonl"
EVAL_FILE = REPO_ROOT / "data" / "distill" / "eval.jsonl"

DEFAULT_HF_NAMESPACE = "kartikey31"
DEFAULT_QUANTS = ["q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"]


@dataclass
class ModelSpec:
    """One row in the run plan."""
    short: str                 # used for --only filtering AND HF suffix
    base_model: str            # HF id passed to 06_train_student.py --model
    batch_size: int
    grad_accum: int
    eval_batch_size: int


# Tuned for an 80GB A100. Adjust batch sizes if you target a smaller card.
# eval_batch_size is the trap on Gemma — 256k vocab × seq_len × batch fp32.
MODELS: list[ModelSpec] = [
    ModelSpec("gemma-3-270m",      "unsloth/gemma-3-270m-it",            64, 1, 4),
    ModelSpec("smollm2-360m",      "HuggingFaceTB/SmolLM2-360M-Instruct", 64, 1, 16),
    ModelSpec("qwen3-0.6b",        "Qwen/Qwen3-0.6B",                     32, 2, 16),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run(cmd: list[str], *, check: bool = True) -> int:
    """Echo + run a subprocess. Returns the exit code; raises on failure
    when check=True."""
    logging.info("$ %s", " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(str(c) for c in cmd)}"
        )
    return proc.returncode


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Per-model steps
# ---------------------------------------------------------------------------


def wipe_student() -> None:
    """Clean slate before each training run so adapters/checkpoints/gguf
    from the previous model don't bleed into this one."""
    if STUDENT_DIR.exists():
        logging.info("Wiping prior %s", STUDENT_DIR)
        shutil.rmtree(STUDENT_DIR)


def train_one(spec: ModelSpec) -> None:
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "06_train_student.py"),
        "--model", spec.base_model,
        "--batch-size", str(spec.batch_size),
        "--grad-accum", str(spec.grad_accum),
        "--eval-batch-size", str(spec.eval_batch_size),
        "--skip-gguf",
        "--skip-comparison",
        "--force",
    ]
    run(cmd)


def export_quants(quants: list[str]) -> None:
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "export_gguf.py"),
        "--role", "student",
        "--quants", ",".join(quants),
    ]
    run(cmd)


def write_model_card(spec: ModelSpec, quants: list[str], train_started: str,
                     train_finished: str) -> Path:
    """Drop a README.md into STUDENT_DIR with provenance + usage notes."""
    train_count = _line_count(TRAIN_FILE)
    eval_count = _line_count(EVAL_FILE)
    gguf_listing = "\n".join(
        f"  - `{p.name}` ({p.stat().st_size / 1e6:.1f} MB)"
        for p in sorted(GGUF_DIR.glob("*.gguf"))
    ) or "  (none — export step did not produce any files)"
    readme = f"""---
license: apache-2.0
base_model: {spec.base_model}
tags:
  - text-generation
  - lora
  - qlora
  - gguf
  - transaction-parser
language:
  - en
  - hi
library_name: peft
---

# txn-parser-{spec.short}

QLoRA fine-tune of [`{spec.base_model}`]({_hf_link(spec.base_model)}) for
extracting structured transaction data (amount, currency, item, category,
type) from free-form Indian-English / code-switched speech and text.

## What's in here

- `adapters/` — PEFT LoRA adapter (rank 32). Load on top of the base model
  with `peft.PeftModel.from_pretrained(base, "{DEFAULT_HF_NAMESPACE}/txn-parser-{spec.short}", subfolder="adapters")`.
- `gguf/` — merged GGUF builds at multiple quantization levels:
{gguf_listing}

## Training data

- {train_count:,} teacher-labeled examples (`data/distill/train.jsonl`)
- {eval_count:,} held-out eval examples (`data/distill/eval.jsonl`)
- Validator-gated: every row's `output` passes the project's grammar +
  amount-parser semantic validator.

## Training config

| Knob | Value |
|---|---|
| Base model | `{spec.base_model}` |
| Method | QLoRA (4-bit) via Unsloth |
| LoRA rank | 32 (alpha 64, dropout 0.0) |
| Epochs | 2 |
| Batch size (train) | {spec.batch_size} |
| Grad accumulation | {spec.grad_accum} |
| Eval batch size | {spec.eval_batch_size} |
| Max seq length | 1024 |
| Learning rate | 2e-4 (warmup 3%) |
| Started | {train_started} |
| Finished | {train_finished} |

## Inference (GGUF, llama.cpp)

```bash
./llama-cli -m txn-parser-{spec.short}-Q4_K_M.gguf \\
    --grammar-file scripts/grammar.gbnf \\
    -p "Spent 200 on chai" -n 256
```

## Reproduce

```bash
git clone https://github.com/kartikeychoudhary/txn-parser.git
cd txn-parser && bash setup.sh
python scripts/train_and_publish.py --only {spec.short}
```

---
*Auto-published by `scripts/train_and_publish.py` on {datetime.now(timezone.utc).isoformat()}.*
"""
    readme_path = STUDENT_DIR / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    return readme_path


def push_to_hf(spec: ModelSpec, namespace: str) -> str:
    """Create the HF repo if missing, then upload adapters/, gguf/, README.

    Returns the repo URL. Requires HF_TOKEN env var (or prior `huggingface-cli login`)."""
    from huggingface_hub import HfApi, create_repo

    repo_id = f"{namespace}/txn-parser-{spec.short}"
    api = HfApi()

    create_repo(repo_id, exist_ok=True, repo_type="model")
    logging.info("Uploading %s -> https://huggingface.co/%s", STUDENT_DIR, repo_id)

    # Skip the Trainer checkpoints (huge, can't be used directly) and any
    # transient training artifacts. Adapters + GGUFs + README is what users
    # actually need.
    api.upload_folder(
        folder_path=str(STUDENT_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message=(
            f"Auto-publish: base={spec.base_model} "
            f"at {datetime.now(timezone.utc).isoformat()}"
        ),
        ignore_patterns=[
            "checkpoints/*",
            "**/optimizer.pt",
            "**/scheduler.pt",
            "**/training_args.bin",
            "**/.cache/**",
        ],
    )
    return f"https://huggingface.co/{repo_id}"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _line_count(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _hf_link(model_id: str) -> str:
    return f"https://huggingface.co/{model_id}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", action="append", default=[],
                   help="Only run this short name. Repeatable. "
                        f"Available: {', '.join(s.short for s in MODELS)}")
    p.add_argument("--skip-push", action="store_true",
                   help="Train + export but skip the HF upload step.")
    p.add_argument("--hf-namespace", default=DEFAULT_HF_NAMESPACE,
                   help=f"HF org/user. Default: {DEFAULT_HF_NAMESPACE}")
    p.add_argument("--quants", default=",".join(DEFAULT_QUANTS),
                   help=f"Comma-separated GGUF quants. Default: {','.join(DEFAULT_QUANTS)}")
    p.add_argument("--keep-on-failure", action="store_true",
                   help="If a model fails, continue to the next instead of aborting.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"train_and_publish_{ts}.log")

    if not TRAIN_FILE.exists():
        logging.error("Missing %s — run Stage 5 first.", TRAIN_FILE)
        return 2
    if not EVAL_FILE.exists():
        logging.warning(
            "%s missing — `python scripts/05_generate_distillation_data.py "
            "--phase eval --force-eval-copy` first.", EVAL_FILE,
        )

    quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    selected = [s for s in MODELS if not args.only or s.short in args.only]
    if not selected:
        logging.error("No models match --only %s. Available: %s",
                      args.only, ", ".join(s.short for s in MODELS))
        return 2

    if not args.skip_push and not os.environ.get("HF_TOKEN"):
        logging.warning(
            "HF_TOKEN env var not set. Upload will fall back to whatever "
            "`huggingface-cli login` stored. If neither is present the push "
            "step WILL fail — re-run with --skip-push or set HF_TOKEN."
        )

    summary: list[dict] = []
    overall_start = time.time()

    for spec in selected:
        logging.info("=" * 72)
        logging.info("MODEL: %s  (base=%s)", spec.short, spec.base_model)
        logging.info("=" * 72)
        per_model_start = time.time()
        per_started_iso = datetime.now(timezone.utc).isoformat()
        record = {"model": spec.short, "base": spec.base_model}
        try:
            wipe_student()
            train_one(spec)
            export_quants(quants)
            per_finished_iso = datetime.now(timezone.utc).isoformat()
            write_model_card(spec, quants, per_started_iso, per_finished_iso)
            if args.skip_push:
                record["status"] = "trained-only"
                record["hf_url"] = None
            else:
                record["hf_url"] = push_to_hf(spec, args.hf_namespace)
                record["status"] = "published"
        except Exception as e:  # noqa: BLE001
            record["status"] = f"failed: {e}"
            logging.exception("Model %s failed", spec.short)
            if not args.keep_on_failure:
                summary.append(
                    {**record, "duration": fmt_duration(time.time() - per_model_start)}
                )
                _print_summary(summary, time.time() - overall_start)
                return 1
        record["duration"] = fmt_duration(time.time() - per_model_start)
        summary.append(record)

    _print_summary(summary, time.time() - overall_start)
    return 0 if all(r["status"] in {"published", "trained-only"} for r in summary) else 1


def _print_summary(summary: list[dict], total_seconds: float) -> None:
    logging.info("=" * 72)
    logging.info("RUN SUMMARY  (total %s)", fmt_duration(total_seconds))
    logging.info("=" * 72)
    for r in summary:
        logging.info(
            "  %-18s %-12s %-40s %s",
            r["model"], r["duration"], r["status"],
            r.get("hf_url") or "",
        )
    # Also dump as JSON next to the log for downstream automation.
    report_path = LOGS_DIR / f"train_and_publish_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Summary JSON: %s", report_path)


if __name__ == "__main__":
    sys.exit(main())
