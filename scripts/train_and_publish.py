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
# 06_train_student.py and export_gguf.py BOTH hardcode this path. We train
# into it, then RENAME to a per-model dir (models/student-<short>/) so the
# artifacts coexist after the run and a failure in model N doesn't clobber
# the published bits from model N-1.
STUDENT_DIR = REPO_ROOT / "models" / "student"
LOGS_DIR = REPO_ROOT / "logs"
TRAIN_FILE = REPO_ROOT / "data" / "distill" / "train.jsonl"
EVAL_FILE = REPO_ROOT / "data" / "distill" / "eval.jsonl"

DEFAULT_HF_NAMESPACE = "kartikey31"
DEFAULT_HF_REPO_NAME = "txn-parser"   # single repo; per-model artifacts go in subfolders
DEFAULT_QUANTS = ["q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"]


def final_dir_for(short: str) -> Path:
    """Per-model output dir, e.g. models/student-gemma-3-270m/."""
    return REPO_ROOT / "models" / f"student-{short}"


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


def wipe_student(also: Path | None = None) -> None:
    """Clean slate before each training run so adapters/checkpoints/gguf
    from the previous model don't bleed into this one.

    `also` lets us wipe the per-model final dir too if it exists (e.g. from
    a prior aborted run of the same model)."""
    if STUDENT_DIR.exists():
        logging.info("Wiping prior %s", STUDENT_DIR)
        shutil.rmtree(STUDENT_DIR)
    if also is not None and also.exists():
        logging.info("Wiping prior %s", also)
        shutil.rmtree(also)


def promote_to_final(short: str) -> Path:
    """Rename models/student/ -> models/student-<short>/ and return the
    new path. Called AFTER train+export+readme so the renamed dir contains
    a complete deliverable."""
    final = final_dir_for(short)
    if final.exists():
        shutil.rmtree(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(STUDENT_DIR), str(final))
    logging.info("Promoted student/ -> %s", final)
    return final


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


def normalize_gguf_names(short: str, quants: list[str]) -> list[str]:
    """Rename freshly-exported GGUFs to txn-parser-<short>-<QUANT>.gguf so
    the per-base-model artifacts have predictable, self-documenting names
    even after the per-model folder structure on HF (the file basename
    alone tells you the base model + quant).

    Returns the list of final filenames. Mutates files in place.
    """
    gguf_dir = STUDENT_DIR / "gguf"
    if not gguf_dir.exists():
        return []
    # Map upper-case quant tokens we expect (e.g. "Q4_K_M", "F16") back to
    # the requested forms so we can detect which quant each output file is.
    expected = {q.upper(): q for q in quants}
    final_names: list[str] = []
    for src in sorted(gguf_dir.glob("*.gguf")):
        upper_stem = src.stem.upper()
        matched = next((q for q in expected if q in upper_stem), None)
        if matched is None:
            logging.warning(
                "GGUF %s did not match any requested quant (%s) — skipping rename",
                src.name, ",".join(expected),
            )
            final_names.append(src.name)
            continue
        new_name = f"txn-parser-{short}-{matched}.gguf"
        dst = gguf_dir / new_name
        if dst == src:
            final_names.append(src.name)
            continue
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        logging.info("Renamed %s -> %s", src.name, new_name)
        final_names.append(new_name)
    return final_names


def write_model_card(spec: ModelSpec, quants: list[str], train_started: str,
                     train_finished: str, repo_name: str, namespace: str) -> Path:
    """Drop a README.md into STUDENT_DIR with provenance + usage notes."""
    train_count = _line_count(TRAIN_FILE)
    eval_count = _line_count(EVAL_FILE)
    gguf_dir = STUDENT_DIR / "gguf"
    repo_id = f"{namespace}/{repo_name}"
    gguf_listing = "\n".join(
        f"  - [`{spec.short}/gguf/{p.name}`](https://huggingface.co/{repo_id}/resolve/main/{spec.short}/gguf/{p.name})"
        f"  ({p.stat().st_size / 1e6:.1f} MB)"
        for p in sorted(gguf_dir.glob("*.gguf"))
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

# txn-parser / {spec.short}

QLoRA fine-tune of [`{spec.base_model}`]({_hf_link(spec.base_model)}) for
extracting structured transaction data (amount, currency, item, category,
type) from free-form Indian-English / code-switched speech and text.

This model lives in subfolder **`{spec.short}/`** of the
[`{repo_id}`](https://huggingface.co/{repo_id}) repo, alongside
sibling fine-tunes of other base models trained on the same data.

## What's in here

- `{spec.short}/adapters/` — PEFT LoRA adapter (rank 32). Load on top of the base model
  with `peft.PeftModel.from_pretrained(base, "{repo_id}", subfolder="{spec.short}/adapters")`.
- `{spec.short}/gguf/` — merged GGUF builds at multiple quantization levels
  (file names follow `txn-parser-{spec.short}-<QUANT>.gguf`):
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

## Download a single GGUF

```bash
huggingface-cli download {repo_id} \\
    {spec.short}/gguf/txn-parser-{spec.short}-Q4_K_M.gguf \\
    --local-dir .
```

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


def validate_hf_auth(namespace: str) -> None:
    """Fail fast if HF credentials can't push to <namespace>/*.

    Run BEFORE any training so the user finds out about a bad/missing
    token in seconds instead of after a multi-hour A100 run. Checks (in
    order):

      1. `huggingface_hub` is installed.
      2. `whoami()` succeeds with whatever token is reachable
         (HF_TOKEN env var OR cached token from `huggingface-cli login`).
      3. The configured namespace is either the authenticated user OR an
         org the user belongs to.
      4. The token reports 'write' role when that field is available
         (fine-grained tokens may not expose it — we warn rather than
         fail in that case).
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit(
            f"huggingface_hub not installed: {e}. "
            "Run `pip install huggingface_hub` or rerun setup.sh."
        )

    api = HfApi()
    try:
        info = api.whoami()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Hugging Face auth check FAILED — cannot reach whoami endpoint "
            f"with the available credentials.\n  Underlying error: {e}\n"
            "Fix one of:\n"
            "  - export HF_TOKEN=hf_xxx   (token with write scope)\n"
            "  - huggingface-cli login    (interactive, stores in ~/.cache/huggingface)\n"
            "Or pass --skip-push to train without publishing."
        )

    user = info.get("name") or info.get("email") or "<unknown>"
    orgs = [o.get("name") for o in (info.get("orgs") or []) if o.get("name")]
    if namespace == user:
        logging.info("HF auth OK: user '%s' (matches --hf-namespace).", user)
    elif namespace in orgs:
        logging.info(
            "HF auth OK: user '%s' has access to org '%s'.", user, namespace,
        )
    else:
        raise SystemExit(
            f"HF auth ok for user '{user}', but the configured namespace "
            f"'{namespace}' is neither that user nor one of their orgs "
            f"({orgs or 'none'}). Either:\n"
            f"  - pass --hf-namespace {user}\n"
            f"  - or use a token from an account that owns '{namespace}'."
        )

    # Best-effort write-scope check. Classic tokens expose role; fine-grained
    # tokens don't — for those we'd have to attempt an actual write, which
    # we don't want to do for a noop preflight.
    auth = info.get("auth") or {}
    access = auth.get("accessToken") or {}
    role = access.get("role")
    if role == "write":
        logging.info("HF token role: write — preflight complete.")
    elif role == "read":
        raise SystemExit(
            "HF token has READ scope only. Generate a write token at "
            "https://huggingface.co/settings/tokens and re-export HF_TOKEN."
        )
    elif role:
        logging.info("HF token role: %r — assuming push will work.", role)
    else:
        logging.info(
            "HF token role not reported (likely fine-grained token) — "
            "preflight skipped the role check. The first create_repo / "
            "upload_folder call will surface any permission issue."
        )


def push_to_hf(spec: ModelSpec, folder: Path, namespace: str, repo_name: str) -> str:
    """Create the single shared HF repo if missing, then upload `folder`
    (adapters/, gguf/, README) into the per-base-model subfolder
    `<short>/` inside that repo.

    Returns the URL to the per-model subfolder (browseable on HF).
    Requires HF_TOKEN env var (or prior `huggingface-cli login`)."""
    from huggingface_hub import HfApi, create_repo

    repo_id = f"{namespace}/{repo_name}"
    api = HfApi()

    create_repo(repo_id, exist_ok=True, repo_type="model")
    logging.info(
        "Uploading %s -> https://huggingface.co/%s/tree/main/%s",
        folder, repo_id, spec.short,
    )

    # Skip the Trainer checkpoints (huge, can't be used directly) and any
    # transient training artifacts. Adapters + GGUFs + README is what users
    # actually need. path_in_repo segregates each base model's artifacts
    # so they coexist in one repo without name collisions.
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=spec.short,
        commit_message=(
            f"[{spec.short}] auto-publish: base={spec.base_model} "
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
    return f"https://huggingface.co/{repo_id}/tree/main/{spec.short}"


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
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO_NAME,
                   help=f"HF repo name (not the full id). Default: {DEFAULT_HF_REPO_NAME}. "
                        f"All base models go into ONE repo, each in its own "
                        f"`<short>/` subfolder.")
    p.add_argument("--quants", default=",".join(DEFAULT_QUANTS),
                   help=f"Comma-separated GGUF quants. Default: {','.join(DEFAULT_QUANTS)}")
    p.add_argument("--batch", type=int, default=0,
                   help="Override per-device training batch size for ALL "
                        "models in the run. 0 = keep each model's tuned default "
                        "(see MODELS table). Useful when you're on a smaller GPU "
                        "than the A100 the defaults target.")
    p.add_argument("--eval-batch", type=int, default=0,
                   help="Override eval batch size for ALL models. 0 = per-model "
                        "default. Drop this to 2 on a 40GB A100 to avoid OOM at "
                        "eval boundaries on Gemma's 256k-vocab logits.")
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

    # Apply CLI overrides to the selected specs. We mutate copies rather
    # than the module-level MODELS so a downstream --only re-run sees the
    # original defaults.
    if args.batch or args.eval_batch:
        from dataclasses import replace
        selected = [
            replace(
                s,
                batch_size=args.batch or s.batch_size,
                eval_batch_size=args.eval_batch or s.eval_batch_size,
            )
            for s in selected
        ]
        logging.info(
            "CLI override: batch_size=%s eval_batch_size=%s (0 = keep per-model default)",
            args.batch or "default", args.eval_batch or "default",
        )

    # HF preflight runs BEFORE training so an invalid/missing token fails
    # the run in seconds instead of after hours of GPU time.
    if not args.skip_push:
        if not os.environ.get("HF_TOKEN"):
            logging.info(
                "HF_TOKEN env var not set — will try cached token from "
                "`huggingface-cli login` if present."
            )
        validate_hf_auth(args.hf_namespace)

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
            wipe_student(also=final_dir_for(spec.short))
            train_one(spec)
            export_quants(quants)
            normalize_gguf_names(spec.short, quants)
            per_finished_iso = datetime.now(timezone.utc).isoformat()
            write_model_card(
                spec, quants, per_started_iso, per_finished_iso,
                repo_name=args.hf_repo, namespace=args.hf_namespace,
            )
            # Rename models/student/ -> models/student-<short>/ so the
            # artifacts coexist across the run AND the published folder
            # has a self-documenting name on disk.
            final_dir = promote_to_final(spec.short)
            record["local_dir"] = str(final_dir.relative_to(REPO_ROOT))
            if args.skip_push:
                record["status"] = "trained-only"
                record["hf_url"] = None
            else:
                record["hf_url"] = push_to_hf(
                    spec, final_dir, args.hf_namespace, args.hf_repo,
                )
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
    logging.info("=" * 78)
    logging.info("RUN SUMMARY  (total %s)", fmt_duration(total_seconds))
    logging.info("=" * 78)
    for r in summary:
        logging.info(
            "  %-18s %-10s %-30s local=%s hf=%s",
            r["model"], r["duration"], r["status"],
            r.get("local_dir") or "—",
            r.get("hf_url") or "—",
        )
    # Also dump as JSON next to the log for downstream automation.
    report_path = LOGS_DIR / f"train_and_publish_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Summary JSON: %s", report_path)


if __name__ == "__main__":
    sys.exit(main())
