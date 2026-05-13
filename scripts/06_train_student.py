#!/usr/bin/env python3
"""Stage 6: Fine-tune the student model (Gemma 3 270M) on teacher-labeled data.

Thin wrapper around ``_training.run_training``; differs from Stage 3 in:

  - Base model:    google/gemma-3-270m-it  (much smaller)
  - Training data: data/distill/train.jsonl  (teacher-labeled, ~30k examples)
  - LoRA rank:     32  (higher than teacher — smaller base needs more capacity)
  - Epochs:        2   (more data, fewer epochs to avoid overfit)
  - GGUF quant:    Q4_K_M  (this is what ships to Android)

After training, runs ``scripts/04_eval.py`` against both teacher and student
GGUFs and prints a side-by-side comparison table.

Usage:
    python scripts/06_train_student.py
    python scripts/06_train_student.py --skip-gguf
    python scripts/06_train_student.py --skip-comparison     # train only
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _training import TrainingConfig, run_training  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
EVAL_RESULTS_DIR = REPO_ROOT / "eval_results"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "04_eval.py"


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune the student (Gemma 3 270M) with Unsloth + QLoRA.",
    )
    p.add_argument("--model", default="unsloth/gemma-3-270m-it",
                   help="Base model. Default: unsloth/gemma-3-270m-it. "
                        "Fall back to google/gemma-3-270m-it if unsloth doesn't host it.")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=2.0,
                   help="Spec: 2 epochs on the larger teacher-labeled set.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Student is much smaller — bigger batch fits.")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=32,
                   help="Spec: r=32 for the student (more capacity).")
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gguf-quant", default="q4_k_m",
                   help="GGUF quantization (spec calls for q4_k_m for the student).")
    p.add_argument("--skip-gguf", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--skip-comparison", action="store_true",
                   help="Skip the teacher-vs-student eval pass after training.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Post-training comparison
# ---------------------------------------------------------------------------


def _first_gguf(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    matches = sorted(directory.glob("*.gguf"))
    return matches[0] if matches else None


def _run_eval(gguf_path: Path, name: str | None = None) -> bool:
    cmd = [sys.executable, str(EVAL_SCRIPT), "--model", str(gguf_path)]
    if name:
        cmd += ["--name", name]
    logging.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logging.warning("Eval failed for %s (rc=%d)", gguf_path, result.returncode)
        return False
    return True


def _summarize_results(path: Path) -> dict | None:
    if not path.exists():
        return None
    n = n_json = n_schema = n_exact = 0
    lat_sum = 0.0
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            if r.get("json_valid"):
                n_json += 1
            if r.get("schema_valid"):
                n_schema += 1
            if r.get("exact_match"):
                n_exact += 1
            lat_sum += float(r.get("latency_ms") or 0)
    if n == 0:
        return None
    return {
        "n": n,
        "json_valid_pct": n_json / n * 100,
        "schema_valid_pct": n_schema / n * 100,
        "exact_match_pct": n_exact / n * 100,
        "mean_latency_ms": lat_sum / n,
    }


def _print_table(teacher: dict, student: dict, teacher_name: str, student_name: str) -> None:
    bar = "=" * 64
    logging.info(bar)
    logging.info("TEACHER vs STUDENT")
    logging.info(bar)
    header = f"{'Metric':<18} {'Teacher':>18} {'Student':>18}  {'Δ':>6}"
    logging.info(header)
    logging.info("-" * len(header))
    rows = [
        ("Eval examples",   f"{teacher['n']}",                       f"{student['n']}",                       ""),
        ("JSON valid",      f"{teacher['json_valid_pct']:.1f}%",     f"{student['json_valid_pct']:.1f}%",     f"{student['json_valid_pct'] - teacher['json_valid_pct']:+.1f}"),
        ("Schema valid",    f"{teacher['schema_valid_pct']:.1f}%",   f"{student['schema_valid_pct']:.1f}%",   f"{student['schema_valid_pct'] - teacher['schema_valid_pct']:+.1f}"),
        ("Exact match",     f"{teacher['exact_match_pct']:.1f}%",    f"{student['exact_match_pct']:.1f}%",    f"{student['exact_match_pct'] - teacher['exact_match_pct']:+.1f}"),
        ("Mean latency ms", f"{teacher['mean_latency_ms']:.1f}",     f"{student['mean_latency_ms']:.1f}",     f"{student['mean_latency_ms'] - teacher['mean_latency_ms']:+.1f}"),
    ]
    for label, t, s, d in rows:
        logging.info(f"{label:<18} {t:>18} {s:>18}  {d:>6}")
    logging.info(bar)
    logging.info("Targets — Teacher: >98%% JSON / >97%% schema / >85%% exact")
    logging.info("Targets — Student: >97%% JSON / >95%% schema / >80%% exact")
    logging.info("Eval results: %s/{%s,%s}.jsonl", EVAL_RESULTS_DIR, teacher_name, student_name)


def run_comparison() -> None:
    teacher_gguf = _first_gguf(REPO_ROOT / "models" / "teacher" / "gguf")
    student_gguf = _first_gguf(REPO_ROOT / "models" / "student" / "gguf")

    if teacher_gguf is None:
        logging.warning("No teacher GGUF found — skipping comparison. "
                        "Run Stage 3 (or pass --skip-gguf=False) to produce one.")
        return
    if student_gguf is None:
        logging.warning("No student GGUF found — skipping comparison.")
        return

    logging.info("Evaluating teacher (%s) and student (%s) on data/clean/eval.jsonl",
                 teacher_gguf.name, student_gguf.name)

    _run_eval(teacher_gguf)
    _run_eval(student_gguf)

    teacher_name = teacher_gguf.stem
    student_name = student_gguf.stem
    teacher_summary = _summarize_results(EVAL_RESULTS_DIR / f"{teacher_name}.jsonl")
    student_summary = _summarize_results(EVAL_RESULTS_DIR / f"{student_name}.jsonl")

    if teacher_summary is None or student_summary is None:
        logging.warning("Could not load eval results for comparison.")
        return

    _print_table(teacher_summary, student_summary, teacher_name, student_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"06_train_student_{ts}.log")
    logging.info("Stage 6: training student model")

    cfg = TrainingConfig(
        model_name=args.model,
        output_dir=REPO_ROOT / "models" / "student",
        train_file=REPO_ROOT / "data" / "distill" / "train.jsonl",
        eval_file=REPO_ROOT / "data" / "distill" / "eval.jsonl",
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        gguf_quant=args.gguf_quant,
        skip_gguf=args.skip_gguf,
        resume=args.resume,
        force=args.force,
    )

    rc = run_training(cfg)
    if rc != 0:
        return rc

    if args.skip_comparison:
        logging.info("--skip-comparison set; skipping teacher/student eval.")
        return 0

    run_comparison()
    return 0


if __name__ == "__main__":
    sys.exit(main())
