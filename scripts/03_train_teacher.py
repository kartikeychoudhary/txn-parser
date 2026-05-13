#!/usr/bin/env python3
"""Stage 3: Fine-tune the teacher model (Gemma 4 E2B) with Unsloth + QLoRA.

Thin wrapper around ``_training.run_training``; see ``_training.py`` for
the actual training loop. Outputs:

  models/teacher/adapters/        LoRA adapter (mergeable later)
  models/teacher/gguf/            merged + Q3_K_M-quantized GGUF
  models/teacher/checkpoints/     intermediate trainer checkpoints (last 2)

Usage:
    python scripts/03_train_teacher.py
    python scripts/03_train_teacher.py --skip-gguf            # fast iteration
    python scripts/03_train_teacher.py --resume               # continue
    python scripts/03_train_teacher.py --max-steps 20         # smoke test
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _training import TrainingConfig, run_training  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"


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
        description="Fine-tune the teacher (Gemma 4 E2B) with Unsloth + QLoRA.",
    )
    p.add_argument("--model", default="unsloth/gemma-4-E2B-it",
                   help="Base model. Default: unsloth/gemma-4-E2B-it (pre-quantized 4-bit).")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-device training batch size. 5060 Ti 16GB: 4-8; "
                        "A100 80GB: 16-32. effective_bs = batch_size * grad_accum.")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Bump to keep effective batch the same when VRAM is tight.")
    p.add_argument("--eval-batch-size", type=int, default=8,
                   help="Per-device eval batch size. Keep small (4-16) even when "
                        "train batch is huge — Trainer materializes fp32 logits and "
                        "will OOM otherwise.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gguf-quant", default="q3_k_m",
                   help="GGUF quantization (spec calls for q3_k_m).")
    p.add_argument("--skip-gguf", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"03_train_teacher_{ts}.log")
    logging.info("Stage 3: training teacher model")

    cfg = TrainingConfig(
        model_name=args.model,
        output_dir=REPO_ROOT / "models" / "teacher",
        train_file=REPO_ROOT / "data" / "clean" / "train.jsonl",
        eval_file=REPO_ROOT / "data" / "clean" / "eval.jsonl",
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        eval_batch_size=args.eval_batch_size,
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
    return run_training(cfg)


if __name__ == "__main__":
    sys.exit(main())
