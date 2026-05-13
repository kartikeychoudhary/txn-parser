#!/usr/bin/env python3
"""Re-export a trained LoRA adapter to GGUF without retraining.

Useful when:
  - Stage 3 / 6 finished training but GGUF export failed or got interrupted
  - You want to try a different quantization (e.g. q8_0 vs q4_k_m) on the
    same trained adapter
  - The previously-exported GGUF is broken (chat template mismatch, etc.)
    and you want to nuke + retry

Loads ``models/<role>/adapters/`` via Unsloth fp16, runs
``model.save_pretrained_gguf(...)`` with the requested quant, and then
consolidates the layout so ``models/<role>/gguf/`` ends up with only the
final ``*.gguf`` files (Unsloth scatters intermediates into a sibling
``<gguf_dir>_gguf/`` directory — see project_unsloth_gguf_layout memory).

Usage:
    python scripts/export_gguf.py --role student
    python scripts/export_gguf.py --role student --quant q8_0
    python scripts/export_gguf.py --role teacher --quant q3_k_m
    python scripts/export_gguf.py --role student --keep-existing  # don't wipe gguf/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
sys.path.insert(0, str(Path(__file__).resolve().parent))


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


def consolidate(gguf_dir: Path) -> None:
    """Move *.gguf from <gguf_dir>_gguf/ into gguf_dir/ and remove the staging dir."""
    sibling = gguf_dir.parent / f"{gguf_dir.name}_gguf"
    if not sibling.exists():
        return
    gguf_files = [f for f in sibling.iterdir() if f.is_file() and f.suffix.lower() == ".gguf"]
    if not gguf_files:
        return

    # Clear intermediate safetensors/configs from the target.
    for stale in list(gguf_dir.iterdir()):
        if stale.suffix.lower() == ".gguf":
            continue  # leave existing .gguf files in place
        try:
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
        except OSError as e:
            logging.warning("Could not remove intermediate %s: %s", stale, e)

    for f in gguf_files:
        target = gguf_dir / f.name
        if target.exists():
            target.unlink()
        shutil.move(str(f), str(target))

    try:
        shutil.rmtree(sibling)
    except OSError as e:
        logging.warning("Could not remove %s: %s", sibling, e)

    logging.info("Consolidated %d GGUF file(s) into %s", len(gguf_files), gguf_dir)


def main() -> int:
    p = argparse.ArgumentParser(description="Re-export an existing adapter to GGUF.")
    p.add_argument("--role", choices=["teacher", "student"], required=True,
                   help="Which model role to export.")
    p.add_argument("--quant", default=None,
                   help="GGUF quantization. Default: q3_k_m for teacher, q4_k_m for student. "
                        "Other useful options: q8_0 (less lossy), q5_k_m, bf16 (fp).")
    p.add_argument("--quants", default=None,
                   help="Comma-separated list of quants to export in one model load. "
                        "Example: bf16,q8_0,q5_k_m,q4_k_m,q3_k_m. Overrides --quant. "
                        "Implies --keep-existing across the list so files coexist.")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--keep-existing", action="store_true",
                   help="Don't wipe the existing gguf/ dir before re-exporting "
                        "(useful when you want both quants side-by-side).")
    args = p.parse_args()

    if args.quants:
        quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    else:
        quants = [args.quant or ("q3_k_m" if args.role == "teacher" else "q4_k_m")]
    adapters_dir = REPO_ROOT / "models" / args.role / "adapters"
    gguf_dir = REPO_ROOT / "models" / args.role / "gguf"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"export_gguf_{args.role}_{ts}.log")

    if not (adapters_dir / "adapter_config.json").exists():
        logging.error("No adapter at %s — run Stage 3 or 6 first.", adapters_dir)
        return 2

    logging.info("Exporting %s adapter -> GGUF (quants: %s)", args.role, ", ".join(quants))
    logging.info("Adapter: %s", adapters_dir)
    logging.info("Output : %s", gguf_dir)

    # Only wipe the gguf dir on the FIRST export if --keep-existing wasn't passed.
    # When exporting multiple quants in a single invocation, never wipe between
    # them — that's the whole point.
    if not args.keep_existing and not args.quants and gguf_dir.exists():
        for stale in list(gguf_dir.iterdir()):
            try:
                if stale.is_dir():
                    shutil.rmtree(stale)
                else:
                    stale.unlink()
            except OSError as e:
                logging.warning("Could not remove %s: %s", stale, e)
        # Also nuke the sibling staging dir from any prior failed run.
        sibling = gguf_dir.parent / f"{gguf_dir.name}_gguf"
        if sibling.exists():
            shutil.rmtree(sibling, ignore_errors=True)

    gguf_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — these pull in CUDA.
    from unsloth import FastLanguageModel  # noqa: E402

    logging.info("Loading adapter (fp16) once — same loaded model used for all quants")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapters_dir),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    failed_quants: list[str] = []
    for quant in quants:
        logging.info("=" * 60)
        logging.info("save_pretrained_gguf -> %s  (quant=%s)", gguf_dir, quant)
        try:
            model.save_pretrained_gguf(
                str(gguf_dir), tokenizer, quantization_method=quant,
            )
            consolidate(gguf_dir)
        except Exception as e:  # noqa: BLE001
            logging.error("GGUF export failed for quant=%s: %s", quant, e)
            failed_quants.append(quant)
            # Clean up any partial staging dir so it doesn't break the next quant.
            sibling = gguf_dir.parent / f"{gguf_dir.name}_gguf"
            if sibling.exists():
                shutil.rmtree(sibling, ignore_errors=True)
            continue

    # Confirm the result.
    out_files = sorted(p.name for p in gguf_dir.glob("*.gguf"))
    if out_files:
        logging.info("=" * 60)
        logging.info("Final GGUF files in %s:", gguf_dir)
        for f in out_files:
            sz = (gguf_dir / f).stat().st_size / 1e6
            logging.info("  %s  (%.1f MB)", f, sz)
    else:
        logging.warning("No .gguf files in %s after export — check the log above.", gguf_dir)
        return 1

    if failed_quants:
        logging.warning("Some quants failed: %s", ", ".join(failed_quants))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
