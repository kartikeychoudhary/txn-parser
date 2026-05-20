#!/usr/bin/env python3
"""Stage 1a: Generate raw dataset batches by calling the DeepSeek API.

Reads ``data_gen_prompt.md``, substitutes the ``BATCH_FOCUS`` section for
each of the 25 batches, and writes ``data/raw/batch_NN.jsonl``.

The script is idempotent: batches whose output file already exists are
skipped unless ``--force`` is passed.

Usage examples:
    set DEEPSEEK_API_KEY=sk-...
    python scripts/01_generate_dataset.py
    python scripts/01_generate_dataset.py --batches 5
    python scripts/01_generate_dataset.py --model deepseek-chat --max-tokens 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import BATCH_FOCUSES, call_deepseek  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILE = REPO_ROOT / "data_gen_prompt.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw"
LOGS_DIR = REPO_ROOT / "logs"

assert len(BATCH_FOCUSES) == 25, "Must define exactly 25 batch focuses"


def load_prompt_template() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


def render_prompt(template: str, focus: str) -> str:
    """Replace the BATCH_FOCUS placeholder block with the given focus string."""
    marker = "## BATCH_FOCUS"
    if marker not in template:
        raise ValueError(f"Prompt file missing '{marker}' section")
    head, _, tail = template.partition(marker)
    rest_lines = tail.splitlines()
    next_section_idx: int | None = None
    for i, line in enumerate(rest_lines[1:], start=1):
        if line.startswith("## "):
            next_section_idx = i
            break
    if next_section_idx is None:
        raise ValueError("Could not find a section after BATCH_FOCUS")
    after = "\n".join(rest_lines[next_section_idx:])
    return f"{head}{marker}\n{focus}\n\n{after}"


def parse_jsonl_response(text: str) -> tuple[list[dict], int]:
    """Parse the model's response into JSONL records. Returns (records, dropped)."""
    records: list[dict] = []
    dropped = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if not isinstance(obj, dict) or "input" not in obj or "output" not in obj:
            dropped += 1
            continue
        records.append(obj)
    return records, dropped


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate raw fine-tuning dataset batches via DeepSeek API."
    )
    parser.add_argument("--batches", type=int, default=25,
                        help="How many consecutive batches to run starting from --start-batch (1..25).")
    parser.add_argument("--start-batch", type=int, default=1,
                        help="1-indexed batch number to start from (1..25).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write batch_NN.jsonl files.")
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="DeepSeek model ID (e.g. deepseek-chat, deepseek-reasoner).")
    parser.add_argument("--base-url", default="https://api.deepseek.com",
                        help="DeepSeek-compatible API base URL.")
    parser.add_argument("--max-tokens", type=int, default=8000,
                        help="Max completion tokens per request.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. Higher = more diversity across batches.")
    parser.add_argument("--timeout", type=int, default=300,
                        help="HTTP timeout in seconds per request.")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Max retries on transient API errors with exponential backoff.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate batches even if their output files already exist.")
    args = parser.parse_args()

    if not 1 <= args.batches <= 25:
        parser.error("--batches must be between 1 and 25")
    if not 1 <= args.start_batch <= 25:
        parser.error("--start-batch must be between 1 and 25")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY env var is not set.", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"01_generate_dataset_{ts}.log")
    logging.info("Stage 1a: generating dataset batches")
    logging.info("Output dir: %s", args.output_dir)
    logging.info("Model: %s  base_url: %s", args.model, args.base_url)
    logging.info("max_tokens=%d  temperature=%s", args.max_tokens, args.temperature)

    template = load_prompt_template()
    end_batch = min(25, args.start_batch + args.batches - 1)
    batch_indices = list(range(args.start_batch, end_batch + 1))

    total_records = 0
    total_dropped = 0
    failed_batches: list[int] = []

    for batch_num in tqdm(batch_indices, desc="batches"):
        focus = BATCH_FOCUSES[batch_num - 1]
        out_path = args.output_dir / f"batch_{batch_num:02d}.jsonl"

        if out_path.exists() and not args.force:
            existing = sum(1 for _ in out_path.open(encoding="utf-8"))
            logging.info("[batch %02d] exists with %d records, skipping (use --force to overwrite)",
                         batch_num, existing)
            total_records += existing
            continue

        prompt = render_prompt(template, focus)

        records: list[dict] = []
        dropped = 0
        last_err: Exception | None = None
        success = False
        for attempt in range(1, args.max_retries + 1):
            try:
                logging.info("[batch %02d] attempt %d  focus=%s", batch_num, attempt, focus)
                text = call_deepseek(
                    prompt,
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                records, dropped = parse_jsonl_response(text)
                if not records:
                    raise RuntimeError("API returned 0 valid JSONL records")
                success = True
                break
            except (requests.RequestException, RuntimeError, KeyError, ValueError) as e:
                last_err = e
                wait = min(60, 2 ** attempt)
                logging.warning("[batch %02d] attempt %d failed: %s. Retrying in %ds",
                                batch_num, attempt, e, wait)
                time.sleep(wait)

        if not success:
            logging.error("[batch %02d] giving up after %d attempts: %s",
                          batch_num, args.max_retries, last_err)
            failed_batches.append(batch_num)
            continue

        write_jsonl(out_path, records)
        logging.info("[batch %02d] wrote %d records (%d parse-dropped) -> %s",
                     batch_num, len(records), dropped, out_path)
        total_records += len(records)
        total_dropped += dropped

    logging.info("=" * 60)
    logging.info("Done. Total records on disk: %d. Parse-dropped: %d.",
                 total_records, total_dropped)
    if failed_batches:
        logging.error("Failed batches (re-run to retry): %s", failed_batches)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
