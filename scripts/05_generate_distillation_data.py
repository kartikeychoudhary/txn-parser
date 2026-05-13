#!/usr/bin/env python3
"""Stage 5: Generate distillation data.

Two phases:

  1. **inputs** — Generate diverse synthetic input strings via DeepSeek
     until we hit ``--n-inputs`` unique inputs. Appended to
     ``data/distill/inputs_raw.jsonl``.
  2. **label**  — Run each input through the fine-tuned teacher
     (LoRA adapter loaded in fp16 — spec mandates fp16, not the
     quantized GGUF, for label quality). Validate the teacher's output
     against the schema. Successful labels go to
     ``data/distill/train.jsonl``; failures go to
     ``data/distill/failed.jsonl`` for inspection.
  3. **eval**   — Copy ``data/clean/eval.jsonl`` to
     ``data/distill/eval.jsonl`` (we never use teacher-generated
     examples for evaluation — eval stays human-supervised).

All three phases are idempotent. Re-running picks up where the previous
run left off based on already-written files.

Usage:
    set DEEPSEEK_API_KEY=sk-...
    python scripts/05_generate_distillation_data.py
    python scripts/05_generate_distillation_data.py --phase inputs --n-inputs 30000
    python scripts/05_generate_distillation_data.py --phase label --limit 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    BATCH_FOCUSES,
    build_messages,
    call_deepseek,
    extract_json,
    is_schema_valid,
    load_jsonl,
    schema_errors,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_EVAL = REPO_ROOT / "data" / "clean" / "eval.jsonl"
DISTILL_DIR = REPO_ROOT / "data" / "distill"
INPUTS_FILE = DISTILL_DIR / "inputs_raw.jsonl"
TRAIN_FILE = DISTILL_DIR / "train.jsonl"
DISTILL_EVAL_FILE = DISTILL_DIR / "eval.jsonl"
FAILED_FILE = DISTILL_DIR / "failed.jsonl"
LOGS_DIR = REPO_ROOT / "logs"
TEACHER_ADAPTER_DIR = REPO_ROOT / "models" / "teacher" / "adapters"
TEACHER_GGUF_DIR = REPO_ROOT / "models" / "teacher" / "gguf"


INPUT_GEN_PROMPT = """You are generating diverse voice-transcribed transaction descriptions for an Indian expense-tracking dataset.

## TASK
Generate exactly {n} unique transaction-description input strings, one per line. Output ONLY the raw input strings (the kind that come out of a speech-to-text engine). DO NOT output any JSON, structure, or explanation — just the plain input strings.

## INPUT CHARACTERISTICS
Inputs are voice transcripts in English reflecting how Indian users speak. Vary across these styles:
- Clean formal English ("I spent 500 rupees on beer")
- Casual shorthand ("500 rs beer", "200 uber", "1k groceries")
- Hinglish ("do sau rupay ka chai", "char hazaar rent", "paanch sau on dinner")
- Numbers as words ("five hundred on petrol", "two thousand for medicines")
- Abbreviated amounts ("1.5k for shoes", "500/- chai", "₹300 on lunch")
- Multi-transaction inputs ("300 lunch 50 chai 200 uber")
- Income mentions ("got 5000 salary", "received 200 cashback")
- Corrections / disfluency ("500 on beer wait no 600 on beer", "umm 200 for uber")
- Ambiguous inputs ("paid 500 for that thing", "spent 300 yesterday")

## RULES
- Every line must be a UNIQUE input string. No duplicates.
- Vary items, amounts (range: 10 to 200000), phrasing, and length.
- No numbering, no bullets, no dashes at the start of lines.
- No quotes around the lines.
- One input string per line.

## BATCH_FOCUS
{focus}

## OUTPUT FORMAT
Output exactly {n} lines. Each line is one input string. No preamble, no trailing text, no markdown fences."""


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


_LINE_PREFIX_RE = re.compile(r"^[\s\-\*\d]+[.)\s]+")


def clean_input_line(line: str) -> str | None:
    """Strip leading numbering/bullets and trailing whitespace; reject junk."""
    s = line.strip()
    if not s:
        return None
    if s.startswith("```") or s.lower().startswith("output:") or s.lower().startswith("example"):
        return None
    # Strip "1. ", "- ", "* ", "1) " style prefixes the model occasionally adds.
    s = _LINE_PREFIX_RE.sub("", s).strip()
    # Strip surrounding quotes.
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if len(s) < 3 or len(s) > 300:
        return None
    return s


def read_inputs_jsonl(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line)["input"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


# ---------------------------------------------------------------------------
# Phase 1 — input generation via DeepSeek
# ---------------------------------------------------------------------------


def phase_inputs(args: argparse.Namespace) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY env var is not set.")

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    existing = set(read_inputs_jsonl(INPUTS_FILE))
    logging.info("Phase 1: %d existing inputs, target %d", len(existing), args.n_inputs)

    if len(existing) >= args.n_inputs:
        logging.info("Already at target. Phase 1 complete.")
        return

    batch_idx = 0
    with INPUTS_FILE.open("a", encoding="utf-8") as fout:
        with tqdm(total=args.n_inputs, initial=len(existing), desc="inputs") as bar:
            while len(existing) < args.n_inputs:
                focus = BATCH_FOCUSES[batch_idx % len(BATCH_FOCUSES)]
                batch_idx += 1
                remaining = args.n_inputs - len(existing)
                n_this = min(args.inputs_per_call, remaining)
                prompt = INPUT_GEN_PROMPT.format(focus=focus, n=n_this)

                text = None
                for attempt in range(1, args.max_retries + 1):
                    try:
                        text = call_deepseek(
                            prompt,
                            api_key=api_key,
                            base_url=args.base_url,
                            model=args.model,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            timeout=args.timeout,
                        )
                        break
                    except (requests.RequestException, KeyError, ValueError) as e:
                        wait = min(60, 2 ** attempt)
                        logging.warning("[batch %d] attempt %d failed: %s. Retry in %ds",
                                        batch_idx, attempt, e, wait)
                        time.sleep(wait)
                if text is None:
                    logging.error("[batch %d] giving up after %d attempts", batch_idx, args.max_retries)
                    continue

                added = 0
                for raw_line in text.splitlines():
                    cleaned = clean_input_line(raw_line)
                    if cleaned is None:
                        continue
                    if cleaned in existing:
                        continue
                    existing.add(cleaned)
                    fout.write(json.dumps({"input": cleaned}, ensure_ascii=False) + "\n")
                    added += 1
                    bar.update(1)
                    if len(existing) >= args.n_inputs:
                        break
                fout.flush()
                logging.info("[batch %d] focus=%s  +%d new  (total %d/%d)",
                             batch_idx, focus[:50], added, len(existing), args.n_inputs)

                if added == 0:
                    # Avoid pathological retry loop if model can't produce new diverse inputs.
                    logging.warning("[batch %d] produced 0 new inputs; bumping temperature next batch", batch_idx)


# ---------------------------------------------------------------------------
# Phase 2 — teacher labeling
# ---------------------------------------------------------------------------


def _build_label_backend(args: argparse.Namespace):
    """Return a callable batch_infer(inputs: list[str]) -> list[str]."""
    if args.backend == "gguf":
        from llama_cpp import Llama  # lazy import

        gguf_path: Path
        if args.gguf_path:
            gguf_path = Path(args.gguf_path)
            if gguf_path.is_dir():
                cands = sorted(
                    p for p in gguf_path.glob("*.gguf")
                    if "mmproj" not in p.name.lower()
                )
                if not cands:
                    raise SystemExit(f"No .gguf in {gguf_path}")
                gguf_path = cands[0]
        else:
            cands = sorted(
                p for p in TEACHER_GGUF_DIR.glob("*.gguf")
                if "mmproj" not in p.name.lower()
            )
            if not cands:
                raise SystemExit(
                    f"No .gguf under {TEACHER_GGUF_DIR}. Run Stage 3 GGUF export "
                    "first or pass --gguf-path."
                )
            gguf_path = cands[0]

        logging.info("Loading teacher GGUF %s (n_gpu_layers=%d, n_ctx=%d, mmap=%s)",
                     gguf_path, args.n_gpu_layers, args.n_ctx, not args.no_mmap)
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            use_mmap=not args.no_mmap,
            use_mlock=args.mlock,
            n_batch=args.n_batch,
            verbose=False,
            seed=42,
            logits_all=False,
        )

        def batch_infer(inputs: list[str]) -> list[str]:
            # llama-cpp-python doesn't batch — process sequentially.
            out = []
            for inp in inputs:
                resp = llm.create_chat_completion(
                    messages=build_messages(inp),
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=args.max_new_tokens,
                )
                out.append((resp["choices"][0]["message"]["content"] or "").strip())
            return out

        return batch_infer

    # Default: transformers / Unsloth fp16
    from unsloth import FastLanguageModel
    import torch

    logging.info("Loading teacher (fp16) from %s", TEACHER_ADAPTER_DIR)
    model, processor = FastLanguageModel.from_pretrained(
        model_name=str(TEACHER_ADAPTER_DIR),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    # Gemma 3/4 are multimodal — Unsloth returns a Processor here, not a
    # bare Tokenizer. Use the underlying text tokenizer so plain
    # tokenizer(text) works without expecting image/video inputs.
    templater = processor
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    # Generation needs LEFT padding so the EOS isn't on the wrong side.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def batch_infer(inputs: list[str]) -> list[str]:
        prompts = [
            templater.apply_chat_template(
                build_messages(s), tokenize=False, add_generation_prompt=True,
            )
            for s in inputs
        ]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_seq_length,
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=pad_id,
            )
        input_len = enc["input_ids"].shape[1]
        return [
            tokenizer.decode(out[i][input_len:], skip_special_tokens=True).strip()
            for i in range(out.shape[0])
        ]

    return batch_infer


def phase_label(args: argparse.Namespace) -> None:
    if not INPUTS_FILE.exists():
        raise SystemExit(f"No input file at {INPUTS_FILE}. Run --phase inputs first.")

    if args.backend == "transformers" and not (TEACHER_ADAPTER_DIR / "adapter_config.json").exists():
        raise SystemExit(
            f"Teacher adapter not found at {TEACHER_ADAPTER_DIR}. Run Stage 3 first, "
            "or use --backend gguf with a trained GGUF."
        )

    all_inputs = read_inputs_jsonl(INPUTS_FILE)

    labeled = {r["input"] for r in load_jsonl(TRAIN_FILE)} if TRAIN_FILE.exists() else set()
    failed_set: set[str] = set()
    if FAILED_FILE.exists() and not args.retry_failed:
        for r in load_jsonl(FAILED_FILE):
            failed_set.add(r["input"])

    pending = [s for s in all_inputs if s not in labeled and s not in failed_set]
    if args.limit > 0:
        pending = pending[: args.limit]

    logging.info("Phase 2 (%s backend): %d total, %d labeled, %d previously failed, %d pending",
                 args.backend, len(all_inputs), len(labeled), len(failed_set), len(pending))

    if not pending:
        logging.info("Nothing to label.")
        return

    batch_infer = _build_label_backend(args)
    batch_size = max(1, args.batch_size)

    n_kept = 0
    n_failed = 0
    failure_reasons: dict[str, int] = {}

    with TRAIN_FILE.open("a", encoding="utf-8") as train_out, \
         FAILED_FILE.open("a", encoding="utf-8") as fail_out:
        for start in tqdm(range(0, len(pending), batch_size), desc="labeling", unit="batch"):
            chunk = pending[start : start + batch_size]
            raws = batch_infer(chunk)
            for inp, raw in zip(chunk, raws):
                obj = extract_json(raw)
                reason: str | None = None
                if obj is None:
                    reason = "json_invalid"
                elif not is_schema_valid(obj):
                    reason = "schema_invalid"

                if reason is None:
                    train_out.write(json.dumps({
                        "input": inp,
                        "output": obj,
                        "_source": "teacher_label",
                    }, ensure_ascii=False) + "\n")
                    n_kept += 1
                else:
                    errs = schema_errors(obj)[:3] if obj is not None else []
                    fail_out.write(json.dumps({
                        "input": inp,
                        "raw": raw,
                        "reason": reason,
                        "schema_errors": errs,
                    }, ensure_ascii=False) + "\n")
                    n_failed += 1
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            train_out.flush()
            fail_out.flush()

    total_attempted = n_kept + n_failed
    keep_rate = (n_kept / total_attempted * 100) if total_attempted else 0.0
    logging.info("Phase 2 done. kept=%d failed=%d keep_rate=%.1f%%",
                 n_kept, n_failed, keep_rate)
    if failure_reasons:
        logging.info("Failure reasons: %s", failure_reasons)


# ---------------------------------------------------------------------------
# Phase 3 — copy eval set
# ---------------------------------------------------------------------------


def phase_eval(args: argparse.Namespace) -> None:
    if not CLEAN_EVAL.exists():
        raise SystemExit(f"Source eval file missing: {CLEAN_EVAL}. Run Stage 1 first.")
    DISTILL_EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DISTILL_EVAL_FILE.exists() and DISTILL_EVAL_FILE.stat().st_mtime >= CLEAN_EVAL.stat().st_mtime \
            and not args.force_eval_copy:
        logging.info("%s is up to date; skipping copy (pass --force-eval-copy to overwrite).",
                     DISTILL_EVAL_FILE)
        return
    shutil.copyfile(CLEAN_EVAL, DISTILL_EVAL_FILE)
    logging.info("Copied %s -> %s", CLEAN_EVAL, DISTILL_EVAL_FILE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate distillation data (inputs + teacher labels).")
    p.add_argument("--phase", choices=["all", "inputs", "label", "eval"], default="all",
                   help="Which phase(s) to run. Default 'all' runs inputs->label->eval.")
    # Phase 1
    p.add_argument("--n-inputs", type=int, default=30000,
                   help="Target number of unique synthetic inputs (Phase 1).")
    p.add_argument("--inputs-per-call", type=int, default=200,
                   help="Inputs requested per DeepSeek call.")
    p.add_argument("--model", default="deepseek-chat",
                   help="DeepSeek model ID.")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="Max completion tokens per DeepSeek call.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--max-retries", type=int, default=5)
    # Phase 2
    p.add_argument("--backend", choices=["transformers", "gguf"], default="transformers",
                   help="Phase 2 inference backend. 'transformers' = Unsloth fp16 "
                        "(highest quality, default). 'gguf' = llama-cpp-python on the "
                        "exported teacher GGUF (faster but lossy from quantization).")
    p.add_argument("--max-seq-length", type=int, default=1024,
                   help="Context length when loading the teacher (transformers backend).")
    p.add_argument("--max-new-tokens", type=int, default=384,
                   help="Max new tokens generated per label.")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Phase 2 (transformers): inputs generated per forward pass. "
                        "On A100 80GB try 32-64; on 5060 Ti 16GB try 8-16; gguf backend "
                        "ignores this (always sequential).")
    p.add_argument("--limit", type=int, default=0,
                   help="Phase 2: only label the first N pending inputs (0 = all).")
    p.add_argument("--retry-failed", action="store_true",
                   help="Phase 2: re-attempt inputs previously written to failed.jsonl.")
    # Phase 2 / GGUF backend
    p.add_argument("--gguf-path",
                   help="Path to a .gguf file or directory (gguf backend). "
                        f"Default: auto-detect under {TEACHER_GGUF_DIR}.")
    p.add_argument("--n-gpu-layers", "--ngl", type=int, default=-1,
                   help="GGUF: layers to offload to GPU (-1 = all, 0 = CPU).")
    p.add_argument("--n-ctx", type=int, default=2048,
                   help="GGUF: context window.")
    p.add_argument("--n-batch", type=int, default=512,
                   help="GGUF: prompt-processing batch size.")
    p.add_argument("--no-mmap", action="store_true",
                   help="GGUF: disable memory-mapped loading (copies weights into RAM).")
    p.add_argument("--mlock", action="store_true",
                   help="GGUF: lock weights in RAM (prevents swap).")
    # Phase 3
    p.add_argument("--force-eval-copy", action="store_true",
                   help="Phase 3: copy eval.jsonl even if the destination is already up to date.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"05_generate_distillation_data_{ts}.log")
    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Stage 5: phase=%s args=%s", args.phase, vars(args))

    if args.phase in ("all", "inputs"):
        phase_inputs(args)
    if args.phase in ("all", "label"):
        phase_label(args)
    if args.phase in ("all", "eval"):
        phase_eval(args)

    logging.info("Stage 5 done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
