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
import concurrent.futures
import json
import logging
import math
import os
import queue
import re
import shutil
import sys
import threading
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
    clean_input_line,
    extract_json,
    load_jsonl,
    normalize_input,        # NEW in Slice 2
    parse_amounts,
    serialize_validation_result,
    validate_example,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_EVAL = REPO_ROOT / "data" / "clean" / "eval.jsonl"

# DISTILL_DIR can be overridden by tests via env var. ALL derived paths
# (INPUTS_FILE, TRAIN_FILE, ...) must be computed AFTER this assignment.
DISTILL_DIR = Path(os.environ.get("DISTILL_DIR_OVERRIDE", REPO_ROOT / "data" / "distill"))
INPUTS_FILE = DISTILL_DIR / "inputs_raw.jsonl"
TRAIN_FILE = DISTILL_DIR / "train.jsonl"
DISTILL_EVAL_FILE = DISTILL_DIR / "eval.jsonl"
FAILED_FILE = DISTILL_DIR / "failed.jsonl"
LOGS_DIR = REPO_ROOT / "logs"
TEACHER_ADAPTER_DIR = REPO_ROOT / "models" / "teacher" / "adapters"
TEACHER_GGUF_DIR = REPO_ROOT / "models" / "teacher" / "gguf"

# Slice 2: multi-provider input generation tunables.
MAX_CONSECUTIVE_EMPTY_BATCHES = 3
MAX_CONSECUTIVE_PROVIDER_ERRORS = 3
MAX_CONSECUTIVE_DUPLICATES_BEFORE_GIVEUP = 200


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


def read_inputs_jsonl(path: Path) -> list[str]:
    """Load existing input strings from a JSONL file.

    Malformed rows are skipped with a warning (per spec §6.1):
      - Bad JSON → bad_json counter
      - Non-object rows → bad_shape counter
      - Missing or non-string 'input' field → bad_shape counter
    """
    if not path.exists():
        return []
    out: list[str] = []
    bad_json = 0
    bad_shape = 0
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning("read_inputs_jsonl: %s:%d invalid JSON: %s",
                                path.name, line_no, e)
                bad_json += 1
                continue
            if not isinstance(obj, dict):
                logging.warning("read_inputs_jsonl: %s:%d row is not an object, skipping",
                                path.name, line_no)
                bad_shape += 1
                continue
            if "input" not in obj or not isinstance(obj["input"], str):
                logging.warning(
                    "read_inputs_jsonl: %s:%d missing or non-string 'input' field, skipping",
                    path.name, line_no,
                )
                bad_shape += 1
                continue
            out.append(obj["input"])
    if bad_json or bad_shape:
        logging.info(
            "read_inputs_jsonl: loaded %d inputs from %s; skipped bad_json=%d bad_shape=%d",
            len(out), path, bad_json, bad_shape,
        )
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
    if FAILED_FILE.exists():
        for r in load_jsonl(FAILED_FILE):
            reason = r.get("reason")
            if args.retry_failed:
                # Old behavior: re-attempt every failed row regardless of reason.
                continue
            if args.retry_validation_failed and reason == "validation_failed":
                continue
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
            try:
                raws = batch_infer(chunk)
                batch_error: str | None = None
            except Exception as e:  # noqa: BLE001
                logging.error("teacher batch failed at start=%d size=%d: %s",
                              start, len(chunk), e)
                raws = [""] * len(chunk)
                batch_error = repr(e)

            for inp, raw in zip(chunk, raws):
                candidates = parse_amounts(inp)
                cand_payload = [
                    {
                        "value": c.value, "raw": c.raw, "span": list(c.span),
                        "status": c.status, "source": c.source,
                        "currency_hint": c.currency_hint,
                    }
                    for c in candidates
                ]

                if batch_error is not None:
                    fail_out.write(json.dumps({
                        "input": inp,
                        "raw_output": None,
                        "candidates": cand_payload,
                        "reason": "teacher_error",
                        "error": batch_error,
                    }, ensure_ascii=False) + "\n")
                    n_failed += 1
                    failure_reasons["teacher_error"] = failure_reasons.get("teacher_error", 0) + 1
                    continue

                obj = extract_json(raw)
                if obj is None:
                    fail_out.write(json.dumps({
                        "input": inp,
                        "raw_output": raw,
                        "candidates": cand_payload,
                        "reason": "json_parse_failed",
                    }, ensure_ascii=False) + "\n")
                    n_failed += 1
                    failure_reasons["json_parse_failed"] = failure_reasons.get("json_parse_failed", 0) + 1
                    continue

                result = validate_example(inp, obj, mode="strict")
                if result.ok:
                    train_out.write(json.dumps({
                        "input": inp,
                        "output": obj,
                        "_source": "teacher_label",
                    }, ensure_ascii=False) + "\n")
                    n_kept += 1
                else:
                    fail_out.write(json.dumps({
                        "input": inp,
                        "raw_output": raw,
                        "parsed_output": obj,
                        "candidates": cand_payload,
                        "validation": serialize_validation_result(result),
                        "reason": "validation_failed",
                    }, ensure_ascii=False) + "\n")
                    n_failed += 1
                    failure_reasons["validation_failed"] = failure_reasons.get("validation_failed", 0) + 1
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
# Multi-provider input generation (Slice 2). Concurrency model:
#   - One shared ThreadPoolExecutor, sized to min(sum(threads), global_max).
#   - For each provider with quota > 0, submit min(threads, max(1, ceil(quota/batch_size)))
#     worker tasks.
#   - Workers produce (input_text, provider_name, model_id) tuples to a
#     BOUNDED queue (prevents racing far ahead of the writer and wasting
#     API calls when writer has reached target). queue.put uses timeout +
#     stop_event re-check so executor shutdown cannot hang.
#   - Main thread is the SOLE writer to inputs_raw.jsonl.
#   - Completion: main sets stop_event when accepted_total >= pending.
#
# Quota is a soft scheduling hint, not a strict per-provider cap. See
# spec §5.2 for the rationale.
#
# Duplicate-stall guard is GLOBAL: a flood of duplicates from any one
# provider can stop the whole run. Slice 4 metrics may add per-provider
# health tracking; for Slice 2, global guard is acceptable.
# ---------------------------------------------------------------------------

_focus_counters: dict[str, int] = {}
_focus_lock = threading.Lock()


def _next_focus(provider_name: str) -> str:
    """Deterministic round-robin over BATCH_FOCUSES, keyed per provider."""
    with _focus_lock:
        i = _focus_counters.get(provider_name, 0)
        _focus_counters[provider_name] = i + 1
    return BATCH_FOCUSES[i % len(BATCH_FOCUSES)]


def _build_input_prompt(n: int, batch_focus: str) -> str:
    """Format the existing INPUT_GEN_PROMPT with the requested count + focus."""
    return INPUT_GEN_PROMPT.format(n=n, focus=batch_focus)


def _provider_worker(
    *,
    provider,                             # LLMProvider
    pcfg,                                 # ProviderConfig
    batch_size: int,
    results_queue: "queue.Queue",
    stop_event: threading.Event,
) -> None:
    """Run one provider's batches until stop_event is set or local caps hit.

    Maintains separate failure streaks:
      - empty_response_streak: parser returned 0 cleaned lines
      - provider_error_streak: exception bubbled through retries
    """
    empty_streak = 0
    error_streak = 0
    while not stop_event.is_set():
        if empty_streak >= MAX_CONSECUTIVE_EMPTY_BATCHES:
            logging.warning(
                "Provider %s: %d empty batches in a row, worker exiting early.",
                pcfg.name, empty_streak,
            )
            return
        if error_streak >= MAX_CONSECUTIVE_PROVIDER_ERRORS:
            logging.warning(
                "Provider %s: %d API errors in a row, worker exiting early.",
                pcfg.name, error_streak,
            )
            return
        prompt = _build_input_prompt(batch_size, batch_focus=_next_focus(pcfg.name))
        try:
            lines = provider.generate_inputs(prompt, batch_size)
        except Exception as e:    # noqa: BLE001 — surface any provider failure as an error streak
            logging.error("Provider %s batch failed: %s", pcfg.name, e)
            error_streak += 1
            continue
        if not lines:
            empty_streak += 1
            continue
        empty_streak = 0
        error_streak = 0
        model_id = getattr(provider, "model", None)
        for line in lines:
            if stop_event.is_set():
                return
            # Bounded queue: if writer is behind, wait briefly and retry,
            # but always re-check stop_event so executor shutdown can't hang.
            while not stop_event.is_set():
                try:
                    results_queue.put((line, provider.name, model_id), timeout=0.5)
                    break
                except queue.Full:
                    continue


def phase_inputs_multi_provider(args: argparse.Namespace) -> None:
    """Slice 2 multi-provider input-generation loop.

    Reads cfg.input_generation.{target_inputs, batch_size, providers}.
    On resume, dedupes against existing inputs_raw.jsonl and only
    generates the remaining `pending` inputs.
    """
    from generation_config import load_generation_config
    from generation_orchestrator import allocate_quota
    from llm_providers import create_provider

    cfg = load_generation_config(args.provider_config)
    ig = cfg.input_generation
    if not ig.enabled:
        logging.info("Phase inputs (multi-provider): input_generation.enabled=False — nothing to do.")
        return

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    existing_inputs = read_inputs_jsonl(INPUTS_FILE)
    existing_normalized = {normalize_input(s) for s in existing_inputs}
    existing_unique = len(existing_normalized)
    pending = max(0, ig.target_inputs - existing_unique)
    logging.info(
        "Phase inputs (multi-provider): existing_unique=%d target=%d pending=%d",
        existing_unique, ig.target_inputs, pending,
    )
    if pending == 0:
        logging.info("Already at target. Multi-provider phase complete.")
        return

    if args.n_inputs and args.n_inputs != ig.target_inputs:
        logging.warning(
            "--n-inputs=%d ignored in multi-provider mode; "
            "cfg.input_generation.target_inputs=%d is the source of truth.",
            args.n_inputs, ig.target_inputs,
        )

    providers = {p.name: create_provider(p) for p in ig.providers}
    quota = allocate_quota(pending, ig.providers)
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    stop_event = threading.Event()
    futures: list[concurrent.futures.Future] = []

    # Worker count per provider: min(threads, max(1, ceil(quota/batch_size))).
    # Providers with zero quota get zero workers (no calls made).
    total_workers = 0
    for pcfg in ig.providers:
        q = quota[pcfg.name]
        if q <= 0:
            continue
        n_workers = min(pcfg.threads, max(1, math.ceil(q / ig.batch_size)))
        total_workers += n_workers
    pool_size = min(total_workers, cfg.rate_limits.global_max_workers) if total_workers else 1

    # Bounded queue: prevents workers racing far ahead of the writer and
    # making wasted API calls when the writer has already reached target.
    # Size = batch_size * total_workers * 2 gives one full batch of slack
    # per worker before put() blocks.
    queue_max = max(1, ig.batch_size * max(1, total_workers) * 2)
    results_queue: "queue.Queue[tuple[str, str, object]]" = queue.Queue(maxsize=queue_max)

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool, \
         INPUTS_FILE.open("a", encoding="utf-8") as fout, \
         tqdm(total=pending, initial=0, desc="multi-provider inputs") as bar:

        for pcfg in ig.providers:
            q = quota[pcfg.name]
            if q <= 0:
                continue
            n_workers = min(pcfg.threads, max(1, math.ceil(q / ig.batch_size)))
            for _ in range(n_workers):
                futures.append(pool.submit(
                    _provider_worker,
                    provider=providers[pcfg.name],
                    pcfg=pcfg,
                    batch_size=ig.batch_size,
                    results_queue=results_queue,
                    stop_event=stop_event,
                ))

        accepted_total = 0
        consecutive_dupes = 0
        while accepted_total < pending:
            # Exit if all workers finished AND no more queued results.
            if all(f.done() for f in futures) and results_queue.empty():
                logging.warning(
                    "Multi-provider input loop: providers exhausted before "
                    "reaching target (accepted=%d, pending=%d).",
                    accepted_total, pending,
                )
                break
            try:
                input_text, provider_name, model_id = results_queue.get(timeout=1.0)
            except queue.Empty:
                # Timeout — workers may still be in flight. Do NOT increment
                # the dupe counter; just loop.
                continue
            normalized = normalize_input(input_text)
            if normalized in existing_normalized:
                consecutive_dupes += 1
                if consecutive_dupes >= MAX_CONSECUTIVE_DUPLICATES_BEFORE_GIVEUP:
                    logging.warning(
                        "Multi-provider input loop saw %d consecutive duplicates "
                        "without accepting a new input — providers exhausted unique "
                        "supply. Stopping with partial progress (accepted=%d / pending=%d).",
                        consecutive_dupes, accepted_total, pending,
                    )
                    stop_event.set()
                    break
                continue
            consecutive_dupes = 0
            existing_normalized.add(normalized)
            row = {
                "input": input_text,
                "_source": "synthetic_input",
                "_provider": provider_name,
                "_model": model_id,    # may be None for FakeProvider
                "_batch_id": batch_id,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            accepted_total += 1
            if accepted_total % cfg.rate_limits.write_flush_every == 0:
                fout.flush()
            bar.update(1)

        stop_event.set()
        fout.flush()

    logging.info(
        "Phase inputs (multi-provider) done. accepted=%d existing_unique=%d target=%d",
        accepted_total, existing_unique + accepted_total, ig.target_inputs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
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
    p.add_argument("--retry-validation-failed", action="store_true",
                   help="Phase 2: re-attempt rows in failed.jsonl where "
                        "reason == 'validation_failed'. Analogous to --retry-failed "
                        "but scoped to semantic-validation failures.")
    # Slice 1: multi-provider scaffolding. Real execution lands in Slice 2/3.
    p.add_argument("--provider-config", type=Path, default=None,
                   help="Path to a multi-provider generation config JSON. "
                        "Use with --dry-run-quota to inspect the config, or with "
                        "--multi-provider --phase inputs (Slice 2) to run real "
                        "input generation.")
    p.add_argument("--dry-run-quota", action="store_true",
                   help="With --provider-config: parse + validate config, "
                        "print quota allocations and scheduler preview, exit 0. "
                        "No generation runs.")
    p.add_argument("--multi-provider", action="store_true",
                   help="Run multi-provider execution. Slice 2 supports "
                        "--phase inputs only; label and all exit via parser.error.")
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
    return p, p.parse_args()


def _enforce_flag_contract(
    args: argparse.Namespace, parser: argparse.ArgumentParser,
) -> None:
    """Multi-provider CLI contract — see spec §7.

    Every error path uses parser.error which exits with code 2.
    Dry-run wins: phase restrictions are skipped when --dry-run-quota is present.
    """
    # Standalone-flag errors.
    if args.multi_provider and not args.provider_config:
        parser.error("--multi-provider requires --provider-config.")
    if args.dry_run_quota and not args.provider_config:
        parser.error("--dry-run-quota requires --provider-config.")
    # --provider-config requires SOMETHING (either dry-run or multi-provider).
    if args.provider_config and not (args.dry_run_quota or args.multi_provider):
        parser.error(
            "--provider-config requires --dry-run-quota or --multi-provider."
        )
    # Multi-provider phase restrictions for Slice 2.
    # Dry-run wins: phase checks skipped when --dry-run-quota is present.
    if args.multi_provider and args.provider_config and not args.dry_run_quota:
        if args.phase == "label":
            parser.error(
                "--multi-provider --phase label is reserved for Slice 3 "
                "(validator-gated output labeling)."
            )
        if args.phase == "eval":
            parser.error(
                "--multi-provider --phase eval is not supported; "
                "use legacy mode (no --multi-provider) for eval."
            )
        if args.phase == "all":
            parser.error(
                "--multi-provider does not support --phase all in Slice 2; "
                "specify --phase inputs, or use legacy mode (no --multi-provider)."
            )
        # --phase inputs is the only allowed combo; falls through to main().


def _run_dry_run(config_path: Path) -> int:
    """Load config + print render_dry_run output. Exit 0 on success, 2 on bad config."""
    from generation_config import ConfigError, load_generation_config
    from generation_orchestrator import render_dry_run

    try:
        cfg = load_generation_config(config_path)
    except ConfigError as e:
        logging.error("Invalid provider config: %s", e)
        print(f"Invalid provider config: {e}", file=sys.stderr)
        return 2

    logging.info("Loaded config from %s (version %d)", config_path, cfg.version)
    print(render_dry_run(cfg))
    return 0


def main() -> int:
    parser, args = parse_args()
    _enforce_flag_contract(args, parser)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"05_generate_distillation_data_{ts}.log")

    if args.dry_run_quota:
        assert args.provider_config is not None
        return _run_dry_run(args.provider_config)

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Stage 5: phase=%s args=%s", args.phase, vars(args))

    if args.multi_provider:
        # Slice 2: only --phase inputs reaches here (others rejected by contract).
        assert args.phase == "inputs"
        assert args.provider_config is not None
        phase_inputs_multi_provider(args)
        logging.info("Stage 5 done.")
        return 0

    # Legacy path — unchanged.
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
