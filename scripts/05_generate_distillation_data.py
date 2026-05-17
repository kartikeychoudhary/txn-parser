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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    BATCH_FOCUSES,
    build_messages,
    build_teacher_fp16_backend,
    call_deepseek,
    clean_input_line,
    extract_json,
    is_schema_valid,
    load_jsonl,
    normalize_input,        # NEW in Slice 2
    parse_amounts,
    schema_errors,
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
    from _lib import build_teacher_fp16_backend

    logging.info("Loading teacher (fp16) from %s", TEACHER_ADAPTER_DIR)
    backend = build_teacher_fp16_backend(
        adapter_dir=TEACHER_ADAPTER_DIR,
        max_seq_length=args.max_seq_length,
    )

    def batch_infer(inputs: list[str]) -> list[str]:
        out: list[str] = []
        for s in inputs:
            msgs = build_messages(s)
            out.append(backend.generate_label(msgs, max_new_tokens=args.max_new_tokens))
        return out

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
# Multi-provider OUTPUT labeling (Slice 3). Concurrency model:
#   - Per-input worker via ThreadPoolExecutor; pool size =
#     min(global_max_workers, len(pending)).
#   - Per-provider semaphore enforces pcfg.threads.
#   - Each worker: round-robin scheduler picks providers, calls generate_label,
#     validates, scores. Workers never write to JSONL.
#   - Main thread: as_completed loop reads worker results and writes rows to
#     train.jsonl or failed.jsonl. Sole writer.
#   - LocalTeacherProvider also serializes its own GPU calls via internal lock.
# ---------------------------------------------------------------------------


def _try_one_attempt(
    input_text: str,
    provider,
    pcfg,
    *,
    is_repair: bool = False,
    failure_summary: str = "",
    parser_candidates: list[dict] | None = None,
    provider_priority_rank: int | None = None,
    recorder=None,
    provider_type: str = "",
):
    """Run one provider attempt; validate; score. Never raises."""
    from label_selection import CandidateOutcome, score_candidate, build_repair_prompt
    from metrics import estimate_tokens
    import time

    start = time.perf_counter()
    if is_repair:
        prompt_text = build_repair_prompt(
            input_text,
            candidates=parser_candidates,
            failure_summary=failure_summary,
        )
    else:
        prompt_text = input_text
    try:
        if is_repair:
            raw = provider.generate_label(prompt_text)
        else:
            raw = provider.generate_label(input_text)
    except Exception as e:    # noqa: BLE001 — unify all provider failures
        latency_ms = (time.perf_counter() - start) * 1000.0
        pop = getattr(provider, "pop_last_usage", None)
        usage = pop() if pop else None
        if recorder is not None:
            if usage:
                pt, ct, est = usage["prompt_tokens"], usage["completion_tokens"], False
            else:
                pt = estimate_tokens(prompt_text)
                ct = 0
                est = True
            recorder.record_call(
                provider=pcfg.name,
                provider_type=provider_type,
                model=getattr(provider, "model", None),
                phase="label",
                attempt_type="repair" if is_repair else "initial",
                latency_ms=latency_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
                estimated_tokens=est,
                success=False,
                failure_reason="provider_error",
            )
        return CandidateOutcome(
            provider=pcfg.name,
            model=getattr(provider, "model", None),
            raw_output="",
            parsed_output=None,
            validation=None,
            failure_reason="provider_error",
            score=0,
            is_repair_attempt=is_repair,
            provider_priority_rank=provider_priority_rank,
            error=repr(e),
        )
    latency_ms = (time.perf_counter() - start) * 1000.0
    pop = getattr(provider, "pop_last_usage", None)
    usage = pop() if pop else None
    if recorder is not None:
        if usage:
            pt, ct, est = usage["prompt_tokens"], usage["completion_tokens"], False
        else:
            pt = estimate_tokens(prompt_text)
            ct = estimate_tokens(raw)
            est = True
        recorder.record_call(
            provider=pcfg.name,
            provider_type=provider_type,
            model=getattr(provider, "model", None),
            phase="label",
            attempt_type="repair" if is_repair else "initial",
            latency_ms=latency_ms,
            prompt_tokens=pt,
            completion_tokens=ct,
            estimated_tokens=est,
            success=True,
            failure_reason=None,
        )

    # Post-processing wrapped in a second try/except — a validator bug or
    # malformed parser output must not crash the entire executor pool.
    try:
        parsed = extract_json(raw)
        if parsed is None:
            return CandidateOutcome(
                provider=pcfg.name,
                model=getattr(provider, "model", None),
                raw_output=raw, parsed_output=None, validation=None,
                failure_reason="json_parse_failed",
                score=0, is_repair_attempt=is_repair,
                provider_priority_rank=provider_priority_rank,
            )

        result = validate_example(input_text, parsed, mode="strict")
        validation_dict = serialize_validation_result(result)

        error_codes = {
            e["code"] for e in validation_dict["errors"] if e["severity"] == "error"
        }
        if "SCHEMA_INVALID" in error_codes:
            failure_reason = "schema_invalid"
        elif "SUPERSEDED_AMOUNT_USED" in error_codes:
            failure_reason = "superseded_amount_used"
        elif "CURRENCY_MISMATCH" in error_codes:
            failure_reason = "currency_mismatch"
        elif "SUSPICIOUS_DUPLICATE" in error_codes:
            failure_reason = "suspicious_duplicate"
        elif error_codes:
            failure_reason = "validation_failed"
        else:
            failure_reason = None

        score = score_candidate(
            parsed_output=parsed,
            validation=validation_dict,
            failure_reason=failure_reason,
            provider_priority_rank=provider_priority_rank,
        )
        return CandidateOutcome(
            provider=pcfg.name,
            model=getattr(provider, "model", None),
            raw_output=raw,
            parsed_output=parsed,
            validation=validation_dict,
            failure_reason=failure_reason,
            score=score,
            is_repair_attempt=is_repair,
            provider_priority_rank=provider_priority_rank,
        )
    except Exception as e:    # noqa: BLE001 — post-processing should never bubble
        return CandidateOutcome(
            provider=pcfg.name,
            model=getattr(provider, "model", None),
            raw_output=raw,
            parsed_output=None,
            validation=None,
            failure_reason="validation_failed",
            score=0,
            is_repair_attempt=is_repair,
            provider_priority_rank=provider_priority_rank,
            error=repr(e),
        )


def _highest_priority_provider(cfg, providers_by_name: dict) -> str:
    """First entry of provider_priority, falling back to first declared provider."""
    if cfg.output_generation.provider_priority:
        name = cfg.output_generation.provider_priority[0]
    else:
        name = cfg.output_generation.providers[0].name
    if name not in providers_by_name:
        raise ValueError(
            f"_highest_priority_provider: {name!r} not in providers_by_name"
        )
    return name


def _summarize_failures(outcomes) -> str:
    """≤200-char summary of why all attempts failed."""
    parts = []
    for o in outcomes:
        if o.failure_reason == "provider_error":
            parts.append(f"{o.provider}: provider error")
        elif o.validation:
            codes = [e["code"] for e in o.validation["errors"] if e["severity"] == "error"]
            parts.append(f"{o.provider}: " + ", ".join(codes[:3]))
        elif o.failure_reason:
            parts.append(f"{o.provider}: {o.failure_reason}")
    return "; ".join(parts)[:200]


def _process_one_input(
    input_text: str,
    *,
    providers_by_name: dict,
    pcfgs_by_name: dict,
    scheduler,
    priority_map: dict,
    provider_limits: dict,           # name -> threading.Semaphore(pcfg.threads)
    cfg,
    parser_candidates: list[dict],
    recorder=None,
):
    """Process one input end-to-end. Returns (input_text, best_or_None, all_outcomes).

    Per-provider concurrency caps are honored via provider_limits semaphores.
    """
    from label_selection import pick_best

    outcomes = []

    for _ in range(cfg.output_generation.label_attempts_per_input):
        provider_name = scheduler.next_provider()
        with provider_limits[provider_name]:
            outcomes.append(_try_one_attempt(
                input_text,
                provider=providers_by_name[provider_name],
                pcfg=pcfgs_by_name[provider_name],
                provider_priority_rank=priority_map.get(provider_name),
                recorder=recorder,
                provider_type=getattr(pcfgs_by_name[provider_name], "provider_type", ""),
            ))

    best = pick_best(outcomes)
    if best is not None:
        return (input_text, best, outcomes)

    repair_enabled = (
        cfg.validation.retry_invalid_with_stricter_prompt
        and cfg.validation.max_repair_attempts > 0
    )
    if repair_enabled:
        repair_provider_name = _highest_priority_provider(cfg, providers_by_name)
        for _ in range(cfg.validation.max_repair_attempts):
            with provider_limits[repair_provider_name]:
                repair_outcome = _try_one_attempt(
                    input_text,
                    provider=providers_by_name[repair_provider_name],
                    pcfg=pcfgs_by_name[repair_provider_name],
                    is_repair=True,
                    failure_summary=_summarize_failures(outcomes),
                    parser_candidates=parser_candidates,
                    provider_priority_rank=priority_map.get(repair_provider_name),
                    recorder=recorder,
                    provider_type=getattr(pcfgs_by_name[repair_provider_name], "provider_type", ""),
                )
            outcomes.append(repair_outcome)
            if repair_outcome.failure_reason is None:
                return (input_text, repair_outcome, outcomes)

    return (input_text, None, outcomes)


def _read_labeled_inputs(path) -> set:
    """Set of inputs already present in train.jsonl. Tolerant of legacy rows."""
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("input"), str):
                out.add(obj["input"])
    return out


def _row_has_validation_failure_attempt(obj: dict) -> bool:
    validator_reasons = {
        "validation_failed", "schema_invalid", "superseded_amount_used",
        "currency_mismatch", "suspicious_duplicate",
    }
    for attempt in obj.get("attempts", []):
        if attempt.get("failure_reason") in validator_reasons:
            return True
    return False


def _read_failed_skip_set(path, args) -> set:
    """Inputs to SKIP this run based on failed.jsonl + retry flags."""
    if not path.exists():
        return set()
    if args.retry_failed:
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("input"), str):
                continue
            reason = obj.get("reason")
            if args.retry_validation_failed:
                if reason == "validation_failed":
                    continue
                if reason in {"all_candidates_failed", "repair_exhausted"} \
                        and _row_has_validation_failure_attempt(obj):
                    continue
            out.add(obj["input"])
    return out


def _build_priority_map(og) -> dict:
    if og.provider_priority:
        return {name: i for i, name in enumerate(og.provider_priority)}
    return {p.name: i for i, p in enumerate(og.providers)}


def _serialize_candidate(c) -> dict:
    return {
        "value": c.value, "raw": c.raw, "span": list(c.span),
        "status": c.status, "source": c.source, "currency_hint": c.currency_hint,
    }


def _serialize_outcome(o) -> dict:
    return {
        "provider": o.provider,
        "model": o.model,
        "raw_output": o.raw_output,
        "parsed_output": o.parsed_output,
        "validation": o.validation,
        "score": o.score,
        "failure_reason": o.failure_reason,
        "is_repair_attempt": o.is_repair_attempt,
        "provider_priority_rank": o.provider_priority_rank,
        "error": o.error,
    }


def phase_label_multi_provider(args: argparse.Namespace, recorder=None) -> int:
    """Slice 3 validator-gated multi-provider label generation."""
    from generation_config import load_generation_config
    from generation_orchestrator import RoundRobinScheduler
    from llm_providers import create_provider

    cfg = load_generation_config(args.provider_config)
    og = cfg.output_generation
    if not og.enabled:
        logging.info(
            "Phase label (multi-provider): output_generation.enabled=False — nothing to do."
        )
        return 0

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    all_inputs = read_inputs_jsonl(INPUTS_FILE)

    # Normalize-dedupe in first-seen order.
    seen_norm = set()
    all_inputs_unique = []
    for inp in all_inputs:
        key = normalize_input(inp)
        if key in seen_norm:
            continue
        seen_norm.add(key)
        all_inputs_unique.append(inp)

    labeled_keys = {normalize_input(s) for s in _read_labeled_inputs(TRAIN_FILE)}
    failed_skip_keys = {normalize_input(s) for s in _read_failed_skip_set(FAILED_FILE, args)}
    pending = [
        inp for inp in all_inputs_unique
        if normalize_input(inp) not in labeled_keys
        and normalize_input(inp) not in failed_skip_keys
    ]
    if args.limit > 0:
        pending = pending[:args.limit]
    logging.info(
        "Phase label (multi-provider): all_unique=%d labeled=%d failed_skip=%d pending=%d",
        len(all_inputs_unique), len(labeled_keys), len(failed_skip_keys), len(pending),
    )
    if not pending:
        return 0

    providers_by_name = {p.name: create_provider(p) for p in og.providers}
    pcfgs_by_name = {p.name: p for p in og.providers}
    scheduler = RoundRobinScheduler(og.providers)
    priority_map = _build_priority_map(og)
    # Per-provider concurrency caps honor pcfg.threads.
    provider_limits = {
        p.name: threading.Semaphore(p.threads) for p in og.providers
    }
    pool_size = min(cfg.rate_limits.global_max_workers, max(1, len(pending)))

    n_kept = 0
    n_failed = 0
    n_repaired = 0
    failure_reasons = Counter()
    attempt_failure_reasons = Counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool, \
         TRAIN_FILE.open("a", encoding="utf-8") as train_out, \
         FAILED_FILE.open("a", encoding="utf-8") as fail_out, \
         tqdm(total=len(pending), desc="multi-provider labels") as bar:

        candidates_by_input = {
            inp: [_serialize_candidate(c) for c in parse_amounts(inp)]
            for inp in pending
        }

        futures = {
            pool.submit(
                _process_one_input,
                inp,
                providers_by_name=providers_by_name,
                pcfgs_by_name=pcfgs_by_name,
                scheduler=scheduler,
                priority_map=priority_map,
                provider_limits=provider_limits,
                cfg=cfg,
                parser_candidates=candidates_by_input[inp],
                recorder=recorder,
            ): inp
            for inp in pending
        }

        for fut in concurrent.futures.as_completed(futures):
            input_text, best, all_outcomes = fut.result()
            for o in all_outcomes:
                if o.failure_reason:
                    attempt_failure_reasons[o.failure_reason] += 1

            # Emit label-candidate metrics for every outcome EXCEPT provider_error
            # (provider_error is already accounted for via record_call).
            if recorder is not None:
                for o in all_outcomes:
                    if o.failure_reason == "provider_error":
                        continue
                    recorder.record_label_candidate(
                        provider=o.provider,
                        model=o.model,
                        score=o.score,
                        accepted=(best is not None and o is best),
                        failure_reason=o.failure_reason,
                        is_repair_attempt=o.is_repair_attempt,
                    )

            if best is not None:
                row = {
                    "input": input_text,
                    "output": best.parsed_output,
                    "_source": "multi_provider_label",
                    "_provider": best.provider,
                    "_model": best.model,
                    "_validation_score": best.score,
                    "_attempts": len(all_outcomes),
                }
                train_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_kept += 1
                if best.is_repair_attempt:
                    n_repaired += 1
                if recorder is not None:
                    recorder.record_output_row("train")
            else:
                repair_enabled = (
                    cfg.validation.retry_invalid_with_stricter_prompt
                    and cfg.validation.max_repair_attempts > 0
                )
                if all(o.failure_reason == "provider_error" for o in all_outcomes):
                    top_reason = "provider_error"
                else:
                    top_reason = "repair_exhausted" if repair_enabled else "all_candidates_failed"
                failed_row = {
                    "input": input_text,
                    "reason": top_reason,
                    "candidates": candidates_by_input[input_text],
                    "attempts": [_serialize_outcome(o) for o in all_outcomes],
                }
                fail_out.write(json.dumps(failed_row, ensure_ascii=False) + "\n")
                n_failed += 1
                failure_reasons[top_reason] += 1
                if recorder is not None:
                    recorder.record_output_row("failed")
                    # Explicit repair-exhausted accounting: only count inputs
                    # that actually entered the repair loop.
                    if any(o.is_repair_attempt for o in all_outcomes):
                        recorder.record_repair_exhausted()
            if (n_kept + n_failed) % cfg.rate_limits.write_flush_every == 0:
                train_out.flush()
                fail_out.flush()
            bar.update(1)

        train_out.flush()
        fail_out.flush()

    keep_rate = (n_kept / max(1, n_kept + n_failed)) * 100
    logging.info(
        "Phase label (multi-provider) done. kept=%d (repair=%d) failed=%d keep_rate=%.1f%%",
        n_kept, n_repaired, n_failed, keep_rate,
    )
    if failure_reasons:
        logging.info("Top-level failure reasons: %s", dict(failure_reasons))
    if attempt_failure_reasons:
        logging.info("Attempt-level failure reasons: %s", dict(attempt_failure_reasons))
    return n_kept + n_failed


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
    recorder=None,
    provider_type: str = "",
) -> None:
    """Run one provider's batches until stop_event is set or local caps hit.

    Maintains separate failure streaks:
      - empty_response_streak: parser returned 0 cleaned lines
      - provider_error_streak: exception bubbled through retries
    """
    import time
    from metrics import estimate_tokens as _est
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
        start = time.perf_counter()
        try:
            lines = provider.generate_inputs(prompt, batch_size)
        except Exception as e:    # noqa: BLE001 — surface any provider failure as an error streak
            latency_ms = (time.perf_counter() - start) * 1000.0
            pop = getattr(provider, "pop_last_usage", None)
            usage = pop() if pop else None
            if recorder is not None:
                if usage:
                    pt = usage["prompt_tokens"]
                    ct = usage["completion_tokens"]
                    est = False
                else:
                    pt = _est(prompt)
                    ct = 0
                    est = True
                recorder.record_call(
                    provider=pcfg.name,
                    provider_type=provider_type,
                    model=getattr(provider, "model", None),
                    phase="inputs",
                    attempt_type="input_batch",
                    latency_ms=latency_ms,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    estimated_tokens=est,
                    success=False,
                    failure_reason="provider_error",
                )
            logging.error("Provider %s batch failed: %s", pcfg.name, e)
            error_streak += 1
            continue
        latency_ms = (time.perf_counter() - start) * 1000.0
        pop = getattr(provider, "pop_last_usage", None)
        usage = pop() if pop else None
        if recorder is not None:
            if usage:
                pt = usage["prompt_tokens"]
                ct = usage["completion_tokens"]
                est = False
            else:
                pt = _est(prompt)
                ct = _est("\n".join(lines))
                est = True
            recorder.record_call(
                provider=pcfg.name,
                provider_type=provider_type,
                model=getattr(provider, "model", None),
                phase="inputs",
                attempt_type="input_batch",
                latency_ms=latency_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
                estimated_tokens=est,
                success=True,
                failure_reason=None,
            )
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


def phase_inputs_multi_provider(args: argparse.Namespace, recorder=None) -> int:
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
        return 0

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
        return 0

    # Only warn if the user explicitly passed --n-inputs (i.e. it appears in
    # sys.argv). Comparing args.n_inputs against the default would fire the
    # warning on every normal invocation when target_inputs != default.
    user_set_n_inputs = any(
        a == "--n-inputs" or a.startswith("--n-inputs=") for a in sys.argv[1:]
    )
    if user_set_n_inputs and args.n_inputs != ig.target_inputs:
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
                    recorder=recorder,
                    provider_type=getattr(pcfg, "provider_type", ""),
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
            if recorder is not None:
                recorder.record_output_row("input")
            if accepted_total % cfg.rate_limits.write_flush_every == 0:
                fout.flush()
            bar.update(1)

        stop_event.set()
        fout.flush()

    logging.info(
        "Phase inputs (multi-provider) done. accepted=%d existing_unique=%d target=%d",
        accepted_total, existing_unique + accepted_total, ig.target_inputs,
    )
    return accepted_total


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
                        "--multi-provider --phase inputs or --phase label to run "
                        "real generation.")
    p.add_argument("--dry-run-quota", action="store_true",
                   help="With --provider-config: parse + validate config, "
                        "print quota allocations and scheduler preview, exit 0. "
                        "No generation runs.")
    p.add_argument("--multi-provider", action="store_true",
                   help="Run multi-provider execution. Supports --phase inputs "
                        "(real input generation) and --phase label (validator-gated "
                        "output labeling). --phase eval is legacy-only — omit "
                        "--multi-provider for eval. --phase all exits via parser.error.")
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
        assert args.phase in {"inputs", "label"}
        assert args.provider_config is not None
        from metrics import MetricsRecorder, load_prices
        prices_path = REPO_ROOT / "configs" / "prices.json"
        prices = load_prices(prices_path)
        recorder = MetricsRecorder(
            prices=prices,
            started_at=datetime.now(timezone.utc),
            prices_source=str(prices_path.relative_to(REPO_ROOT))
                          if prices_path.exists() else None,
        )
        if args.phase == "inputs":
            processed = phase_inputs_multi_provider(args, recorder=recorder)
        else:
            processed = phase_label_multi_provider(args, recorder=recorder)
        summary = recorder.finalize(
            phase=args.phase,
            inputs_processed=processed,
            finished_at=datetime.now(timezone.utc),
        )
        metrics_path = DISTILL_DIR / "metrics.json"
        metrics_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for line in recorder.render_stdout(summary).splitlines():
            logging.info(line)
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
