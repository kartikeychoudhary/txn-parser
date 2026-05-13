#!/usr/bin/env python3
"""Stage 4: Evaluate a fine-tuned model against data/clean/eval.jsonl.

Reusable for both the teacher (Stage 3 output) and the student (Stage 6
output). Auto-detects backend from the path:

  - GGUF file or directory containing *.gguf  -> llama-cpp-python
  - Directory with adapter_config.json        -> Unsloth fp16 inference

Reports % JSON-valid, % schema-valid, % exact-match, mean latency, and a
per-category confusion matrix. Per-example results (including failures)
land in ``eval_results/<model_name>.jsonl``.

Usage:
    python scripts/04_eval.py --model models/teacher/gguf
    python scripts/04_eval.py --model models/teacher/adapters --name teacher-fp16
    python scripts/04_eval.py --model models/student/gguf --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    CATEGORIES,
    SYSTEM_PROMPT,
    build_messages,
    extract_json,
    is_schema_valid,
    load_jsonl,
    schema_errors,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = REPO_ROOT / "data" / "clean" / "eval.jsonl"
RESULTS_DIR = REPO_ROOT / "eval_results"
LOGS_DIR = REPO_ROOT / "logs"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class Backend(Protocol):
    name: str

    def infer(self, input_str: str, max_tokens: int) -> tuple[str, float]: ...


class GgufBackend:
    """llama-cpp-python inference over a GGUF file."""

    def __init__(self, gguf_path: Path, *, n_ctx: int, n_gpu_layers: int) -> None:
        from llama_cpp import Llama  # imported lazily so missing dep doesn't break --help

        self.gguf_path = gguf_path
        self.name = gguf_path.stem
        logging.info("Loading GGUF %s  (n_ctx=%d, n_gpu_layers=%d)",
                     gguf_path, n_ctx, n_gpu_layers)
        # Sanity check the build — saves debugging time later when "stuck at 0%"
        # turns out to be a silent CPU fallback.
        try:
            from llama_cpp import llama_cpp as _llcpp
            gpu_capable = _llcpp.llama_supports_gpu_offload()
            logging.info("llama-cpp-python GPU offload available: %s", gpu_capable)
            if not gpu_capable and n_gpu_layers != 0:
                logging.warning(
                    "llama-cpp-python was built WITHOUT CUDA support — inference "
                    "will run on CPU and be very slow. Rebuild with: "
                    "CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python "
                    "--no-cache-dir --force-reinstall"
                )
        except Exception:  # noqa: BLE001
            pass

        self.llm = Llama(
            model_path=str(gguf_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            seed=42,
            logits_all=False,
        )

    def infer(self, input_str: str, max_tokens: int) -> tuple[str, float]:
        t0 = time.perf_counter()
        resp = self.llm.create_chat_completion(
            messages=build_messages(input_str),
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = resp["choices"][0]["message"]["content"] or ""
        return text, latency_ms


class TransformersBackend:
    """Unsloth fp16 inference over a saved LoRA adapter directory."""

    def __init__(self, adapter_dir: Path, *, max_seq_length: int) -> None:
        from unsloth import FastLanguageModel  # lazy import
        import torch

        self.name = adapter_dir.name
        self.torch = torch

        logging.info("Loading adapter (fp16) from %s", adapter_dir)
        self.model, processor = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(self.model)
        self.max_seq_length = max_seq_length

        # Gemma 3/4 are multimodal — Unsloth returns a Processor here, not a
        # bare Tokenizer. Unwrap to the underlying text tokenizer so plain
        # `tokenizer(text)` returns input_ids without expecting image inputs.
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def infer(self, input_str: str, max_tokens: int) -> tuple[str, float]:
        msgs = build_messages(input_str)
        # Prefer the processor's chat template (has the right special tokens
        # for Gemma 3/4); fall back to the tokenizer if processor isn't one.
        templater = self.processor if hasattr(self.processor, "apply_chat_template") else self.tokenizer
        prompt = templater.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        gen = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        return gen, latency_ms


def resolve_backend(path: Path, args: argparse.Namespace) -> Backend:
    if path.is_file() and path.suffix == ".gguf":
        return GgufBackend(path, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)
    if path.is_dir():
        # Skip multimodal projector sidecars — they're for vision input,
        # not standalone LLM weights, and llama.cpp can't load them as a model.
        ggufs = sorted(
            p for p in path.glob("*.gguf") if "mmproj" not in p.name.lower()
        )
        if ggufs:
            return GgufBackend(ggufs[0], n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)
        if (path / "adapter_config.json").exists():
            return TransformersBackend(path, max_seq_length=args.n_ctx)
    raise SystemExit(
        f"Could not detect a model at {path}. Expected a .gguf file, a directory "
        "containing one, or a directory with adapter_config.json."
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def deep_equal(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(
        b, sort_keys=True, ensure_ascii=False
    )


def score_example(expected: dict, predicted_raw: str) -> dict:
    predicted = extract_json(predicted_raw)
    json_valid = predicted is not None
    schema_valid = json_valid and is_schema_valid(predicted)
    exact = schema_valid and deep_equal(expected, predicted)
    errors = []
    if json_valid and not schema_valid:
        errors = schema_errors(predicted)[:3]
    return {
        "predicted": predicted,
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "exact_match": bool(exact),
        "schema_errors": errors,
    }


def update_confusion(
    confusion: dict[tuple[str, str], int],
    expected: dict,
    predicted: dict | None,
) -> bool:
    """Add pairs to the confusion matrix. Returns True if rows were added."""
    if predicted is None:
        return False
    exp_tx = expected.get("transactions") or []
    pred_tx = predicted.get("transactions") or []
    if len(exp_tx) != len(pred_tx):
        return False
    added = False
    for e, p in zip(exp_tx, pred_tx):
        ec = e.get("category", "<missing>")
        pc = p.get("category", "<missing>")
        confusion[(ec, pc)] += 1
        added = True
    return added


def format_confusion_matrix(confusion: dict[tuple[str, str], int]) -> str:
    if not confusion:
        return "(no category pairs — every example had a transaction-count mismatch)"

    short = {c: c[:4] for c in CATEGORIES}
    header = "         " + " ".join(f"{short[c]:>4}" for c in CATEGORIES)
    rows = [header]
    for exp in CATEGORIES:
        row_total = sum(confusion.get((exp, p), 0) for p in CATEGORIES)
        if row_total == 0:
            continue
        cells = []
        for pred in CATEGORIES:
            v = confusion.get((exp, pred), 0)
            cells.append(f"{v:>4}" if v else "   .")
        rows.append(f"{exp:>8} " + " ".join(cells))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned model.")
    p.add_argument("--model", type=Path, required=True,
                   help="Path to a .gguf file, a dir containing one, or an adapter dir.")
    p.add_argument("--eval-file", type=Path, default=EVAL_FILE,
                   help="JSONL file of {input, output} pairs.")
    p.add_argument("--name", type=str, default=None,
                   help="Override name used for eval_results/<name>.jsonl. "
                        "Defaults to the model file/dir stem.")
    p.add_argument("--limit", type=int, default=0,
                   help="Evaluate only the first N examples. 0 = all.")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Max generation tokens per example.")
    p.add_argument("--n-ctx", type=int, default=2048,
                   help="Context length for llama.cpp / Unsloth.")
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="(GGUF backend) layers offloaded to GPU. -1 = all.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"04_eval_{ts}.log")

    if not args.eval_file.exists():
        logging.error("Eval file not found: %s", args.eval_file)
        return 2
    if not args.model.exists():
        logging.error("Model path not found: %s", args.model)
        return 2

    backend = resolve_backend(args.model, args)
    model_name = args.name or backend.name
    logging.info("Evaluating %s on %s", model_name, args.eval_file)

    eval_records = load_jsonl(args.eval_file)
    for r in eval_records:
        r.pop("_source", None)
    if args.limit > 0:
        eval_records = eval_records[: args.limit]
    n = len(eval_records)
    logging.info("Examples: %d", n)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{model_name}.jsonl"

    n_json = n_schema = n_exact = 0
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    confusion_eligible = 0
    latencies: list[float] = []
    failure_buckets: Counter[str] = Counter()

    with results_path.open("w", encoding="utf-8") as fout:
        for ex in tqdm(eval_records, desc=model_name):
            inp = ex["input"]
            expected = ex["output"]
            try:
                raw, latency_ms = backend.infer(inp, max_tokens=args.max_tokens)
            except Exception as e:  # noqa: BLE001
                logging.error("inference failed on %r: %s", inp, e)
                raw, latency_ms = "", 0.0

            scored = score_example(expected, raw)
            latencies.append(latency_ms)
            if scored["json_valid"]:
                n_json += 1
            if scored["schema_valid"]:
                n_schema += 1
            if scored["exact_match"]:
                n_exact += 1
            if update_confusion(confusion, expected, scored["predicted"]):
                confusion_eligible += 1

            if not scored["json_valid"]:
                failure_buckets["json_invalid"] += 1
            elif not scored["schema_valid"]:
                failure_buckets["schema_invalid"] += 1
            elif not scored["exact_match"]:
                failure_buckets["semantic_mismatch"] += 1

            fout.write(json.dumps({
                "input": inp,
                "expected": expected,
                "predicted_raw": raw,
                "predicted": scored["predicted"],
                "json_valid": scored["json_valid"],
                "schema_valid": scored["schema_valid"],
                "exact_match": scored["exact_match"],
                "schema_errors": scored["schema_errors"],
                "latency_ms": round(latency_ms, 2),
            }, ensure_ascii=False) + "\n")

    pct = lambda x: f"{(x / n * 100):.1f}%" if n else "—"  # noqa: E731
    mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
    p95_lat = sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0.0

    bar = "=" * 60
    logging.info(bar)
    logging.info("EVAL: %s", model_name)
    logging.info(bar)
    logging.info("Examples:           %d", n)
    logging.info("JSON valid:         %d  (%s)", n_json, pct(n_json))
    logging.info("Schema valid:       %d  (%s)", n_schema, pct(n_schema))
    logging.info("Exact match:        %d  (%s)", n_exact, pct(n_exact))
    logging.info("Mean latency:       %.1f ms", mean_lat)
    logging.info("P95 latency:        %.1f ms", p95_lat)
    if failure_buckets:
        logging.info("Failures by bucket: %s", dict(failure_buckets))
    logging.info(bar)
    logging.info(
        "Category confusion (rows=expected, cols=predicted, %d examples used):",
        confusion_eligible,
    )
    logging.info("\n%s", format_confusion_matrix(confusion))
    logging.info(bar)
    logging.info("Per-example results -> %s", results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
