#!/usr/bin/env python3
"""Evaluate every published GGUF (all base models × all quants) and write
a single Markdown report suitable for a GitHub README.

Walks `models/student-*/gguf/*.gguf`, runs `04_eval.py` on each (skipping
quants that already have a JSONL result unless --force is passed), then
aggregates `eval_results/*.jsonl` into a Markdown table comparing
JSON-valid / schema-valid / exact-match / latency across every (model,
quant) combo.

Output:
    eval_results/REPORT.md      <- copy/paste straight into README.md
    eval_results/REPORT.json    <- raw numbers for downstream automation

Usage:
    # Run all quants of all trained models, produce REPORT.md.
    python scripts/eval_all_quants.py

    # Only the four-bit quant of every model (fastest sweep):
    python scripts/eval_all_quants.py --quant Q4_K_M

    # Just one model, all its quants:
    python scripts/eval_all_quants.py --model gemma-3-270m

    # Re-eval everything from scratch (ignore cached eval_results/*.jsonl):
    python scripts/eval_all_quants.py --force

    # Use a smaller eval slice while iterating on the report layout:
    python scripts/eval_all_quants.py --limit 50

NOTE: requires `llama-cpp-python` built with CUDA support. On CPU each
quant takes ~5-10 min per 300 examples; on GPU it's ~30-60 s per quant.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "eval_results"
LOGS_DIR = REPO_ROOT / "logs"
EVAL_FILE_DEFAULT = REPO_ROOT / "data" / "clean" / "eval.jsonl"

# Quant ordering used in the report table. Anything not in this list goes
# at the end in alphabetic order.
QUANT_ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / f"eval_all_quants_{ts}.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def discover_gguf_files(model_filter: str | None, quant_filter: str | None) -> list[Path]:
    """Find every GGUF in models/student-*/gguf/ matching the filters."""
    out: list[Path] = []
    for gguf_dir in sorted(REPO_ROOT.glob("models/student-*/gguf")):
        model_short = gguf_dir.parent.name.removeprefix("student-")
        if model_filter and model_filter != model_short:
            continue
        for gguf in sorted(gguf_dir.glob("*.gguf")):
            if quant_filter and quant_filter.upper() not in gguf.stem.upper():
                continue
            out.append(gguf)
    return out


def parse_model_and_quant(gguf_path: Path) -> tuple[str, str]:
    """`models/student-gemma-3-270m/gguf/txn-parser-gemma-3-270m-Q4_K_M.gguf`
    -> ('gemma-3-270m', 'Q4_K_M')

    Falls back to the stem if the naming convention isn't followed."""
    model_short = gguf_path.parent.parent.name.removeprefix("student-")
    stem = gguf_path.stem.upper()
    for q in QUANT_ORDER:
        if q in stem:
            return model_short, q
    # Unknown quant — best-effort: last hyphen-separated chunk
    tail = stem.rsplit("-", 1)[-1] if "-" in stem else stem
    return model_short, tail


def eval_name_for(model_short: str, quant: str) -> str:
    return f"{model_short}-{quant}"


def run_one_eval(gguf: Path, model_short: str, quant: str, args: argparse.Namespace) -> bool:
    """Invoke 04_eval.py for one GGUF. Returns True on success."""
    name = eval_name_for(model_short, quant)
    out_path = RESULTS_DIR / f"{name}.jsonl"
    if out_path.exists() and not args.force:
        logging.info("[skip] %s already evaluated (-> %s). Pass --force to redo.",
                     name, out_path.name)
        return True
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "04_eval.py"),
        "--model", str(gguf),
        "--name", name,
        "--eval-file", str(args.eval_file),
    ]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    if args.no_grammar:
        cmd += ["--no-grammar"]
    logging.info("[eval] %s", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    elapsed = time.time() - t0
    if rc != 0:
        logging.error("[fail] %s exited with rc=%d after %.1fs", name, rc, elapsed)
        return False
    logging.info("[done] %s in %.1fs", name, elapsed)
    return True


def summarize_result_file(jsonl_path: Path) -> dict:
    """Read an eval_results/<name>.jsonl produced by 04_eval.py and return
    aggregated metrics."""
    n = json_valid = schema_valid = exact = 0
    amount_exact = txn_count_exact = duplicate = superseded = 0
    latencies: list[float] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            if r.get("json_valid"): json_valid += 1
            if r.get("schema_valid"): schema_valid += 1
            if r.get("exact_match"): exact += 1
            if r.get("amount_exact"): amount_exact += 1
            if r.get("txn_count_exact"): txn_count_exact += 1
            if r.get("duplicate_transactions_found"): duplicate += 1
            if r.get("superseded_amount_used"): superseded += 1
            lat = float(r.get("latency_ms") or 0)
            if lat > 0:
                latencies.append(lat)

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(0.95 * len(latencies)) - 1] if len(latencies) >= 2 else 0.0

    def pct(x: int) -> float:
        return (x / n * 100) if n else 0.0

    return {
        "n": n,
        "json_valid_pct": round(pct(json_valid), 1),
        "schema_valid_pct": round(pct(schema_valid), 1),
        "exact_match_pct": round(pct(exact), 1),
        "amount_exact_pct": round(pct(amount_exact), 1),
        "txn_count_exact_pct": round(pct(txn_count_exact), 1),
        "duplicate_pct": round(pct(duplicate), 1),
        "superseded_pct": round(pct(superseded), 1),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
    }


def quant_sort_key(q: str) -> tuple[int, str]:
    """Sort known quants in the QUANT_ORDER list order; unknowns last alphabetically."""
    if q in QUANT_ORDER:
        return (QUANT_ORDER.index(q), q)
    return (len(QUANT_ORDER), q)


def format_file_size_mb(gguf_path: Path) -> str:
    if not gguf_path.exists():
        return "—"
    return f"{gguf_path.stat().st_size / 1e6:.0f} MB"


def write_report(results: list[dict], report_md: Path, report_json: Path,
                 eval_file: Path) -> None:
    """Render results into a Markdown table + raw JSON sidecar."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    # Eval set count from any one row (they're all the same eval)
    n_eval = results[0]["metrics"]["n"] if results else 0
    grammar_state = "enabled" if not any(r.get("no_grammar") for r in results) else "disabled"
    generated_at = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append("# txn-parser eval report")
    lines.append("")
    lines.append(f"- **Eval set**: `{eval_file.name}` ({n_eval} examples)")
    lines.append(f"- **Grammar**: {grammar_state} (GBNF constrains output to valid JSON schema)")
    lines.append(f"- **Generated**: {generated_at}")
    lines.append("")
    lines.append(
        "Each row is the trained model at a single quantization. "
        "**Schema valid** is the headline metric for an app — it's what "
        "you need to parse the output reliably. **Exact match** is a strict "
        "byte-equality check against the teacher label."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Quant | Size | JSON valid | Schema valid | Exact match | Amount exact | Mean ms | P95 ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for model in sorted(by_model):
        rows = sorted(by_model[model], key=lambda r: quant_sort_key(r["quant"]))
        for r in rows:
            m = r["metrics"]
            lines.append(
                f"| `{model}` | **{r['quant']}** | {r['size']} | "
                f"{m['json_valid_pct']:.1f}% | "
                f"**{m['schema_valid_pct']:.1f}%** | "
                f"{m['exact_match_pct']:.1f}% | "
                f"{m['amount_exact_pct']:.1f}% | "
                f"{m['mean_latency_ms']:.0f} | "
                f"{m['p95_latency_ms']:.0f} |"
            )
    lines.append("")
    lines.append("## Per-model detail")
    lines.append("")

    for model in sorted(by_model):
        rows = sorted(by_model[model], key=lambda r: quant_sort_key(r["quant"]))
        best = max(rows, key=lambda r: (
            r["metrics"]["schema_valid_pct"],
            r["metrics"]["exact_match_pct"],
            -r["metrics"]["mean_latency_ms"],
        ))
        lines.append(f"### `{model}`")
        lines.append("")
        lines.append(f"Best schema_valid: **{best['quant']}** "
                     f"({best['metrics']['schema_valid_pct']:.1f}%, "
                     f"{best['metrics']['mean_latency_ms']:.0f} ms mean).")
        lines.append("")
        lines.append("| Quant | Schema | Exact | Amt | TxnCount | Dup% | Super% | Mean ms | P95 ms |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            m = r["metrics"]
            lines.append(
                f"| {r['quant']} | {m['schema_valid_pct']:.1f}% | "
                f"{m['exact_match_pct']:.1f}% | {m['amount_exact_pct']:.1f}% | "
                f"{m['txn_count_exact_pct']:.1f}% | "
                f"{m['duplicate_pct']:.1f}% | {m['superseded_pct']:.1f}% | "
                f"{m['mean_latency_ms']:.0f} | {m['p95_latency_ms']:.0f} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("*Generated by `python scripts/eval_all_quants.py`. "
                 "Per-example results live in `eval_results/<model>-<quant>.jsonl`.*")
    lines.append("")

    report_md.write_text("\n".join(lines), encoding="utf-8")
    report_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logging.info("Wrote %s and %s", report_md, report_json)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None,
                   help="Only eval this base-model short name "
                        "(e.g. 'gemma-3-270m'). Default: all under models/student-*.")
    p.add_argument("--quant", default=None,
                   help="Only eval this quant (e.g. 'Q4_K_M'). Default: all.")
    p.add_argument("--eval-file", type=Path, default=EVAL_FILE_DEFAULT,
                   help="JSONL eval set passed to 04_eval.py.")
    p.add_argument("--limit", type=int, default=0,
                   help="Forward to 04_eval.py --limit (use a small N to iterate).")
    p.add_argument("--no-grammar", action="store_true",
                   help="Run unconstrained inference (no GBNF grammar). "
                        "Useful for measuring raw model quality; turn back on "
                        "for the numbers you'd see in production.")
    p.add_argument("--force", action="store_true",
                   help="Re-evaluate even if eval_results/<name>.jsonl exists.")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip the eval pass entirely; just aggregate existing "
                        "eval_results/*.jsonl into REPORT.md. Use when you've "
                        "already run evals manually and only want the report.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_eval:
        if not args.eval_file.exists():
            logging.error("Eval file not found: %s", args.eval_file)
            return 2
        gguf_files = discover_gguf_files(args.model, args.quant)
        if not gguf_files:
            logging.error(
                "No GGUFs found under models/student-*/gguf/ matching "
                "model=%s quant=%s. Train first with scripts/train_and_publish.py.",
                args.model, args.quant,
            )
            return 2
        logging.info("Found %d GGUF(s) to evaluate.", len(gguf_files))
        for gguf in gguf_files:
            model_short, quant = parse_model_and_quant(gguf)
            run_one_eval(gguf, model_short, quant, args)

    # Aggregation pass: read every eval_results/<short>-<QUANT>.jsonl that
    # corresponds to a GGUF we know about, even if --skip-eval skipped the
    # eval phase.
    aggregated: list[dict] = []
    discovered = discover_gguf_files(args.model, args.quant)
    for gguf in discovered:
        model_short, quant = parse_model_and_quant(gguf)
        name = eval_name_for(model_short, quant)
        result_jsonl = RESULTS_DIR / f"{name}.jsonl"
        if not result_jsonl.exists():
            logging.warning("[miss] no eval result for %s — skipping in report.", name)
            continue
        metrics = summarize_result_file(result_jsonl)
        aggregated.append({
            "model": model_short,
            "quant": quant,
            "size": format_file_size_mb(gguf),
            "gguf": str(gguf.relative_to(REPO_ROOT)),
            "results_path": str(result_jsonl.relative_to(REPO_ROOT)),
            "metrics": metrics,
            "no_grammar": args.no_grammar,
        })

    if not aggregated:
        logging.error("No eval results available to aggregate. Re-run without --skip-eval "
                      "or pass --force if results are stale.")
        return 1

    report_md = RESULTS_DIR / "REPORT.md"
    report_json = RESULTS_DIR / "REPORT.json"
    write_report(aggregated, report_md, report_json, args.eval_file)

    # Echo a one-screen preview so the user can spot regressions immediately
    # without opening the file.
    preview_lines = report_md.read_text(encoding="utf-8").splitlines()
    bar = "=" * 72
    logging.info(bar)
    logging.info("PREVIEW: %s", report_md)
    logging.info(bar)
    for line in preview_lines:
        logging.info(line)
    logging.info(bar)
    logging.info("Paste %s into your GitHub README's eval section.", report_md.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
