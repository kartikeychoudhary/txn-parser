#!/usr/bin/env python3
"""Rebuild GGUFs from a trained adapter using raw llama.cpp, bypassing
Unsloth's ``save_pretrained_gguf`` wrapper.

Why this exists
---------------
Unsloth's wrapper strips the BOS token from the chat template embedded in
the GGUF, which on small models (e.g. Gemma 3 270M) can drop quality by
20+ percentage points at inference time vs the fp16 adapter — even at
"lossless" BF16 quantization. This script:

    1. Merges the LoRA into 16-bit HF safetensors via Unsloth (no GGUF step)
    2. Restores the training-time ``chat_template.jinja`` if anything went
       missing during the merge
    3. Runs llama.cpp's ``convert_hf_to_gguf.py`` directly to produce a
       BF16 GGUF whose embedded template matches what training saw
    4. Calls ``llama-quantize`` for each requested target quant (Q8_0,
       Q5_K_M, Q4_K_M, ...)

Outputs land in ``models/<role>/gguf/`` with a configurable filename
prefix (``-fixed`` by default) so they don't collide with prior exports.

Usage
-----
    # Defaults: rebuild student with bf16,q8_0,q5_k_m,q4_k_m
    python scripts/rebuild_gguf.py --role student

    # Just Q8_0
    python scripts/rebuild_gguf.py --role student --quants q8_0

    # Teacher with custom quants + auto-eval each output
    python scripts/rebuild_gguf.py --role teacher --quants q3_k_m,q4_k_m --eval

    # Custom llama.cpp install path
    python scripts/rebuild_gguf.py --role student --llama-cpp /path/to/llama.cpp
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"

DEFAULT_LLAMA_CPP_DIRS = [
    Path("/root/.unsloth/llama.cpp"),                            # Unsloth's install
    Path.home() / ".unsloth/llama.cpp",
    REPO_ROOT / "llama.cpp",
    Path("/opt/llama.cpp"),
]

# Map lowercase quant names -> the canonical name llama-quantize expects.
QUANT_ALIASES = {
    "bf16": "BF16", "f16": "F16", "fp16": "F16",
    "q8_0": "Q8_0",
    "q6_k": "Q6_K",
    "q5_k_m": "Q5_K_M", "q5_k_s": "Q5_K_S", "q5_0": "Q5_0", "q5_1": "Q5_1",
    "q4_k_m": "Q4_K_M", "q4_k_s": "Q4_K_S", "q4_0": "Q4_0", "q4_1": "Q4_1",
    "q3_k_m": "Q3_K_M", "q3_k_s": "Q3_K_S", "q3_k_l": "Q3_K_L",
    "q2_k": "Q2_K",
}


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


def find_llama_cpp(user_path: str | None) -> Path:
    candidates = [Path(user_path)] if user_path else DEFAULT_LLAMA_CPP_DIRS
    for cand in candidates:
        if (cand / "convert_hf_to_gguf.py").exists():
            return cand
    raise SystemExit(
        f"Could not find llama.cpp install. Tried: {[str(c) for c in candidates]}\n"
        "Pass --llama-cpp <path> or run training once so Unsloth installs it."
    )


def find_quantize_binary(llama_cpp_dir: Path) -> Path:
    for rel in ("build/bin/llama-quantize", "llama-quantize", "build/llama-quantize"):
        p = llama_cpp_dir / rel
        if p.exists() and p.is_file():
            return p
    # Last resort: scan the tree.
    found = list(llama_cpp_dir.rglob("llama-quantize"))
    found = [p for p in found if p.is_file()]
    if found:
        return found[0]
    raise SystemExit(
        f"Could not find llama-quantize under {llama_cpp_dir}. "
        "Build llama.cpp first or pass --llama-cpp <path>."
    )


def run(cmd: list[str], **kw) -> None:
    """Run a subprocess, streaming output to our log."""
    logging.info("$ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        raise SystemExit(f"Command failed (rc={result.returncode}): {cmd[0]}")


def merge_adapter(adapter_dir: Path, merged_dir: Path, max_seq_length: int) -> None:
    """Merge LoRA into the base model and save as 16-bit HF safetensors."""
    logging.info("Loading adapter %s (Unsloth fp16)", adapter_dir)
    # Lazy import — these pull in CUDA.
    from unsloth import FastLanguageModel  # noqa: E402

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    logging.info("Merging LoRA -> 16-bit safetensors at %s", merged_dir)
    model.save_pretrained_merged(
        str(merged_dir), tokenizer, save_method="merged_16bit",
    )


def ensure_chat_template(adapter_dir: Path, merged_dir: Path) -> None:
    """Make sure the merged dir carries the training-time chat template."""
    src = adapter_dir / "chat_template.jinja"
    dst = merged_dir / "chat_template.jinja"
    if src.exists():
        if not dst.exists() or src.read_text() != dst.read_text():
            logging.info("Copying training chat_template.jinja from %s", src)
            shutil.copy(src, dst)
    cfg_path = merged_dir / "tokenizer_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        tpl = cfg.get("chat_template")
        if isinstance(tpl, str):
            logging.info(
                "Embedded chat template — length=%d, has_bos=%s",
                len(tpl), "<bos>" in tpl or "bos_token" in tpl,
            )
        else:
            logging.warning("tokenizer_config.json has no chat_template entry")


def convert_to_bf16_gguf(
    llama_cpp_dir: Path, merged_dir: Path, out_path: Path,
) -> None:
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(out_path),
        "--outtype", "bf16",
    ])


def quantize(quantize_bin: Path, bf16_gguf: Path, out_path: Path, quant: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([str(quantize_bin), str(bf16_gguf), str(out_path), quant])


def _norm_quants(raw: str) -> list[str]:
    out = []
    for q in raw.split(","):
        q = q.strip()
        if not q:
            continue
        key = q.lower()
        if key not in QUANT_ALIASES:
            raise SystemExit(f"Unknown quant '{q}'. Known: {sorted(QUANT_ALIASES)}")
        out.append(QUANT_ALIASES[key])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", choices=["teacher", "student"], required=True)
    p.add_argument("--quants", default="bf16,q8_0,q5_k_m,q4_k_m",
                   help="Comma-separated list. Default: bf16,q8_0,q5_k_m,q4_k_m.")
    p.add_argument("--llama-cpp", default=None,
                   help="llama.cpp install dir. Default: auto-detect.")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--prefix", default="-fixed",
                   help="Suffix added to the model stem in output filenames "
                        "so rebuilt files don't collide with prior Unsloth exports. "
                        "Default: '-fixed' (so 'gemma-3-270m-it-fixed.Q8_0.gguf').")
    p.add_argument("--keep-merged", action="store_true",
                   help="Don't delete models/<role>/merged_16bit/ after the conversion.")
    p.add_argument("--eval", action="store_true",
                   help="Run scripts/04_eval.py on each produced GGUF (--limit 50).")
    p.add_argument("--eval-limit", type=int, default=50,
                   help="--limit value passed to 04_eval.py when --eval is on.")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"rebuild_gguf_{args.role}_{ts}.log")

    quants = _norm_quants(args.quants)
    if "BF16" not in quants:
        # We always need bf16 as the conversion intermediate.
        quants = ["BF16"] + quants
        logging.info("Forcing BF16 into the quant list (needed as conversion intermediate).")

    llama_cpp_dir = find_llama_cpp(args.llama_cpp)
    quantize_bin = find_quantize_binary(llama_cpp_dir)
    logging.info("llama.cpp:      %s", llama_cpp_dir)
    logging.info("llama-quantize: %s", quantize_bin)

    adapter_dir = REPO_ROOT / "models" / args.role / "adapters"
    merged_dir = REPO_ROOT / "models" / args.role / "merged_16bit"
    gguf_dir = REPO_ROOT / "models" / args.role / "gguf"

    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"No adapter at {adapter_dir} — train Stage 3 or 6 first.")

    # Step 1+2: merge + verify template
    if not (merged_dir / "config.json").exists() or not args.keep_merged:
        if merged_dir.exists():
            logging.info("Removing stale %s", merged_dir)
            shutil.rmtree(merged_dir, ignore_errors=True)
        merge_adapter(adapter_dir, merged_dir, args.max_seq_length)
    else:
        logging.info("Reusing existing merged dir %s", merged_dir)
    ensure_chat_template(adapter_dir, merged_dir)

    # Step 3: produce BF16 GGUF (the canonical artifact that the chat template is embedded in)
    base_stem = next(iter(merged_dir.glob("config.json")), None)
    # Pick a clean stem from the base model name in config.json so filenames are sane.
    try:
        cfg = json.loads((merged_dir / "config.json").read_text())
        model_type = cfg.get("model_type") or args.role
    except Exception:  # noqa: BLE001
        model_type = args.role

    # We always emit BF16 first since quantize needs it as input.
    bf16_out = gguf_dir / f"{model_type}{args.prefix}.BF16.gguf"
    convert_to_bf16_gguf(llama_cpp_dir, merged_dir, bf16_out)

    produced = [bf16_out]

    # Step 4: quantize for each non-BF16 target
    for q in quants:
        if q == "BF16":
            continue
        out = gguf_dir / f"{model_type}{args.prefix}.{q}.gguf"
        try:
            quantize(quantize_bin, bf16_out, out, q)
            produced.append(out)
        except SystemExit as e:
            logging.error("Quantize failed for %s: %s", q, e)
            continue

    # Step 5: cleanup intermediate merged dir unless asked to keep it
    if not args.keep_merged:
        logging.info("Removing intermediate %s (pass --keep-merged to retain).", merged_dir)
        shutil.rmtree(merged_dir, ignore_errors=True)

    # Step 6: report
    logging.info("=" * 60)
    logging.info("Produced GGUFs in %s:", gguf_dir)
    for f in produced:
        if f.exists():
            logging.info("  %s  (%.1f MB)", f.name, f.stat().st_size / 1e6)

    # Step 7 (optional): auto-eval
    if args.eval:
        eval_script = REPO_ROOT / "scripts" / "04_eval.py"
        for f in produced:
            if not f.exists():
                continue
            name = f.stem  # e.g. gemma3-fixed.Q8_0
            logging.info("=" * 60)
            logging.info("Evaluating %s", f.name)
            run([
                sys.executable, str(eval_script),
                "--model", str(f),
                "--name", name,
                "--limit", str(args.eval_limit),
            ])

    return 0


if __name__ == "__main__":
    sys.exit(main())
