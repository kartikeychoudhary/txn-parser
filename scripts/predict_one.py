"""Quick one-off predict against any GGUF, using the same chat formatting as eval.

Usage:
    python scripts/predict_one.py --model models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf "500 rs on beer 50 rs on candy"
    python scripts/predict_one.py --model models/teacher/gguf/gemma-4-e2b-it.Q3_K_M.gguf "do sau rupay ka chai"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import build_messages, extract_json  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Predict one input against a GGUF.")
    p.add_argument("--model", required=True, help="Path to a .gguf file.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="-1 = all on GPU, 0 = CPU only.")
    p.add_argument("input", help="The transcribed transaction string to parse.")
    args = p.parse_args()

    from llama_cpp import Llama  # lazy import
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        verbose=False,
        seed=42,
    )
    resp = llm.create_chat_completion(
        messages=build_messages(args.input),
        temperature=0.0, top_p=1.0,
        max_tokens=args.max_tokens,
    )
    raw = resp["choices"][0]["message"]["content"] or ""
    parsed = extract_json(raw)

    print(f"MODEL  : {args.model}")
    print(f"INPUT  : {args.input}")
    print(f"RAW    : {raw}")
    print(f"PARSED : {json.dumps(parsed, indent=2, ensure_ascii=False) if parsed else '(not valid JSON)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
