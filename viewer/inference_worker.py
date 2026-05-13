#!/usr/bin/env python3
"""Long-running llama-cpp-python inference worker for the Stage 7 playground.

Protocol (JSON Lines over stdin/stdout, one message per line):

  Ready    : {"ready": true, "model": "<basename>"}            (worker -> parent)
  Request  : {"id": "<uuid>", "input": "<text>", "max_tokens": 512}
  Response : {"id": "<uuid>", "output": "<text>", "latency_ms": 123.4,
              "tokens": 87, "error": null}

Stderr is reserved for human-readable logs (the parent forwards them to
the console). NEVER write anything that isn't a JSON Lines message to
stdout — the server parses each line of stdout as a protocol message.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def log(msg: str) -> None:
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7 inference worker.")
    parser.add_argument("--model", required=True, help="Path to a .gguf file.")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="-1 offloads everything to GPU; 0 forces CPU.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        log(f"FATAL: model file not found: {model_path}")
        return 2

    log(f"loading {model_path.name}  n_ctx={args.n_ctx}  n_gpu_layers={args.n_gpu_layers}")

    # Imports are lazy so --help works even when these deps aren't installed.
    from llama_cpp import Llama  # noqa: E402
    from _lib import build_messages  # noqa: E402

    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
        seed=42,
        logits_all=False,
    )
    log(f"loaded in {time.perf_counter() - t0:.1f}s")
    send({"ready": True, "model": model_path.name})

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"bad request (JSON parse error): {e}")
            continue

        req_id = req.get("id")
        input_text = req.get("input", "")
        max_tokens = int(req.get("max_tokens") or 512)
        if not isinstance(input_text, str) or not input_text.strip():
            send({"id": req_id, "output": "", "latency_ms": 0,
                  "tokens": 0, "error": "empty input"})
            continue

        try:
            t_start = time.perf_counter()
            resp = llm.create_chat_completion(
                messages=build_messages(input_text),
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            choice = resp["choices"][0]
            output_text = choice["message"]["content"] or ""
            usage = resp.get("usage") or {}
            send({
                "id": req_id,
                "output": output_text,
                "latency_ms": round(latency_ms, 2),
                "tokens": usage.get("completion_tokens", 0),
                "error": None,
            })
        except Exception as e:  # noqa: BLE001
            log(f"inference error for id={req_id}: {e}")
            send({
                "id": req_id,
                "output": "",
                "latency_ms": 0,
                "tokens": 0,
                "error": str(e),
            })

    log("stdin closed, exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
