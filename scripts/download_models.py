#!/usr/bin/env python3
"""Pull trained txn-parser GGUFs (and optionally adapters) from Hugging Face
into the local `models/student-<short>/` layout that `eval_all_quants.py`
expects.

The HF repo layout (produced by `train_and_publish.py`) looks like:

    huggingface.co/kartikey31/txn-parser
    ├── gemma-3-270m/
    │   ├── adapters/
    │   ├── gguf/
    │   │   ├── txn-parser-gemma-3-270m-Q4_K_M.gguf
    │   │   └── ...
    │   └── README.md
    ├── smollm2-360m/
    └── qwen3-0.6b/

This script mirrors each `<short>/` subfolder into:

    models/student-<short>/{adapters,gguf,README.md}

so the downstream eval/inference tools find everything in the expected
location. Cached locally — re-running only fetches files that changed.

Usage:
    # Everything (all base models, all quants):
    python scripts/download_models.py

    # Only one base model:
    python scripts/download_models.py --model gemma-3-270m

    # Only one quant of every model (small, fast, enough for a quick eval):
    python scripts/download_models.py --quant Q4_K_M

    # Just the GGUFs, skip the LoRA adapters:
    python scripts/download_models.py --no-adapters

    # Different upstream repo:
    python scripts/download_models.py --hf-repo somebody/txn-parser-fork
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_REPO = "kartikey31/txn-parser"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def list_remote_models(api, repo_id: str) -> list[str]:
    """Return the top-level subfolder names in the HF repo (each one is a
    base-model short name like 'gemma-3-270m')."""
    siblings = api.list_repo_files(repo_id=repo_id, repo_type="model")
    shorts: set[str] = set()
    for f in siblings:
        # Top-level files (no slash) are repo metadata; skip.
        if "/" not in f:
            continue
        shorts.add(f.split("/", 1)[0])
    return sorted(shorts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                   help=f"Source repo (namespace/name). Default: {DEFAULT_HF_REPO}")
    p.add_argument("--model", action="append", default=[],
                   help="Only download this base-model short name "
                        "(e.g. 'gemma-3-270m'). Repeatable. Default: all.")
    p.add_argument("--quant", default=None,
                   help="Only the specified quant (case-insensitive, e.g. 'Q4_K_M'). "
                        "Default: all quants in each subfolder.")
    p.add_argument("--no-adapters", action="store_true",
                   help="Skip the adapters/ subfolder (saves ~30 MB per model). "
                        "Only useful when you'll only do GGUF inference.")
    p.add_argument("--dest", type=Path, default=REPO_ROOT / "models",
                   help="Destination root. Files land in <dest>/student-<short>/. "
                        f"Default: {REPO_ROOT / 'models'}")
    args = p.parse_args()

    setup_logging()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as e:
        raise SystemExit(
            f"huggingface_hub not installed: {e}. Run setup.sh or "
            "`pip install huggingface_hub[cli]`."
        )

    api = HfApi()

    # Discover which model subfolders exist in the repo, then filter.
    available = list_remote_models(api, args.hf_repo)
    if not available:
        logging.error(
            "No subfolders found in %s — is the repo empty? "
            "Have you run train_and_publish.py yet?",
            args.hf_repo,
        )
        return 2

    selected = [m for m in available if not args.model or m in args.model]
    if not selected:
        logging.error(
            "No models match --model %s. Available in %s: %s",
            args.model, args.hf_repo, ", ".join(available),
        )
        return 2

    logging.info("Repo : %s", args.hf_repo)
    logging.info("Found: %s", ", ".join(available))
    logging.info("Will download: %s", ", ".join(selected))

    for short in selected:
        local_dir = args.dest / f"student-{short}"
        local_dir.mkdir(parents=True, exist_ok=True)
        # allow_patterns is the cheap way to restrict the download to a
        # subset (subfolder + optionally one quant).
        allow = [f"{short}/**"]
        ignore = []
        if args.quant:
            # Keep gguf only matching the quant; keep adapters + README
            # so the local layout is usable for inference.
            qup = args.quant.upper()
            ignore.append(f"{short}/gguf/*.gguf")  # nuke all gguf...
            allow.append(f"{short}/gguf/*{qup}*.gguf")  # ...then re-add only the chosen one
        if args.no_adapters:
            ignore.append(f"{short}/adapters/**")

        logging.info("=" * 64)
        logging.info("Downloading %s -> %s", short, local_dir)
        snapshot_path = snapshot_download(
            repo_id=args.hf_repo,
            repo_type="model",
            allow_patterns=allow,
            ignore_patterns=ignore or None,
            # Write into a tmp HF cache dir, then we move the relevant
            # subfolder into the expected location via local_dir mirroring.
            local_dir=str(args.dest / f"_hfcache-{short}"),
        )
        # snapshot_download with allow_patterns=['short/**'] creates
        # _hfcache-<short>/short/{adapters,gguf,README.md}. We want those
        # files at models/student-<short>/{adapters,gguf,README.md}.
        src_subdir = Path(snapshot_path) / short
        if not src_subdir.exists():
            logging.warning("Downloaded snapshot missing %s — skipping promote.", src_subdir)
            continue

        # Move the contents up one level into student-<short>/.
        import shutil
        for child in src_subdir.iterdir():
            target = local_dir / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))
        # Clean up the intermediate cache dir.
        shutil.rmtree(Path(snapshot_path), ignore_errors=True)

        # Report what landed.
        ggufs = sorted((local_dir / "gguf").glob("*.gguf")) if (local_dir / "gguf").exists() else []
        logging.info("Got %d GGUF(s) for %s:", len(ggufs), short)
        for g in ggufs:
            logging.info("  %s  (%.1f MB)", g.name, g.stat().st_size / 1e6)

    logging.info("=" * 64)
    logging.info("Done. Run `python scripts/eval_all_quants.py` to generate REPORT.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
