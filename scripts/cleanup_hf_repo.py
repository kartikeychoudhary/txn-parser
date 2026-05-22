#!/usr/bin/env python3
"""Post-v1.1 cleanup of the kartikey31/txn-parser HF repo:

  1. Delete the legacy `student/` and `teacher/` folders from before the
     multi-model layout landed.
  2. Upload a fresh repo-root README.md that points at the 3 new
     per-base-model subfolders.

Defaults to a DRY RUN — pass --apply to actually mutate the repo. The
deleted paths are listed before deletion so you can sanity-check.

Usage:
    # Preview what would happen:
    python scripts/cleanup_hf_repo.py

    # Actually delete the old dirs and upload the new README:
    python scripts/cleanup_hf_repo.py --apply

    # Skip just one of the two phases:
    python scripts/cleanup_hf_repo.py --apply --skip-delete
    python scripts/cleanup_hf_repo.py --apply --skip-readme

    # Target a different repo (for testing on a fork):
    python scripts/cleanup_hf_repo.py --apply --hf-repo someone/txn-parser
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_REPO = "kartikey31/txn-parser"
# Anything that exists at one of these top-level paths gets deleted.
LEGACY_FOLDERS = ("student", "teacher")
# Where the new repo-root README is sourced from.
DEFAULT_README_SOURCE = REPO_ROOT / "docs" / "hf_repo_readme.md"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                   help=f"Target repo. Default: {DEFAULT_HF_REPO}")
    p.add_argument("--readme-source", type=Path, default=DEFAULT_README_SOURCE,
                   help=f"Local file uploaded as repo-root README.md. "
                        f"Default: {DEFAULT_README_SOURCE.relative_to(REPO_ROOT)}")
    p.add_argument("--apply", action="store_true",
                   help="Actually perform deletes + upload. Without this flag, "
                        "the script only previews what it WOULD do.")
    p.add_argument("--skip-delete", action="store_true",
                   help="Skip phase 1 (legacy folder deletion).")
    p.add_argument("--skip-readme", action="store_true",
                   help="Skip phase 2 (new README upload).")
    args = p.parse_args()
    setup_logging()

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit(
            f"huggingface_hub not installed: {e}. Run setup.sh or "
            "`pip install huggingface_hub[cli]`."
        )

    api = HfApi()
    # whoami() throws if no token is reachable — surface the same fix-it
    # text train_and_publish.py uses.
    try:
        api.whoami()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Hugging Face auth check FAILED — cannot reach whoami endpoint.\n"
            f"  Underlying error: {e}\n"
            "Fix one of:\n"
            "  - export HF_TOKEN=hf_xxx   (token with write scope)\n"
            "  - huggingface-cli login    (interactive)"
        )

    files = api.list_repo_files(repo_id=args.hf_repo, repo_type="model")
    legacy_files = [f for f in files if f.split("/", 1)[0] in LEGACY_FOLDERS]

    logging.info("Repo: %s", args.hf_repo)
    logging.info("Legacy files matching %s: %d", LEGACY_FOLDERS, len(legacy_files))
    for f in legacy_files[:10]:
        logging.info("  - %s", f)
    if len(legacy_files) > 10:
        logging.info("  ... and %d more", len(legacy_files) - 10)

    if not args.skip_readme:
        if not args.readme_source.exists():
            raise SystemExit(f"README source not found: {args.readme_source}")
        logging.info("Will upload README.md from: %s (%d bytes)",
                     args.readme_source, args.readme_source.stat().st_size)

    if not args.apply:
        logging.info("=" * 64)
        logging.info("DRY RUN — no changes made. Re-run with --apply to commit.")
        return 0

    # Phase 1: delete legacy folders.
    if not args.skip_delete and legacy_files:
        logging.info("=" * 64)
        logging.info("Deleting %d legacy file(s) from %s/{%s}",
                     len(legacy_files), args.hf_repo, ",".join(LEGACY_FOLDERS))
        # delete_files takes a list of paths; one commit per call. Splitting
        # into ≤50-file chunks keeps each commit small enough not to time out.
        for i in range(0, len(legacy_files), 50):
            chunk = legacy_files[i : i + 50]
            api.delete_files(
                repo_id=args.hf_repo, repo_type="model",
                delete_patterns=chunk,
                commit_message=f"v1.1: remove legacy {LEGACY_FOLDERS} ({i // 50 + 1})",
            )
            logging.info("  deleted batch %d (%d files)", i // 50 + 1, len(chunk))
    elif not args.skip_delete:
        logging.info("No legacy files to delete.")

    # Phase 2: upload new repo-root README.
    if not args.skip_readme:
        logging.info("=" * 64)
        logging.info("Uploading %s -> %s/README.md",
                     args.readme_source.name, args.hf_repo)
        api.upload_file(
            path_or_fileobj=str(args.readme_source),
            path_in_repo="README.md",
            repo_id=args.hf_repo, repo_type="model",
            commit_message="v1.1: repo-root README — point at gemma/smollm/qwen subfolders",
        )
        logging.info("Done — README updated.")

    logging.info("=" * 64)
    logging.info(
        "Repo cleanup complete. View at https://huggingface.co/%s", args.hf_repo,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
