"""
Level 1 · Step 5 — Push judge candidates to a PRIVATE Hugging Face dataset repo.

Input : data/interim/<export>/candidates.jsonl  (already PII-redacted, from step 04)
Output: <hf-user>/mbg-chat-candidates  (private dataset)  ->  <export>/candidates.jsonl

Login: uses your saved HF token if you already ran `hf auth login` (or `huggingface-cli login`);
otherwise it asks for the token right here in the terminal (input is hidden). The token is never
written into this project.

Run from mbg-ai-model/:   python src/data/05_push_candidates.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import LOGS, export_dir, latest_export  # noqa: E402

EXPECTED_USER = "ankit-iiitdmj"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Export name (default: latest in data/interim/)")
    ap.add_argument("--repo", default="mbg-chat-candidates")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, get_token, login
    except ImportError:
        raise SystemExit("pip install huggingface_hub   (then run this again)")

    name = args.name or latest_export("interim")
    src = export_dir("interim", name) / "candidates.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} not found — run 04_build_candidates.py first.")

    if not get_token():
        login()                                   # hidden prompt in the terminal
    api = HfApi()
    who = api.whoami()["name"]
    if who != EXPECTED_USER:
        raise SystemExit(f"HF token belongs to {who!r}, expected {EXPECTED_USER!r}.")

    repo_id = f"{who}/{args.repo}"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(src), path_in_repo=f"{name}/candidates.jsonl", repo_id=repo_id,
                    repo_type="dataset", commit_message=f"candidates for {name}")
    rep = LOGS / f"{name}_04_candidates.json"
    if rep.exists():
        api.upload_file(path_or_fileobj=str(rep), path_in_repo=f"{name}/04_candidates_report.json",
                        repo_id=repo_id, repo_type="dataset", commit_message=f"candidates report for {name}")
    info = api.repo_info(repo_id, repo_type="dataset")
    print(f"Pushed {src.name} ({src.stat().st_size / 1e6:.1f} MB) -> {repo_id}/{name}/  private={info.private}")


if __name__ == "__main__":
    main()
