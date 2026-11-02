"""
Shard checkpoint manager for the judge (same idea as the thesis SoilCheckpointManager).

Every shard of judged pairs is pushed to a private HF dataset repo as shards/shard_000123.jsonl.
On a new Colab session, try_load_done() downloads what exists so judging resumes where it stopped.
"""
from __future__ import annotations

import json
import os
import tempfile

from huggingface_hub import HfApi, hf_hub_download


class JudgeShardManager:
    def __init__(self, namespace: str, export: str):
        self.api = HfApi()
        self.repo_id = f"{namespace}/mbg-judge-{export}"
        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)

    def _shard_files(self) -> list[str]:
        files = self.api.list_repo_files(self.repo_id, repo_type="dataset")
        return sorted(f for f in files if f.startswith("shards/shard_") and f.endswith(".jsonl"))

    def try_load_done(self) -> tuple[list[dict], int]:
        """All judged rows already on the Hub + next shard number."""
        rows: list[dict] = []
        files = self._shard_files()
        for f in files:
            p = hf_hub_download(self.repo_id, f, repo_type="dataset")
            with open(p, encoding="utf-8") as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
        next_idx = (int(files[-1].split("_")[-1].split(".")[0]) + 1) if files else 0
        return rows, next_idx

    def save_shard(self, idx: int, rows: list[dict]) -> str:
        path = os.path.join(tempfile.gettempdir(), f"shard_{idx:06d}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        dest = f"shards/shard_{idx:06d}.jsonl"
        self.api.upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=self.repo_id,
                             repo_type="dataset", commit_message=f"judge shard {idx} ({len(rows)} pairs)")
        return dest

    def save_file(self, local_path: str, path_in_repo: str, message: str) -> None:
        self.api.upload_file(path_or_fileobj=local_path, path_in_repo=path_in_repo, repo_id=self.repo_id,
                             repo_type="dataset", commit_message=message)


def health_check(rows: list[dict]) -> None:
    """Abort after the FIRST shard of this session if the judge output looks broken."""
    n = len(rows)
    if n == 0:
        return
    parse_fail = sum(1 for r in rows if r["verdict"] is None) / n
    kq = sum(1 for r in rows if r["keep_question"]) / n
    print(f"health: parse_fail={parse_fail:.1%}  keep_question={kq:.1%}  (n={n})")
    if parse_fail > 0.05:
        raise SystemExit(f"ABORT: {parse_fail:.0%} of judge outputs are not valid JSON — check the model/prompt.")
    if not (0.02 <= kq <= 0.98):
        raise SystemExit(f"ABORT: keep_question={kq:.0%} — judge says (almost) all yes or all no; prompt is broken.")
    print("health check passed")
