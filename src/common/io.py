"""Small JSONL / path helpers shared by the data-prep scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]          # mbg-ai-model/
DATA = ROOT / "data"
LOGS = ROOT / "logs"


def export_dir(stage: str, export_name: str) -> Path:
    """data/<stage>/<export_name>/ — one folder per Chatrace export so new exports never overwrite old ones."""
    d = DATA / stage / export_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_export(stage: str = "interim") -> str:
    """Name of the most recently modified export folder under data/<stage>/."""
    dirs = [p for p in (DATA / stage).iterdir() if p.is_dir()]
    if not dirs:
        raise SystemExit(f"No exports found in data/{stage}/ — run 01_extract_conversations.py first.")
    return max(dirs, key=lambda p: p.stat().st_mtime).name


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_report(name: str, report: dict) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    p = LOGS / f"{name}.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
