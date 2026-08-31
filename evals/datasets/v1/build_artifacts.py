"""Build deterministic, reviewable V1 dataset artifacts.

Run this module after changing cases.py or policies.json. The generated JSONL,
qrels and checksum manifest are committed so CI evaluates an immutable snapshot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from evals.datasets.v1.cases import all_cases


DATA_DIR = Path(__file__).resolve().parent


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build() -> None:
    cases = [asdict(case) for case in all_cases()]
    _write_jsonl(DATA_DIR / "cases.jsonl", cases)
    qrels = {
        case["case_id"]: case["expected"].get("relevance", {})
        for case in cases
        if case["suite"] == "rag"
    }
    (DATA_DIR / "qrels.json").write_text(
        json.dumps(qrels, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tracked = ["manifest.json", "policies.json", "cases.py", "cases.jsonl", "qrels.json"]
    lines = []
    for name in tracked:
        digest = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (DATA_DIR / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
