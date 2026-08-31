from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a reviewed failure into the regression dataset")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", type=Path, default=Path("evals/datasets/v1/regressions.jsonl"))
    args = parser.parse_args()

    matches = [
        json.loads(line)
        for line in args.source.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("fingerprint") == args.fingerprint
    ]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one reviewed failure, found {len(matches)}")
    record = matches[0]
    promoted = {
        "fingerprint": record["fingerprint"],
        "case_id": record["case_id"],
        "suite": record["suite"],
        "failure_code": record.get("failure_code"),
        "reviewer": args.reviewer,
        "review_reason": args.reason,
        "source_artifacts": record.get("artifacts", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
    if record["fingerprint"] in existing:
        raise SystemExit("Failure is already present in the regression dataset")
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(promoted, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
