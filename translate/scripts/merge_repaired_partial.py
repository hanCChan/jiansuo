#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge repaired items into partial eval JSON.")
    p.add_argument("--partial", type=Path, required=True)
    p.add_argument("--repaired", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    partial = json.loads(args.partial.read_text(encoding="utf-8"))
    if not partial:
        raise SystemExit("partial eval input is empty")

    group = partial[0]
    repaired_by_msa: dict[str, str] = {}
    with args.repaired.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("repair_status") != "accepted":
                continue
            repaired_by_msa[row["source_idn"]] = row["msa"]

    added = 0
    for source, msa in repaired_by_msa.items():
        if msa not in group["negative"]:
            group["negative"].append(msa)
            added += 1

    args.output.write_text(json.dumps([group], ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "base_partial_negative_count": len(partial[0]["negative"]),
        "repaired_accepted_count": len(repaired_by_msa),
        "added_negatives": added,
        "final_partial_negative_count": len(group["negative"]),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"merged -> {args.output}")


if __name__ == "__main__":
    main()
