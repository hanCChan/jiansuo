#!/usr/bin/env python3
"""Sanity check for repaired partial eval JSON before embedding."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ENT_RE = re.compile(r"<ENT[^>]*>", re.I)
IDN_RESIDUE = re.compile(
    r"\b(bagaimana|cara|terblokir|password|rekening|kartu|nomor|tidak|bisa|masuk)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity check MSA partial eval JSON.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--output", type=Path, help="Write report JSON")
    return p.parse_args()


def check_group(group: dict, idx: int) -> list[str]:
    errors: list[str] = []
    if set(group.keys()) != {"query", "positive", "negative"}:
        errors.append(f"group[{idx}] keys must be query/positive/negative only")
    if not group.get("query", "").strip():
        errors.append(f"group[{idx}] empty query")
    if not group.get("positive"):
        errors.append(f"group[{idx}] empty positive")
    if not group.get("negative"):
        errors.append(f"group[{idx}] empty negative")
    pos_set = set(group.get("positive", []))
    neg_set = set(group.get("negative", []))
    overlap = pos_set & neg_set
    if overlap:
        errors.append(f"group[{idx}] positive/negative overlap: {len(overlap)}")
    for i, text in enumerate([group.get("query", "")] + group.get("positive", []) + group.get("negative", [])):
        if ENT_RE.search(text):
            errors.append(f"group[{idx}] item {i} has ENT placeholder")
        if IDN_RESIDUE.search(text):
            errors.append(f"group[{idx}] item {i} has Indonesian residue")
    return errors


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    report: dict = {"file": str(args.input), "checks": {}, "errors": [], "samples": []}

    report["checks"]["is_list"] = isinstance(data, list)
    if not isinstance(data, list):
        print("FAIL: not a list")
        sys.exit(1)

    all_errors: list[str] = []
    neg_counts = []
    for i, group in enumerate(data):
        all_errors.extend(check_group(group, i))
        neg_counts.append(len(group.get("negative", [])))

    report["checks"]["group_count"] = len(data)
    report["checks"]["negative_counts"] = neg_counts
    report["errors"] = all_errors
    report["checks"]["passed"] = len(all_errors) == 0

    pool = []
    for gi, group in enumerate(data):
        pool.append(("query", group["query"]))
        for p in group["positive"]:
            pool.append(("positive", p))
        for n in random.sample(group["negative"], min(args.sample, len(group["negative"]))):
            pool.append(("negative", n))
    report["samples"] = [{"role": r, "text": t[:200]} for r, t in random.sample(pool, min(args.sample, len(pool)))]

    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False, indent=2))
    if all_errors:
        print("FAIL", len(all_errors), "issues")
        for e in all_errors[:10]:
            print(" -", e)
        sys.exit(1)
    print("PASS: sanity check ok")
    print("Sample lines:")
    for s in report["samples"][:5]:
        print(f"  [{s['role']}] {s['text']}")


if __name__ == "__main__":
    main()
