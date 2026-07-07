#!/usr/bin/env python3
"""Build strict/near-strict eval JSON from debug + repaired translations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idn_msa.config_loader import load_config
from idn_msa.expand import TranslationItem
from idn_msa.hard_qa import apply_postprocess, hard_qa_item
from idn_msa.runner import prepare_items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export strict eval from debug + repaired.")
    p.add_argument("--debug", type=Path, required=True)
    p.add_argument("--repaired", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="Original IDN eval JSON")
    p.add_argument("--group-index", type=int, default=1)
    p.add_argument("--output-strict", type=Path, required=True)
    p.add_argument("--output-failed", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    with args.input.open(encoding="utf-8") as f:
        records = json.load(f)
    record = records[args.group_index - 1]

    with args.debug.open(encoding="utf-8") as f:
        debug_row = json.loads(f.readline())
    d = debug_row["debug"]

    repaired: dict[str, str] = {}
    with args.repaired.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("repair_status") == "accepted":
                repaired[row["source_idn"]] = row["msa"]

    msa_by_src: dict[str, str] = {d["query_idn"]: d["query_msa"]}
    for p in d["positive"]:
        msa_by_src[p["idn"]] = p["msa"]
    for n in d["negative"]:
        msa_by_src[n["idn"]] = n["msa"]
    msa_by_src.update(repaired)

    items = prepare_items(record, args.group_index, cfg)
    passed_items: list[TranslationItem] = []
    failed_rows: list[dict] = []

    for item in items:
        item.msa_raw = msa_by_src.get(item.source_idn, "")
        apply_postprocess(item, cfg)
        qa = hard_qa_item(item, cfg)
        item.qa = qa
        if qa["hard_pass"]:
            item.final_status = "accepted"
            passed_items.append(item)
        else:
            item.final_status = "failed"
            failed_rows.append(
                {
                    "item_id": item.item_id,
                    "role": item.role,
                    "source_idn": item.source_idn,
                    "msa": item.msa_raw,
                    "hard_errors": qa["hard_errors"],
                }
            )

    query = next(i for i in passed_items if i.role == "query")
    positives = sorted([i for i in passed_items if i.role == "positive"], key=lambda x: x.candidate_index)
    negatives = sorted([i for i in passed_items if i.role == "negative"], key=lambda x: x.candidate_index)

    strict_ready = len(failed_rows) == 0 and len(negatives) == len(record.get("negative", []))
    out_group = {
        "query": query.msa_raw,
        "positive": [p.msa_raw for p in positives],
        "negative": [n.msa_raw for n in negatives],
    }
    payload = [out_group] if strict_ready else []

    args.output_strict.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output_failed.open("w", encoding="utf-8") as f:
        for row in failed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_items": len(items),
        "passed": len(passed_items),
        "failed": len(failed_rows),
        "pass_rate_pct": round(len(passed_items) / len(items) * 100, 2),
        "strict_eval_ready": strict_ready,
        "negative_in_output": len(negatives),
        "negative_expected": len(record.get("negative", [])),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
