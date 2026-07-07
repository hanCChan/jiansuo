#!/usr/bin/env python3
"""Restore Kimi <ENT_*> hallucinations from source text and re-export eval JSON."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idn_msa.config_loader import load_config
from idn_msa.expand import TranslationItem, expand_record
from idn_msa.hard_qa import apply_postprocess, hard_qa_item
from idn_msa.mask import annotate_item
from idn_msa.placeholder_restore import ENT_RE, restore_kimi_placeholders

ENT_RE_CHECK = ENT_RE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix <ENT_*> placeholders in eval JSON.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--source-json", type=Path, required=True)
    p.add_argument("--group-index", type=int, default=1)
    p.add_argument("--debug", type=Path, help="Optional debug jsonl for repaired msa overrides")
    p.add_argument("--repaired", type=Path, help="Optional repaired jsonl overrides by source_idn")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    p.add_argument("--still-bad-out", type=Path, help="Write items still containing ENT")
    return p.parse_args()


def load_msa_overrides(debug: Path | None, repaired: Path | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if debug and debug.exists():
        row = json.loads(debug.read_text(encoding="utf-8").splitlines()[0])
        d = row["debug"]
        overrides[d["query_idn"]] = d["query_msa"]
        for item in d["positive"]:
            overrides[item["idn"]] = item["msa"]
        for item in d["negative"]:
            overrides[item["idn"]] = item["msa"]
    if repaired and repaired.exists():
        for line in repaired.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("repair_status") == "accepted":
                overrides[row["source_idn"]] = row["msa"]
    return overrides


def main() -> None:
    args = parse_args()
    cfg = load_config()
    record = json.loads(args.source_json.read_text(encoding="utf-8"))[args.group_index - 1]
    items = expand_record(record, args.group_index)
    overrides = load_msa_overrides(args.debug, args.repaired)

    fixed_rows: list[dict] = []
    still_bad: list[dict] = []
    fixed_count = 0

    rebuilt = {"query": "", "positive": [], "negative": []}
    for item in items:
        msa = overrides.get(item.source_idn, "")
        if not msa:
            raise SystemExit(f"missing msa for {item.item_id}")

        had_ent = bool(ENT_RE_CHECK.search(msa))
        if had_ent:
            msa, notes = restore_kimi_placeholders(item.source_idn, msa, cfg)
            if ENT_RE_CHECK.search(msa):
                still_bad.append(
                    {
                        "item_id": item.item_id,
                        "source_idn": item.source_idn,
                        "msa": msa,
                        "remaining": ENT_RE_CHECK.findall(msa),
                        "notes": notes,
                    }
                )
            else:
                fixed_count += 1
                fixed_rows.append(
                    {
                        "item_id": item.item_id,
                        "source_idn": item.source_idn,
                        "msa": msa,
                        "notes": notes,
                    }
                )

        item.msa_raw = msa
        masked, entities, terms, actions = annotate_item(item.source_idn, cfg)
        item.masked_idn = masked
        item.entities_found = entities
        item.terms_found = terms
        item.actions_found = actions
        apply_postprocess(item, cfg)
        qa = hard_qa_item(item, cfg)
        item.qa = qa

        if item.role == "query":
            rebuilt["query"] = msa
        elif item.role == "positive":
            rebuilt["positive"].append(msa)
        else:
            rebuilt["negative"].append(msa)

    args.output.write_text(json.dumps([rebuilt], ensure_ascii=False, indent=2), encoding="utf-8")

    ent_left = sum(
        1
        for t in [rebuilt["query"], *rebuilt["positive"], *rebuilt["negative"]]
        if ENT_RE_CHECK.search(t)
    )
    summary = {
        "fixed_placeholder_items": fixed_count,
        "still_with_ent": ent_left,
        "still_bad_items": len(still_bad),
        "total_items": len(items),
        "negative_count": len(rebuilt["negative"]),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.still_bad_out:
        with args.still_bad_out.open("w", encoding="utf-8") as f:
            for row in still_bad:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if ent_left:
        print(f"WARN: {ent_left} texts still contain ENT placeholders")
        sys.exit(2)
    print(f"OK -> {args.output}")


if __name__ == "__main__":
    main()
