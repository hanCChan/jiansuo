#!/usr/bin/env python3
"""Re-run current hard QA rules on saved debug translations (no Kimi calls)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idn_msa.config_loader import load_config
from idn_msa.expand import TranslationItem
from idn_msa.hard_qa import apply_postprocess, hard_qa_item
from idn_msa.mask import annotate_item
from idn_msa.runner import assemble_group_output_debug, assemble_group_output_simple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-evaluate debug translations with current hard QA.")
    p.add_argument("--debug", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    p.add_argument("--partial-out", type=Path, required=True)
    p.add_argument("--failed-out", type=Path, required=True)
    p.add_argument("--input", type=Path, help="Original IDN eval JSON for structure reference")
    return p.parse_args()


def items_from_debug(debug_row: dict, cfg) -> list[TranslationItem]:
    d = debug_row["debug"]
    group_idx = debug_row.get("group_index", 1)
    group_id = d["query_id"]
    items: list[TranslationItem] = []

    items.append(
        TranslationItem(
            item_id=f"{group_id}_query",
            group_id=group_id,
            role="query",
            candidate_index=0,
            source_idn=d["query_idn"],
            msa_raw=d["query_msa"],
        )
    )
    for p in d["positive"]:
        idx = int(p["id"].rsplit("_", 1)[-1])
        items.append(
            TranslationItem(
                item_id=p["id"],
                group_id=group_id,
                role="positive",
                candidate_index=idx,
                source_idn=p["idn"],
                msa_raw=p["msa"],
            )
        )
    for n in d["negative"]:
        idx = int(n["id"].rsplit("_", 1)[-1])
        items.append(
            TranslationItem(
                item_id=n["id"],
                group_id=group_id,
                role="negative",
                candidate_index=idx,
                source_idn=n["idn"],
                msa_raw=n["msa"],
            )
        )

    for item in items:
        masked, entities, terms, actions = annotate_item(item.source_idn, cfg)
        item.masked_idn = masked
        item.entities_found = entities
        item.terms_found = terms
        item.actions_found = actions
        apply_postprocess(item, cfg)
        item.qa = hard_qa_item(item, cfg)
        item.final_status = "accepted" if item.qa["hard_pass"] else "failed"

    return items


def assemble_partial(items: list[TranslationItem]) -> dict | None:
    query = next(i for i in items if i.role == "query")
    positives = [i for i in items if i.role == "positive"]
    negatives = [i for i in items if i.role == "negative"]
    if query.final_status != "accepted":
        return None
    if any(p.final_status != "accepted" for p in positives):
        return None
    accepted_neg = [n for n in negatives if n.final_status == "accepted"]
    return {
        "query": query.msa_raw,
        "positive": [p.msa_raw for p in positives],
        "negative": [n.msa_raw for n in accepted_neg],
    }


def main() -> None:
    args = parse_args()
    cfg = load_config()

    summaries = []
    partial_rows = []
    failed_rows = []

    with args.debug.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            debug_row = json.loads(line)
            items = items_from_debug(debug_row, cfg)

            status_ctr = Counter(i.final_status for i in items)
            err_ctr = Counter()
            warn_ctr = Counter()
            for item in items:
                if item.final_status == "accepted":
                    continue
                for e in item.qa.get("hard_errors", []):
                    err_ctr[e.split(":")[0]] += 1
                for w in item.qa.get("hard_warnings", []):
                    warn_ctr[w.split(":")[0]] += 1
                failed_rows.append(
                    {
                        "item_id": item.item_id,
                        "role": item.role,
                        "source_idn": item.source_idn,
                        "msa": item.msa_raw,
                        "hard_errors": item.qa.get("hard_errors", []),
                        "hard_warnings": item.qa.get("hard_warnings", []),
                    }
                )

            group_id = items[0].group_id
            debug = assemble_group_output_debug(group_id, items)
            partial = assemble_partial(items)

            summary = {
                "query_id": group_id,
                "total_items": len(items),
                "accepted": status_ctr.get("accepted", 0),
                "failed": status_ctr.get("failed", 0),
                "pass_rate_pct": round(status_ctr.get("accepted", 0) / len(items) * 100, 2),
                "strict_eval_ready": status_ctr.get("failed", 0) == 0,
                "partial_eval_ready": partial is not None,
                "partial_negative_count": len(partial["negative"]) if partial else 0,
                "top_hard_errors": dict(err_ctr.most_common(20)),
                "top_warnings": dict(warn_ctr.most_common(10)),
            }
            summaries.append(summary)
            if partial:
                partial_rows.append(partial)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps({"groups": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.partial_out.write_text(
        json.dumps(partial_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with args.failed_out.open("w", encoding="utf-8") as f:
        for row in failed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for s in summaries:
        print(
            f"{s['query_id']}: pass={s['pass_rate_pct']}% "
            f"strict={s['strict_eval_ready']} partial={s['partial_eval_ready']} "
            f"negatives={s['partial_negative_count']}"
        )
    print(f"summary -> {args.summary_out}")
    print(f"partial -> {args.partial_out} ({len(partial_rows)} groups)")
    print(f"failed  -> {args.failed_out} ({len(failed_rows)} items)")


if __name__ == "__main__":
    main()
