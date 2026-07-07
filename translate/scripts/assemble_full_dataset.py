#!/usr/bin/env python3
"""Build full 37-group MSA dataset: cache from group1 + translate 36 missing queries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idn_msa.config_loader import load_config
from idn_msa.expand import TranslationItem, expand_record
from idn_msa.hard_qa import apply_postprocess, hard_qa_item
from idn_msa.kimi_client import KimiClient
from idn_msa.mask import annotate_item
from idn_msa.placeholder_restore import restore_kimi_placeholders
from idn_msa.translate import translate_batch
from idn_msa.translation_cache import TranslationCache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble full MSA eval dataset from group1 cache.")
    p.add_argument("--input", type=Path, required=True, help="Source IDN eval JSON")
    p.add_argument("--group1-strict", type=Path, required=True, help="Group1 strict MSA JSON")
    p.add_argument("--output", type=Path, required=True, help="Full 37-group MSA JSON")
    p.add_argument("--cache-out", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    p.add_argument("--missing-queries-out", type=Path, help="Write translated missing queries jsonl")
    p.add_argument("--concurrency", type=int, default=4)
    return p.parse_args()


def build_group1_cache(records: list[dict], group1_strict: dict) -> TranslationCache:
    cache = TranslationCache()
    g1 = records[0]
    s = group1_strict
    cache.set(g1["query"], s["query"])
    for idn, msa in zip(g1["positive"], s["positive"]):
        cache.set(idn, msa)
    for idn, msa in zip(g1["negative"], s["negative"]):
        cache.set(idn, msa)
    return cache


def prepare_item(item: TranslationItem, cfg, cache: TranslationCache) -> None:
    masked, entities, terms, actions = annotate_item(item.source_idn, cfg)
    item.masked_idn = masked
    item.entities_found = entities
    item.terms_found = terms
    item.actions_found = actions
    cached = cache.get(item.source_idn)
    if cached:
        msa, _ = restore_kimi_placeholders(item.source_idn, cached, cfg)
        item.msa_raw = msa
        apply_postprocess(item, cfg)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    group1_strict = json.loads(args.group1_strict.read_text(encoding="utf-8"))[0]
    cache = build_group1_cache(records, group1_strict)

    # Find and translate missing queries (groups 2-37)
    g1_texts = {records[0]["query"]} | set(records[0]["positive"]) | set(records[0]["negative"])
    missing_items: list[TranslationItem] = []
    for gi, rec in enumerate(records[1:], start=2):
        q = rec["query"]
        if q not in g1_texts and not cache.get(q):
            item = TranslationItem(
                item_id=f"q_{gi:06d}_query",
                group_id=f"q_{gi:06d}",
                role="query",
                candidate_index=0,
                source_idn=q,
            )
            prepare_item(item, cfg, cache)
            missing_items.append(item)

    translated_new = 0
    if missing_items:
        client = KimiClient()
        translate_batch(client, missing_items, cfg, batch_size=10, concurrency=args.concurrency)
        for item in missing_items:
            msa, _ = restore_kimi_placeholders(item.source_idn, item.msa_raw, cfg)
            item.msa_raw = msa
            apply_postprocess(item, cfg)
            qa = hard_qa_item(item, cfg)
            item.qa = qa
            cache.set(item.source_idn, item.msa_raw)
            translated_new += 1
            if args.missing_queries_out:
                args.missing_queries_out.parent.mkdir(parents=True, exist_ok=True)
                with args.missing_queries_out.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "item_id": item.item_id,
                                "source_idn": item.source_idn,
                                "msa": item.msa_raw,
                                "hard_pass": qa["hard_pass"],
                                "hard_errors": qa.get("hard_errors", []),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    # Assemble all 37 groups
    full_output: list[dict] = []
    missing_any: list[dict] = []
    ent_count = 0
    for gi, rec in enumerate(records, start=1):
        group = {"query": "", "positive": [], "negative": []}
        for role, texts in [
            ("query", [rec["query"]]),
            ("positive", rec.get("positive", [])),
            ("negative", rec.get("negative", [])),
        ]:
            for idn in texts:
                msa = cache.get(idn)
                if not msa:
                    missing_any.append({"group": gi, "role": role, "source_idn": idn})
                    continue
                msa, _ = restore_kimi_placeholders(idn, msa, cfg)
                if "<ENT" in msa.upper():
                    ent_count += 1
                if role == "query":
                    group["query"] = msa
                else:
                    group[role].append(msa)
        full_output.append(group)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(full_output, ensure_ascii=False, indent=2), encoding="utf-8")
    cache.save(args.cache_out)

    summary = {
        "total_groups": len(records),
        "cache_entries": len(cache),
        "new_queries_translated": translated_new,
        "missing_queries_expected": 36,
        "cache_misses": len(missing_any),
        "ent_placeholder_remaining": ent_count,
        "group1_negatives": len(full_output[0]["negative"]),
        "per_group_negative_count": len(full_output[1]["negative"]) if len(full_output) > 1 else 0,
        "complete": len(missing_any) == 0 and ent_count == 0,
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing_any:
        print("MISSING:", missing_any[:5])
        sys.exit(1)


if __name__ == "__main__":
    main()
