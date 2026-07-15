#!/usr/bin/env python3
"""Translate pingce_org/qa.json Indonesian fields to MSA for cluster retrieval eval.

Translates only Indonesian text fields:
  Question, Answer, evidence, reference, Question_cluster

Keeps unchanged:
  file_name, skill, latency, *_zh fields

Reuses the IDN->MSA translation pipeline (mask, Kimi, hard_qa, placeholder restore)
and seeds cache from the existing 37-group MSA eval dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idn_msa.config_loader import load_config
from idn_msa.expand import TranslationItem
from idn_msa.hard_qa import apply_postprocess, hard_qa_item
from idn_msa.kimi_client import KimiClient
from idn_msa.mask import annotate_item
from idn_msa.placeholder_restore import restore_kimi_placeholders
from idn_msa.pipeline import process_items_with_retry
from idn_msa.translation_cache import TranslationCache

DEFAULT_INPUT = Path("/data1/hcc/jiansuo/pingce_org/qa.json")
DEFAULT_OUTPUT = Path("/data1/hcc/jiansuo/pingce_org/qa_msa.json")
DEFAULT_IDN_EVAL = Path("/data1/hcc/jiansuo/shuju/cluster_retrieval_intent_eval.json")
DEFAULT_MSA_EVAL = Path(
    "/data1/hcc/jiansuo/results/full_run/cluster_retrieval_intent_eval_msa_full.json"
)
DEFAULT_CACHE = Path("/data1/hcc/jiansuo/translate/output/qa_translation_cache.jsonl")
DEFAULT_DEBUG = Path("/data1/hcc/jiansuo/translate/output/qa_translation_debug.jsonl")
DEFAULT_BASE_URL = "http://10.16.137.2:8000/v1"
DEFAULT_MODEL = "Kimi-K2.6-CT-FP8KV"

IDN_FIELDS = ("Question", "Answer", "evidence", "reference")
ROLE_BY_FIELD = {
    "Question": "negative",
    "Answer": "negative",
    "evidence": "negative",
    "reference": "negative",
    "Question_cluster": "negative",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate qa.json Indonesian fields to MSA.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--idn-eval", type=Path, default=DEFAULT_IDN_EVAL)
    p.add_argument("--msa-eval", type=Path, default=DEFAULT_MSA_EVAL)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--debug", type=Path, default=DEFAULT_DEBUG)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--wave-size", type=int, default=800, help="Unique strings per Kimi wave")
    p.add_argument("--max-items", type=int, default=0, help="Debug: cap untranslated items")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--assemble-only", action="store_true", help="Only build output from cache")
    p.add_argument(
        "--best-effort-finish",
        action="store_true",
        help="Backfill remaining cache entries from latest debug msa_raw, then write output",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def stable_item_id(source_idn: str, field: str) -> str:
    digest = hashlib.sha1(f"{field}|{source_idn}".encode()).hexdigest()[:12]
    return f"qa_{digest}"


def seed_cache_from_eval(cache: TranslationCache, idn_eval: Path, msa_eval: Path) -> int:
    idn_rows = json.loads(idn_eval.read_text(encoding="utf-8"))
    msa_rows = json.loads(msa_eval.read_text(encoding="utf-8"))
    added = 0
    for g_idn, g_msa in zip(idn_rows, msa_rows):
        pairs = [(g_idn["query"], g_msa["query"])]
        pairs.extend(zip(g_idn["positive"], g_msa["positive"]))
        pairs.extend(zip(g_idn["negative"], g_msa["negative"]))
        for src, tgt in pairs:
            if src and tgt and cache.get(src) is None:
                cache.set(src, tgt)
                added += 1
    return added


def collect_unique_strings(rows: list[dict]) -> dict[str, str]:
    """Map source_idn -> field label (first occurrence field name)."""
    mapping: dict[str, str] = {}
    for row in rows:
        for field in IDN_FIELDS:
            text = row.get(field, "")
            if text and text not in mapping:
                mapping[text] = field
        for text in row.get("Question_cluster") or []:
            if text and text not in mapping:
                mapping[text] = "Question_cluster"
    return mapping


def make_item(source_idn: str, field: str, batch_idx: int, cfg) -> TranslationItem:
    item = TranslationItem(
        item_id=stable_item_id(source_idn, field),
        group_id=f"qa_batch_{batch_idx:04d}",
        role=ROLE_BY_FIELD[field],
        candidate_index=0,
        source_idn=source_idn,
    )
    masked, entities, terms, actions = annotate_item(source_idn, cfg)
    item.masked_idn = masked
    item.entities_found = entities
    item.terms_found = terms
    item.actions_found = actions
    return item


def finalize_item(item: TranslationItem, cfg) -> None:
    msa, _ = restore_kimi_placeholders(item.source_idn, item.msa_raw, cfg)
    item.msa_raw = msa
    apply_postprocess(item, cfg)
    item.qa.update(hard_qa_item(item, cfg))
    if item.qa.get("hard_pass"):
        item.final_status = "accepted"
    else:
        item.final_status = "failed"


def load_debug_failures(debug_path: Path) -> set[str]:
    failed: set[str] = set()
    if not debug_path.exists():
        return failed
    with debug_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("final_status") != "accepted":
                failed.add(row["source_idn"])
    return failed


def append_debug_records(debug_path: Path, items: list[TranslationItem]) -> None:
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(
                json.dumps(
                    {
                        "item_id": item.item_id,
                        "field": item.group_id,
                        "source_idn": item.source_idn,
                        "msa_raw": item.msa_raw,
                        "final_status": item.final_status,
                        "qa": item.qa,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def assemble_qa_msa(rows: list[dict], cache: TranslationCache) -> tuple[list[dict], list[str]]:
    out_rows: list[dict] = []
    missing: list[str] = []
    for row in rows:
        new_row = dict(row)
        ok = True
        for field in IDN_FIELDS:
            src = row.get(field, "")
            msa = cache.get(src)
            if not msa:
                missing.append(src)
                ok = False
                break
            new_row[field] = msa
        if not ok:
            continue
        cluster_msa = []
        for src in row.get("Question_cluster") or []:
            msa = cache.get(src)
            if not msa:
                missing.append(src)
                ok = False
                break
            cluster_msa.append(msa)
        if not ok:
            continue
        new_row["Question_cluster"] = cluster_msa
        out_rows.append(new_row)
    return out_rows, missing


def load_latest_debug_msa(debug_path: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    if not debug_path.exists():
        return latest
    with debug_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            msa = row.get("msa_raw")
            src = row.get("source_idn")
            if src and msa:
                latest[src] = msa
    return latest


def backfill_best_effort(
    cache: TranslationCache,
    pending: list[str],
    debug_path: Path,
) -> int:
    latest = load_latest_debug_msa(debug_path)
    added = 0
    for src in pending:
        if cache.get(src):
            continue
        msa = latest.get(src)
        if msa:
            cache.set(src, msa)
            added += 1
    return added


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = load_config()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    cache = TranslationCache(args.cache)
    seeded = seed_cache_from_eval(cache, args.idn_eval, args.msa_eval)
    logging.info("Seeded %s entries from eval cache (total cache=%s)", seeded, len(cache))

    unique = collect_unique_strings(rows)
    pending = [src for src in unique if not cache.get(src)]
    logging.info(
        "qa unique strings=%s cached=%s pending=%s",
        len(unique),
        len(unique) - len(pending),
        len(pending),
    )

    if args.max_items:
        pending = pending[: args.max_items]

    if not args.assemble_only and pending:
        if args.resume and args.debug.exists():
            prev_failed = load_debug_failures(args.debug)
            logging.info("Resume: previous failed unique=%s", len(prev_failed))

        client = KimiClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
        )

        for wave_idx, start in enumerate(range(0, len(pending), args.wave_size), start=1):
            wave_src = pending[start : start + args.wave_size]
            items = [
                make_item(src, unique[src], wave_idx, cfg)
                for src in wave_src
                if not cache.get(src)
            ]
            if not items:
                continue

            logging.info(
                "Wave %s: translating %s items (%s/%s)",
                wave_idx,
                len(items),
                start + len(wave_src),
                len(pending),
            )
            process_items_with_retry(
                client=client,
                items=items,
                cfg=cfg,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                enable_semantic_qa=False,
                enable_relation_qa=False,
                enable_backtranslation=False,
                cache=cache,
            )

            for item in items:
                if item.msa_raw:
                    msa, _ = restore_kimi_placeholders(item.source_idn, item.msa_raw, cfg)
                    item.msa_raw = msa
                if item.final_status == "accepted" and item.msa_raw:
                    cache.set(item.source_idn, item.msa_raw)
                elif item.msa_raw and item.qa.get("hard_pass"):
                    cache.set(item.source_idn, item.msa_raw)

            append_debug_records(args.debug, items)
            cache.save(args.cache)
            accepted = sum(1 for i in items if i.final_status == "accepted")
            logging.info(
                "Wave %s done: accepted=%s failed=%s cache=%s",
                wave_idx,
                accepted,
                len(items) - accepted,
                len(cache),
            )

        cache.save(args.cache)

    if args.best_effort_finish and pending:
        added = backfill_best_effort(cache, pending, args.debug)
        logging.info("Best-effort backfill from debug: %s/%s pending", added, len(pending))
        cache.save(args.cache)

    out_rows, missing = assemble_qa_msa(rows, cache)
    if missing:
        unique_missing = len(set(missing))
        logging.warning(
            "Incomplete translation: %s rows ready / %s total, missing unique strings=%s",
            len(out_rows),
            len(rows),
            unique_missing,
        )
        summary = {
            "input_rows": len(rows),
            "output_rows": len(out_rows),
            "missing_unique_strings": unique_missing,
            "cache_entries": len(cache),
            "best_effort_finish": bool(args.best_effort_finish),
        }
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if len(out_rows) < len(rows) and not args.best_effort_finish:
            raise SystemExit(
                f"Translation incomplete ({len(out_rows)}/{len(rows)} rows). "
                f"Re-run with --resume after checking {args.debug}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s rows -> %s", len(out_rows), args.output)


if __name__ == "__main__":
    main()
