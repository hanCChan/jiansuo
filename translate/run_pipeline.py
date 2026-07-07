#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from idn_msa.config_loader import load_config
from idn_msa.kimi_client import KimiClient
from idn_msa.runner import run_group, write_msa_eval_json, write_qa_report
from idn_msa.translation_cache import TranslationCache

DEFAULT_INPUT = Path("/data1/hcc/jiansuo/shuju/cluster_retrieval_intent_eval.json")
DEFAULT_OUTPUT_DIR = Path("/data1/hcc/jiansuo/translate/output")
DEFAULT_CACHE = DEFAULT_OUTPUT_DIR / "translation_cache.jsonl"
DEFAULT_BASE_URL = "http://10.16.137.2:8000/v1"
DEFAULT_MODEL = "Kimi-K2.6-CT-FP8KV"
MSA_EVAL_NAME = "cluster_retrieval_intent_eval_msa.json"
MSA_PARTIAL_NAME = "cluster_retrieval_intent_eval_msa_partial.json"
MSA_DEBUG_NAME = "cluster_retrieval_intent_eval_msa_debug.jsonl"
QA_REPORT_NAME = "qa_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Indonesian retrieval eval JSON to MSA with QA pipeline."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel Kimi requests per group (one batch per request)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Kimi thinking via chat_template_kwargs.enable_thinking",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=1, help="Number of query groups to process")
    parser.add_argument("--enable-semantic-qa", action="store_true")
    parser.add_argument("--enable-relation-qa", action="store_true")
    parser.add_argument("--enable-backtranslation", action="store_true")
    parser.add_argument(
        "--relation-sample-limit",
        type=int,
        default=20,
        help="Only run relation QA on first N candidates per group (0 = all)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="Global IDN->MSA cache jsonl (dedup across groups)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_debug_records(debug_path: Path) -> list[dict]:
    if not debug_path.exists():
        return []
    records: list[dict] = []
    with debug_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def rebuild_outputs(debug_records: list[dict], output_dir: Path) -> None:
    strict_records = [record["simple"] for record in debug_records if record.get("eval_ready")]
    partial_records = [record["partial"] for record in debug_records if record.get("partial_ready")]
    write_msa_eval_json(strict_records, output_dir / MSA_EVAL_NAME)
    write_msa_eval_json(partial_records, output_dir / MSA_PARTIAL_NAME)
    write_qa_report(debug_records, output_dir / QA_REPORT_NAME)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    msa_eval_path = args.output_dir / MSA_EVAL_NAME
    debug_jsonl_path = args.output_dir / MSA_DEBUG_NAME
    qa_report_path = args.output_dir / QA_REPORT_NAME

    with args.input.open(encoding="utf-8") as f:
        records = json.load(f)

    cfg = load_config()
    cache = TranslationCache(args.cache)
    if cache:
        logging.info("Loaded translation cache entries: %s", len(cache))
    client = KimiClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        enable_thinking=args.enable_thinking,
    )

    debug_records = load_debug_records(debug_jsonl_path) if args.resume else []
    done_ids = {record["query_id"] for record in debug_records}

    end_index = min(args.start_index + args.max_groups, len(records))
    selected = records[args.start_index:end_index]

    with debug_jsonl_path.open("a", encoding="utf-8") as debug_f:
        for offset, record in enumerate(selected):
            group_idx = args.start_index + offset + 1
            group_id = f"q_{group_idx:06d}"
            if group_id in done_ids:
                logging.info("Skip completed group %s", group_id)
                continue

            logging.info("Processing group %s", group_id)
            result = run_group(
                client=client,
                record=record,
                group_idx=group_idx,
                cfg=cfg,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                enable_semantic_qa=args.enable_semantic_qa,
                enable_relation_qa=args.enable_relation_qa,
                enable_backtranslation=args.enable_backtranslation,
                relation_sample_limit=args.relation_sample_limit,
                cache=cache,
            )
            debug_records = [r for r in debug_records if r["query_id"] != result["query_id"]]
            debug_records.append(result)
            debug_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            debug_f.flush()
            rebuild_outputs(debug_records, args.output_dir)

            if result["eval_ready"]:
                logging.info("Group %s accepted for strict eval output", group_id)
            elif result.get("partial_ready"):
                logging.warning(
                    "Group %s partial eval ready (%s accepted negatives); strict eval skipped",
                    group_id,
                    len(result["partial"]["negative"]),
                )
            else:
                logging.warning(
                    "Group %s failed QA; kept in debug only: %s",
                    group_id,
                    result["debug"]["qa"]["failed_items"],
                )

    cache.save(args.cache)
    logging.info("Saved translation cache entries: %s -> %s", len(cache), args.cache)
    logging.info("MSA strict eval: %s", msa_eval_path)
    logging.info("MSA partial eval: %s", args.output_dir / MSA_PARTIAL_NAME)
    logging.info("Debug: %s", debug_jsonl_path)
    logging.info("QA report: %s", qa_report_path)


if __name__ == "__main__":
    main()
