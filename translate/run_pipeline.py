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
from idn_msa.runner import run_group, write_qa_report

DEFAULT_INPUT = Path("/data1/hcc/jiansuo/shuju/cluster_retrieval_intent_eval.json")
DEFAULT_OUTPUT_DIR = Path("/data1/hcc/jiansuo/translate/output")
DEFAULT_BASE_URL = "http://10.16.137.2:8000/v1"
DEFAULT_MODEL = "Kimi-K2.6-CT-FP8KV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Indonesian retrieval eval JSON to MSA with QA pipeline."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=20)
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
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "msa_translated.jsonl"
    qa_report = args.output_dir / "qa_report.json"

    with args.input.open(encoding="utf-8") as f:
        records = json.load(f)

    cfg = load_config()
    client = KimiClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )

    done_ids: set[str] = set()
    if args.resume and output_jsonl.exists():
        with output_jsonl.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done_ids.add(row["query_id"])

    end_index = min(args.start_index + args.max_groups, len(records))
    selected = records[args.start_index:end_index]

    groups_out: list[dict] = []
    if args.resume and qa_report.exists():
        with qa_report.open(encoding="utf-8") as f:
            old = json.load(f)
            groups_out = old.get("groups", [])

    with output_jsonl.open("a", encoding="utf-8") as out_f:
        for offset, record in enumerate(selected):
            group_idx = args.start_index + offset + 1
            group_id = f"q_{group_idx:06d}"
            if group_id in done_ids:
                logging.info("Skip completed group %s", group_id)
                continue

            logging.info("Processing group %s", group_id)
            group_result = run_group(
                client=client,
                record=record,
                group_idx=group_idx,
                cfg=cfg,
                batch_size=args.batch_size,
                enable_semantic_qa=args.enable_semantic_qa,
                enable_relation_qa=args.enable_relation_qa,
                enable_backtranslation=args.enable_backtranslation,
                relation_sample_limit=args.relation_sample_limit,
            )
            out_f.write(json.dumps(group_result, ensure_ascii=False) + "\n")
            out_f.flush()
            groups_out = [g for g in groups_out if g.get("query_id") != group_result["query_id"]]
            groups_out.append(group_result)
            write_qa_report(groups_out, qa_report)

    logging.info("Done. Output: %s", output_jsonl)


if __name__ == "__main__":
    main()
