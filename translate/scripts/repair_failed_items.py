#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idn_msa.config_loader import load_config
from idn_msa.expand import TranslationItem
from idn_msa.hard_qa import apply_postprocess, hard_qa_item
from idn_msa.kimi_client import KimiClient
from idn_msa.mask import annotate_item
from idn_msa.mask import restore_text

REPAIR_SYSTEM = """你是金融客服语料翻译修复器。请根据错误原因修复 MSA 译文。

要求：
1. 只输出 JSON: {"translation_msa": "..."}
2. 不要解释，不要 Markdown
3. 不要改变原始业务意图
4. 必须保留源文中的品牌/产品/服务名（BCA, myBCA, Flazz, PIN, OTP, QRIS, OneKlik, KlikBCA 等）
5. password -> كلمة المرور；PIN 必须保留 PIN 或 الرقم السري (PIN)
6. membuka blokir -> فك الحظر / إلغاء الحظر
7. 禁止使用 <ENT_...> 占位符，必须写出真实实体名
8. 不要删除产品名，不要留下空洞句子
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Targeted repair for P0/P1 failed items.")
    p.add_argument("--triage", type=Path, required=True)
    p.add_argument("--repaired-out", type=Path, required=True)
    p.add_argument("--dropped-out", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    p.add_argument("--base-url", default="http://10.16.137.2:8000/v1")
    p.add_argument("--model", default="Kimi-K2.6-CT-FP8KV")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="Skip Kimi calls; only classify outputs")
    return p.parse_args()


def _to_item(row: dict, cfg) -> TranslationItem:
    item = TranslationItem(
        item_id=row["item_id"],
        group_id="q_000001",
        role=row["role"],
        candidate_index=0,
        source_idn=row["source_idn"],
        msa_raw=row["msa"],
    )
    masked, entities, terms, actions = annotate_item(item.source_idn, cfg)
    item.masked_idn = masked
    item.entities_found = entities
    item.terms_found = terms
    item.actions_found = actions
    return item


def _repair_one(client: KimiClient, row: dict, cfg) -> dict:
    user_prompt = json.dumps(
        {
            "item_id": row["item_id"],
            "role": row["role"],
            "source_idn": row["source_idn"],
            "previous_translation_msa": row["msa"],
            "hard_errors": row.get("hard_errors", []),
            "repair_reason": row.get("reason", ""),
        },
        ensure_ascii=False,
    )
    result = client.chat_json(REPAIR_SYSTEM, user_prompt)
    msa = restore_text(result["translation_msa"], cfg)
    item = _to_item({**row, "msa": msa}, cfg)
    apply_postprocess(item, cfg)
    qa = hard_qa_item(item, cfg)
    return {
        **row,
        "msa_before": row["msa"],
        "msa": msa,
        "qa_after_repair": qa,
        "repair_status": "accepted" if qa["hard_pass"] else "failed",
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()

    triage_rows = []
    with args.triage.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                triage_rows.append(json.loads(line))

    to_repair = [r for r in triage_rows if r["repair_priority"] in {"P0", "P1"}]
    rule_rows = [r for r in triage_rows if r["repair_priority"] == "RULE"]
    drop_rows = [r for r in triage_rows if r["repair_priority"] == "P2_DROP"]

    repaired_rows: list[dict] = []
    still_failed: list[dict] = []

    if args.dry_run:
        for row in to_repair:
            row = {**row, "repair_status": "dry_run"}
            repaired_rows.append(row)
    else:
        client = KimiClient(base_url=args.base_url, model=args.model)
        workers = min(args.concurrency, max(len(to_repair), 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_repair_one, client, row, cfg): row for row in to_repair}
            for fut in as_completed(futures):
                out = fut.result()
                if out["repair_status"] == "accepted":
                    repaired_rows.append(out)
                else:
                    still_failed.append(out)

    # RULE items: re-QA with current rules only
    for row in rule_rows:
        item = _to_item(row, cfg)
        apply_postprocess(item, cfg)
        qa = hard_qa_item(item, cfg)
        out = {**row, "qa_after_repair": qa, "repair_status": "accepted" if qa["hard_pass"] else "rule_still_fail"}
        if qa["hard_pass"]:
            repaired_rows.append(out)
        else:
            still_failed.append(out)

    dropped = list(drop_rows)
    for row in still_failed:
        if row.get("role") == "negative":
            row = {**row, "repair_priority": "P2_DROP", "repair_action": "drop", "reason": "repair failed or low value"}
            dropped.append(row)

    args.repaired_out.parent.mkdir(parents=True, exist_ok=True)
    with args.repaired_out.open("w", encoding="utf-8") as f:
        for row in repaired_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.dropped_out.open("w", encoding="utf-8") as f:
        for row in dropped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    status_ctr = Counter(r.get("repair_status") for r in repaired_rows + still_failed)
    summary = {
        "failed_before": len(triage_rows),
        "rule_false_positive": len(rule_rows),
        "sent_to_repair": len(to_repair),
        "repair_accepted": sum(1 for r in repaired_rows if r.get("repair_status") == "accepted"),
        "repair_failed": len(still_failed),
        "dropped": len(dropped),
        "by_repair_status": dict(status_ctr),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
