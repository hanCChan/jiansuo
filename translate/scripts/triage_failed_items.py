#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idn_msa.config_loader import load_triage_config
from idn_msa.triage import triage_failed_item


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Triage failed reeval items by repair priority.")
    p.add_argument("--failed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    triage_cfg = load_triage_config()
    rows = []
    priority_ctr: Counter[str] = Counter()
    action_ctr: Counter[str] = Counter()

    with args.failed.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            triaged = triage_failed_item(row, triage_cfg)
            rows.append(triaged)
            priority_ctr[triaged["repair_priority"]] += 1
            action_ctr[triaged["repair_action"]] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_failed": len(rows),
        "by_priority": dict(priority_ctr),
        "by_action": dict(action_ctr),
        "to_repair": priority_ctr.get("P0", 0) + priority_ctr.get("P1", 0),
        "rule_adjust": priority_ctr.get("RULE", 0),
        "to_drop": priority_ctr.get("P2_DROP", 0),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"triage -> {args.output}")


if __name__ == "__main__":
    main()
