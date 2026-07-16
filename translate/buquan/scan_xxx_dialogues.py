#!/usr/bin/env python3
"""Scan XXX distribution and build local clusters (no Kimi)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xxx_utils import build_all_clusters, classify_hint_type, find_xxx_tokens, has_xxx

DEFAULT_INPUT = Path("/data1/hcc/jiansuo/dh/dialogue_20260615_BCA_clean_test_msa.json")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "dialogue_test_xxx_scan.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan XXX turns and local clusters.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--gap-max", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dialogues = json.loads(args.input.read_text(encoding="utf-8"))

    turns_total = 0
    turns_with_xxx = 0
    hint_ctr: Counter[str] = Counter()
    cluster_sizes: list[int] = []
    dialogues_with_xxx = 0

    by_dialogue, clusters = build_all_clusters(dialogues, gap_max=args.gap_max)

    for dlg in dialogues:
        did = int(dlg["dialogue_id"])
        refs = by_dialogue[did]
        turns_total += len(refs)
        local_xxx = sum(1 for r in refs if has_xxx(r.content_msa))
        if local_xxx:
            dialogues_with_xxx += 1
            turns_with_xxx += local_xxx
            for r in refs:
                if has_xxx(r.content_msa):
                    hint_ctr[r.hint_type()] += 1

    for cluster in clusters:
        cluster_sizes.append(len(cluster.turns))

    heavy_dialogues = []
    for dlg in dialogues:
        did = int(dlg["dialogue_id"])
        n = sum(1 for t in dlg.get("turns", []) if has_xxx(t.get("content_msa", "")))
        if n:
            heavy_dialogues.append({"dialogue_id": did, "xxx_turns": n})
    heavy_dialogues.sort(key=lambda x: x["xxx_turns"], reverse=True)

    report = {
        "input": str(args.input),
        "gap_max": args.gap_max,
        "dialogues_total": len(dialogues),
        "dialogues_with_xxx": dialogues_with_xxx,
        "turns_total": turns_total,
        "turns_with_xxx": turns_with_xxx,
        "clusters_total": len(clusters),
        "cluster_size": {
            "min": min(cluster_sizes) if cluster_sizes else 0,
            "max": max(cluster_sizes) if cluster_sizes else 0,
            "avg": round(sum(cluster_sizes) / len(cluster_sizes), 2) if cluster_sizes else 0,
        },
        "hint_type_distribution": dict(hint_ctr),
        "top_dialogues_by_xxx_turns": heavy_dialogues[:20],
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "dialogue_id": c.dialogue_id,
                "topic_hint": c.topic_hint(),
                "turns": [t.turn for t in c.turns],
                "xxx_count": sum(len(find_xxx_tokens(t.content_msa)) for t in c.turns),
            }
            for c in clusters
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "clusters"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
