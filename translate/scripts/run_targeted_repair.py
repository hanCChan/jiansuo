#!/usr/bin/env python3
"""Orchestrate triage -> repair -> merge for targeted failed-item recovery."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run targeted repair pipeline on failed reeval items.")
    p.add_argument("--failed", type=Path, required=True)
    p.add_argument("--partial", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    triage = out / "failed_after_reeval_triage.jsonl"
    triage_summary = out / "triage_summary.json"
    repaired = out / "repaired_items.jsonl"
    dropped = out / "dropped_items.jsonl"
    repair_summary = out / "repair_summary.json"
    merged = out / "cluster_retrieval_intent_eval_msa_repaired_partial.json"
    merge_summary = out / "merge_summary.json"

    run([
        sys.executable, str(SCRIPTS / "triage_failed_items.py"),
        "--failed", str(args.failed),
        "--output", str(triage),
        "--summary-out", str(triage_summary),
    ])

    repair_cmd = [
        sys.executable, str(SCRIPTS / "repair_failed_items.py"),
        "--triage", str(triage),
        "--repaired-out", str(repaired),
        "--dropped-out", str(dropped),
        "--summary-out", str(repair_summary),
        "--concurrency", str(args.concurrency),
    ]
    if args.dry_run:
        repair_cmd.append("--dry-run")
    run(repair_cmd)

    if not args.dry_run:
        run([
            sys.executable, str(SCRIPTS / "merge_repaired_partial.py"),
            "--partial", str(args.partial),
            "--repaired", str(repaired),
            "--output", str(merged),
            "--summary-out", str(merge_summary),
        ])


if __name__ == "__main__":
    main()
