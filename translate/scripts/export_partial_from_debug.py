#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export partial MSA eval JSON from debug jsonl.")
    p.add_argument("--debug", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-out", type=Path, help="Optional accepted-only translation cache jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_rows: list[dict] = []
    cache_rows: list[dict] = []

    with args.debug.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            d = row["debug"]
            failed = set(d["qa"]["failed_items"])
            query_key = "q_000001_query" if d["query_id"] == "q_000001" else f"{d['query_id']}_query"
            if d["query_id"] in failed or query_key in failed:
                continue

            positives = [p for p in d["positive"] if p["id"] not in failed]
            negatives = [n["msa"] for n in d["negative"] if n["id"] not in failed]
            if not positives:
                continue

            out_rows.append(
                {
                    "query": d["query_msa"],
                    "positive": [p["msa"] for p in positives],
                    "negative": negatives,
                }
            )

            if args.cache_out:
                cache_rows.append({"source_idn": d["query_idn"], "msa": d["query_msa"]})
                for p in d["positive"]:
                    if p["id"] not in failed:
                        cache_rows.append({"source_idn": p["idn"], "msa": p["msa"]})
                for n in d["negative"]:
                    if n["id"] not in failed:
                        cache_rows.append({"source_idn": n["idn"], "msa": n["msa"]})

    args.output.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_rows)} groups to {args.output}")

    if args.cache_out:
        seen: set[str] = set()
        args.cache_out.parent.mkdir(parents=True, exist_ok=True)
        with args.cache_out.open("w", encoding="utf-8") as f:
            for row in cache_rows:
                if row["source_idn"] in seen:
                    continue
                seen.add(row["source_idn"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(seen)} cache entries to {args.cache_out}")


if __name__ == "__main__":
    main()
