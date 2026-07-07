"""Merge multiple partial intent retrieval reports into one summary file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge partial retrieval eval reports")
    parser.add_argument("reports", nargs="+", help="partial report JSON files")
    parser.add_argument("-o", "--output", required=True, help="merged output path")
    args = parser.parse_args()

    merged: dict | None = None
    for path_str in args.reports:
        path = Path(path_str)
        report = json.loads(path.read_text(encoding="utf-8"))
        if merged is None:
            merged = {
                "eval_set": report.get("eval_set"),
                "num_queries": report.get("num_queries"),
                "default_device": report.get("default_device"),
                "batch_size": report.get("batch_size"),
                "score_cache_dir": report.get("score_cache_dir"),
                "metrics_doc": report.get("metrics_doc", {}),
                "mode_doc": report.get("mode_doc", {}),
                "models": {},
            }
        for alias, model_result in report.get("models", {}).items():
            if alias not in merged["models"]:
                merged["models"][alias] = model_result
                continue
            existing = merged["models"][alias]
            existing_modes = existing.setdefault("modes", {})
            existing_modes.update(model_result.get("modes", {}))
            for key in ("path", "backend", "device", "configured_modes"):
                if key in model_result:
                    existing[key] = model_result[key]

    if merged is None:
        raise SystemExit("no reports to merge")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {len(args.reports)} reports -> {out}")
    print(f"Models: {', '.join(merged['models'].keys())}")


if __name__ == "__main__":
    main()
