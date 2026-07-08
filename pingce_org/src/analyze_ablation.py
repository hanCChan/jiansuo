"""Analyze ablation results: leaderboard, badcases, BGE mode top10 diff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_ablation_results(report: dict) -> list[dict]:
    rows: list[dict] = []
    for alias, model in report.get("models", {}).items():
        for mode, result in model.get("modes", {}).items():
            summary = result.get("summary", {})
            rows.append(
                {
                    "alias": alias,
                    "ablation_label": model.get("ablation_label"),
                    "backend": model.get("backend"),
                    "mode": mode,
                    "recall@1": summary.get("recall@1"),
                    "recall@3": summary.get("recall@3"),
                    "recall@5": summary.get("recall@5"),
                    "eval_sec": result.get("eval_sec"),
                    "hybrid_weights": model.get("hybrid_weights"),
                    "hybrid_sparse_weight": model.get("hybrid_sparse_weight"),
                    "query_style": model.get("query_style"),
                    "task_description": model.get("task_description"),
                }
            )
    rows.sort(key=lambda r: (-(r["recall@1"] or 0), -(r["recall@3"] or 0)))
    return rows


def compare_bge_multimode_top10(multimode_report: dict) -> dict:
    modes = ("dense", "colbert", "dense+sparse", "hybrid")
    model = multimode_report.get("models", {}).get("bge_m3", {})
    per_mode: dict[str, list] = {}
    for mode in modes:
        block = model.get("modes", {}).get(mode, {})
        per_mode[mode] = block.get("per_query", [])

    if not all(per_mode[m] for m in ("dense", "colbert", "dense+sparse")):
        return {"error": "missing bge_m3 per_query in multimode report"}

    n = len(per_mode["dense"])
    pairwise: dict[str, dict] = {}
    for a, b in (("dense", "colbert"), ("dense", "dense+sparse"), ("colbert", "dense+sparse")):
        same_top10 = 0
        same_top1 = 0
        diffs = []
        for i in range(n):
            qa = per_mode[a][i]
            qb = per_mode[b][i]
            ta = qa.get("top10_indices") or qa.get("topk_indices")
            tb = qb.get("top10_indices") or qb.get("topk_indices")
            if ta is None or tb is None:
                continue
            if ta == tb:
                same_top10 += 1
            if ta and tb and ta[0] == tb[0]:
                same_top1 += 1
            if ta != tb:
                diffs.append(
                    {
                        "cluster_id": qa.get("cluster_id", i),
                        "query": qa.get("query", "")[:120],
                        f"{a}_top10": ta,
                        f"{b}_top10": tb,
                    }
                )
        pairwise[f"{a}_vs_{b}"] = {
            "same_top10_count": same_top10,
            "same_top1_count": same_top1,
            "total_queries": n,
            "same_top10_ratio": round(same_top10 / n, 4) if n else 0,
            "same_top1_ratio": round(same_top1 / n, 4) if n else 0,
            "sample_diffs": diffs[:5],
        }
    return pairwise


def build_badcase_table(report: dict, *, ref_alias: str = "bge_hybrid_baseline") -> dict:
    ref_model = report.get("models", {}).get(ref_alias)
    if ref_model is None:
        # fallback: any model with per_query
        for alias, model in report.get("models", {}).items():
            for mode_block in model.get("modes", {}).values():
                if mode_block.get("per_query"):
                    ref_alias = alias
                    ref_model = model
                    break
            if ref_model:
                break
    if ref_model is None:
        return {"error": "no reference model with per_query"}

    ref_mode = next(iter(ref_model["modes"]))
    ref_rows = {
        r["cluster_id"]: r for r in ref_model["modes"][ref_mode]["per_query"]
    }

    comparisons: list[dict] = []
    for alias, model in report.get("models", {}).items():
        if alias == ref_alias:
            continue
        for mode, block in model.get("modes", {}).items():
            for row in block.get("per_query", []):
                cid = row["cluster_id"]
                ref = ref_rows.get(cid)
                if ref is None:
                    continue
                ref_hit = ref.get("hit@1", ref.get("recall@1", 0) == 1.0)
                cur_hit = row.get("hit@1", row.get("recall@1", 0) == 1.0)
                if ref_hit and not cur_hit:
                    comparisons.append(
                        {
                            "cluster_id": cid,
                            "pattern": f"{ref_alias}_hit_{alias}_miss",
                            "query": row.get("query"),
                            "ref_rank": ref.get("pos_rank"),
                            "cur_rank": row.get("pos_rank"),
                        }
                    )
                elif (not ref_hit) and cur_hit:
                    comparisons.append(
                        {
                            "cluster_id": cid,
                            "pattern": f"{ref_alias}_miss_{alias}_hit",
                            "query": row.get("query"),
                            "ref_rank": ref.get("pos_rank"),
                            "cur_rank": row.get("pos_rank"),
                        }
                    )

    all_miss: dict[int, list[str]] = {}
    all_hit: dict[int, list[str]] = {}
    for alias, model in report.get("models", {}).items():
        for mode, block in model.get("modes", {}).items():
            for row in block.get("per_query", []):
                cid = row["cluster_id"]
                hit = row.get("hit@1", row.get("recall@1", 0) == 1.0)
                if hit:
                    all_hit.setdefault(cid, []).append(alias)
                else:
                    all_miss.setdefault(cid, []).append(alias)

    universal_miss = [
        {"cluster_id": cid, "query": next(
            (r["query"] for m in report["models"].values()
             for b in m["modes"].values()
             for r in b["per_query"] if r["cluster_id"] == cid), ""
        )}
        for cid, missers in all_miss.items()
        if len(missers) == len(report.get("models", {}))
    ]

    return {
        "reference": ref_alias,
        "pairwise_diff_count": len(comparisons),
        "pairwise_samples": comparisons[:30],
        "universal_miss": universal_miss[:20],
        "per_query_hit_counts": {
            str(cid): {"hit": hits, "miss": len(report["models"]) - len(hits)}
            for cid, hits in sorted(all_hit.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ablation retrieval reports")
    parser.add_argument("--ablation-report", required=True)
    parser.add_argument("--multimode-report", default=None)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    ablation = load_report(Path(args.ablation_report))
    leaderboard = flatten_ablation_results(ablation)
    analysis = {
        "leaderboard": leaderboard,
        "badcases": build_badcase_table(ablation),
    }

    if args.multimode_report:
        mm_path = Path(args.multimode_report)
        if mm_path.is_file():
            analysis["bge_mode_top10_check"] = compare_bge_multimode_top10(
                load_report(mm_path)
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Analysis saved -> {out}")
    print("\n=== Ablation Leaderboard (recall@1) ===")
    for row in leaderboard[:15]:
        label = row.get("ablation_label") or row["alias"]
        print(
            f"  {row['recall@1']:.4f}  R@3={row['recall@3']:.4f}  "
            f"R@5={row['recall@5']:.4f}  {label}"
        )


if __name__ == "__main__":
    main()
