"""Evaluate retrieval with cluster-averaged candidate scores.

Each FAQ candidate is scored as the mean similarity between the query and:
  [original Question] + Question_cluster (5 variants from qa.json)
Ranking and recall@k use these averaged scores.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from embedding_backends import create_backend  # noqa: E402
from eval_intent_retrieval import (  # noqa: E402
    aggregate_metrics,
    load_eval,
    metrics_from_scores,
    normalize_model_cfg,
    release_backend,
    resolve_eval_path,
    resolve_path,
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_qa_index(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for row in rows:
        question = row["Question"]
        cluster = row.get("Question_cluster") or []
        if not cluster:
            raise ValueError(f"qa entry missing Question_cluster: {question[:80]}")
        index[question] = row
    return index


def candidate_variants(question: str, qa_index: dict[str, dict]) -> list[str]:
    row = qa_index.get(question)
    if row is None:
        raise KeyError(f"question not found in qa index: {question[:80]}")
    variants = [question, *list(row["Question_cluster"])]
    return variants


def score_candidates_cluster_avg(
    backend,
    query: str,
    candidates: list[str],
    qa_index: dict[str, dict],
) -> np.ndarray:
    scores = np.empty(len(candidates), dtype=np.float32)
    for i, cand in enumerate(candidates):
        variants = candidate_variants(cand, qa_index)
        variant_scores = backend.score(query, variants)
        scores[i] = float(np.mean(variant_scores))
    return scores


def eval_backend_on_dataset(
    data: list[dict],
    backend,
    qa_index: dict[str, dict],
    *,
    cache_dir: Path | None = None,
    cache_tag: str = "",
) -> list[dict]:
    rows: list[dict] = []
    for i, item in enumerate(data):
        candidates = item["positive"] + item["negative"]
        labels = [1] * len(item["positive"]) + [0] * len(item["negative"])

        scores: np.ndarray | None = None
        cache_path = None
        if cache_dir is not None:
            cache_path = cache_dir / cache_tag / f"{i:04d}.npy"
            if cache_path.is_file():
                scores = np.load(cache_path)

        if scores is None:
            scores = score_candidates_cluster_avg(backend, item["query"], candidates, qa_index)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, scores)

        row = metrics_from_scores(scores, labels)
        row["cluster_id"] = i
        row["query"] = item["query"]
        rows.append(row)
    return rows


def eval_cache_tag(
    eval_path: Path,
    qa_path: Path,
    alias: str,
    mode: str,
    device: str,
    model_path: str,
) -> str:
    eval_stat = eval_path.stat()
    qa_stat = qa_path.stat()
    digest = hashlib.sha1(
        f"cluster_avg|{eval_path.resolve()}|{eval_stat.st_mtime_ns}|{eval_stat.st_size}|"
        f"{qa_path.resolve()}|{qa_stat.st_mtime_ns}|{qa_stat.st_size}|"
        f"{alias}|{mode}|{device}|{model_path}".encode()
    ).hexdigest()[:12]
    return f"{alias}__{mode}__{digest}"


def resolve_qa_path(args: argparse.Namespace, cfg: dict) -> Path:
    eval_cfg = cfg.get("eval", {})
    cluster_cfg = eval_cfg.get("cluster", {})
    if args.qa_json:
        return resolve_path(args.qa_json)
    if cluster_cfg.get("qa_json"):
        return resolve_path(cluster_cfg["qa_json"])
    return resolve_path("./output/qa.json")


def print_summary(report: dict) -> None:
    print("\n=== Intent Retrieval Benchmark (cluster-avg score) ===")
    print(f"eval_set: {report['eval_set']}")
    print(f"qa_json: {report['qa_json']}")
    print(f"queries: {report['num_queries']}")
    print(
        "scoring: mean(query vs [Question] + 5x Question_cluster) per candidate, then recall@k"
    )
    for alias, model_result in report["models"].items():
        print(f"\n## {alias} ({model_result['path']}, backend={model_result['backend']})")
        for mode, result in model_result["modes"].items():
            m = result["summary"]
            print(f"  [{mode}]")
            print(
                f"    recall@1={m['recall@1']:.4f}  recall@3={m['recall@3']:.4f}  "
                f"recall@5={m['recall@5']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark embeddings with Question+Question_cluster averaged scores"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None, help="eval JSON path")
    parser.add_argument(
        "--qa-json",
        default=None,
        help="FAQ JSON with Question_cluster; default from config eval.cluster.qa_json",
    )
    parser.add_argument("--models", nargs="*", default=None, help="model aliases")
    parser.add_argument("--modes", nargs="*", default=None, help="override modes")
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for non-BGE backends; BGE-M3 accepts cuda:4, 4, "
        "4,5,6,7, or all (persistent multi-GPU pool). Default from config eval.device",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="encoding batch size")
    parser.add_argument("--cache-dir", default=None, help="score cache dir")
    parser.add_argument("--no-cache", action="store_true", help="disable score cache")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    eval_cfg = cfg.get("eval", {})
    cluster_cfg = eval_cfg.get("cluster", {})
    emb_cfg = cfg.get("embedding", {})

    eval_path = resolve_eval_path(args, cfg)
    qa_path = resolve_qa_path(args, cfg)
    if not eval_path.is_file():
        raise SystemExit(f"eval set not found: {eval_path}")
    if not qa_path.is_file():
        raise SystemExit(f"qa json not found: {qa_path}")

    data = load_eval(eval_path)
    qa_index = load_qa_index(qa_path)

    raw_models: dict = eval_cfg.get("models", {})
    if not raw_models:
        raise SystemExit("config.eval.models is empty")

    selected = args.models or list(raw_models.keys())
    unknown = [m for m in selected if m not in raw_models]
    if unknown:
        raise SystemExit(f"unknown model aliases: {unknown}; available={list(raw_models)}")

    device = args.device or eval_cfg.get("device", emb_cfg.get("device", "cuda"))
    batch_size = args.batch_size or int(
        eval_cfg.get("batch_size", emb_cfg.get("batch_size", 64))
    )

    cache_dir: Path | None = None
    if not args.no_cache:
        cache_arg = args.cache_dir or cluster_cfg.get("cache_dir")
        if cache_arg:
            cache_dir = resolve_path(cache_arg)

    report = {
        "eval_set": str(eval_path),
        "qa_json": str(qa_path),
        "num_queries": len(data),
        "device": device,
        "batch_size": batch_size,
        "score_cache_dir": str(cache_dir) if cache_dir else None,
        "scoring_doc": (
            "For each candidate FAQ, score = mean similarity between query and "
            "[Question, ...Question_cluster (5 variants from qa.json)]. "
            "Candidates are ranked by this average score; recall@k follows the "
            "same pos/neg mixed pool as eval_intent_retrieval.py."
        ),
        "metrics_doc": {
            "recall@1": "fraction of positive samples whose avg-score rank is <= 1",
            "recall@3": "fraction of positive samples whose avg-score rank is <= 3",
            "recall@5": "fraction of positive samples whose avg-score rank is <= 5",
        },
        "models": {},
    }

    out_dir = resolve_path(eval_cfg.get("report_dir", "output/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = cluster_cfg.get("report_file", "intent_retrieval_cluster_eval.json")

    for alias in selected:
        model_cfg = normalize_model_cfg(raw_models[alias])
        modes = args.modes or model_cfg["modes"]

        model_report = {
            "path": model_cfg["path"],
            "backend": model_cfg["backend"],
            "configured_modes": model_cfg["modes"],
            "modes": {},
        }

        backend = None
        try:
            for mode in modes:
                print(f"Evaluating {alias} / {mode} (cluster-avg) ...")
                if backend is None:
                    t0 = time.time()
                    backend = create_backend(model_cfg, mode, device, batch_size)
                    load_sec = time.time() - t0
                    print(f"  loaded {alias} in {load_sec:.1f}s")
                else:
                    backend.set_mode(mode)
                    load_sec = 0.0
                    print(f"  reused cached model for {alias} / {mode}")

                t1 = time.time()
                cache_tag = eval_cache_tag(
                    eval_path, qa_path, alias, mode, device, model_cfg["path"]
                )
                per_query = eval_backend_on_dataset(
                    data,
                    backend,
                    qa_index,
                    cache_dir=cache_dir,
                    cache_tag=cache_tag,
                )
                eval_sec = time.time() - t1
                model_report["modes"][mode] = {
                    "mode": mode,
                    "load_sec": round(load_sec, 2),
                    "eval_sec": round(eval_sec, 2),
                    "score_cache_tag": cache_tag if cache_dir else None,
                    "summary": aggregate_metrics(per_query),
                    "per_query": per_query,
                }
        finally:
            if backend is not None:
                release_backend(backend)

        report["models"][alias] = model_report

    print_summary(report)
    out_path = out_dir / report_file
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport saved -> {out_path}")


if __name__ == "__main__":
    main()
