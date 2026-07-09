"""Evaluate retrieval with cluster-averaged candidate scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from embedding_backends import (  # noqa: E402
    create_backend,
    load_corpus_cache,
    save_corpus_cache,
)
from eval_intent_retrieval import (  # noqa: E402
    aggregate_metrics,
    load_eval,
    metrics_from_scores,
    normalize_model_cfg,
    release_backend,
    resolve_eval_path,
    resolve_model_device,
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
    return [question, *list(row["Question_cluster"])]


def build_variant_layout(
    candidates: list[str], qa_index: dict[str, dict]
) -> tuple[list[str], list[int]]:
    variant_groups = [candidate_variants(cand, qa_index) for cand in candidates]
    flat_variants: list[str] = []
    group_sizes: list[int] = []
    for group in variant_groups:
        flat_variants.extend(group)
        group_sizes.append(len(group))
    return flat_variants, group_sizes


def load_or_build_corpus_cache(
    backend,
    flat_variants: list[str],
    *,
    cache_dir: Path | None,
    cache_tag: str,
) -> Any:
    corpus_path = None
    if cache_dir is not None:
        corpus_path = cache_dir / cache_tag / f"corpus__{backend.corpus_cache_name()}.pkl"
        cached = load_corpus_cache(corpus_path)
        if cached is not None:
            print(f"  loaded variant corpus cache ({len(flat_variants)} variants)")
            return cached

    print(f"  encoding shared variant corpus ({len(flat_variants)} variants) ...")
    t0 = time.time()
    corpus = backend.encode_cluster_corpus(flat_variants)
    print(f"  variant corpus ready in {time.time() - t0:.1f}s")
    if corpus_path is not None:
        save_corpus_cache(corpus_path, corpus)
    return corpus


def build_canonical_candidates(data: list[dict]) -> list[str]:
    """Stable candidate order shared across queries (same set, per-query order differs)."""
    return sorted({c for item in data for c in item["positive"] + item["negative"]})


def score_candidates_cluster_avg(
    backend,
    query: str,
    candidates: list[str],
    qa_index: dict[str, dict],
    *,
    corpus=None,
    group_sizes: list[int] | None = None,
    cand_to_idx: dict[str, int] | None = None,
) -> np.ndarray:
    if corpus is not None and group_sizes is not None and cand_to_idx is not None:
        canon_scores = backend.score_cluster_avg_corpus(query, group_sizes, corpus)
        return np.asarray(
            [canon_scores[cand_to_idx[cand]] for cand in candidates], dtype=np.float32
        )
    variant_groups = [candidate_variants(cand, qa_index) for cand in candidates]
    return backend.score_cluster_avg(query, variant_groups)


def eval_backend_on_dataset(
    data: list[dict],
    backend,
    qa_index: dict[str, dict],
    *,
    cache_dir: Path | None = None,
    cache_tag: str = "",
) -> list[dict]:
    rows: list[dict] = []
    canonical_candidates = build_canonical_candidates(data)
    cand_to_idx = {cand: i for i, cand in enumerate(canonical_candidates)}
    flat_variants, group_sizes = build_variant_layout(canonical_candidates, qa_index)
    corpus = load_or_build_corpus_cache(
        backend,
        flat_variants,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
    )

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
            scores = score_candidates_cluster_avg(
                backend,
                item["query"],
                candidates,
                qa_index,
                corpus=corpus,
                group_sizes=group_sizes,
                cand_to_idx=cand_to_idx,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, scores)

        row = metrics_from_scores(scores, labels)
        for key in ("pos_rank", "hit@1", "hit@3", "hit@5", "topk_indices", "topk_scores"):
            row.pop(key, None)
        row["cluster_id"] = i
        row["query"] = item["query"]
        rows.append(row)
    return rows


def cluster_cache_tag(
    eval_path: Path,
    qa_path: Path,
    alias: str,
    mode: str,
    device: str,
    model_path: str,
    model_cfg: dict,
) -> str:
    eval_stat = eval_path.stat()
    qa_stat = qa_path.stat()
    cfg_bits = ""
    cfg_keys = (
        "query_style",
        "task_description",
        "hybrid_weights",
        "hybrid_dense_weight",
        "hybrid_sparse_weight",
        "gemma_task",
    )
    cfg_bits = "|".join(f"{k}={model_cfg.get(k)}" for k in cfg_keys if model_cfg.get(k) is not None)
    digest = hashlib.sha1(
        f"cluster_avg|canon_v2|{eval_path.resolve()}|{eval_stat.st_mtime_ns}|{eval_stat.st_size}|"
        f"{qa_path.resolve()}|{qa_stat.st_mtime_ns}|{qa_stat.st_size}|"
        f"{alias}|{mode}|{device}|{model_path}|{cfg_bits}".encode()
    ).hexdigest()[:12]
    return f"{alias}__{mode}__{digest}"


def resolve_qa_path(args: argparse.Namespace, cfg: dict) -> Path:
    eval_cfg = cfg.get("eval", {})
    cluster_cfg = eval_cfg.get("cluster", {})
    if args.qa_json:
        return resolve_path(args.qa_json)
    if cluster_cfg.get("qa_json"):
        return resolve_path(cluster_cfg["qa_json"])
    return resolve_path("./qa.json")


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
    parser.add_argument("--config", default="config_cluster_msa.yaml")
    parser.add_argument("--input", default=None, help="eval JSON path")
    parser.add_argument("--qa-json", default=None, help="FAQ JSON with Question_cluster")
    parser.add_argument("--models", nargs="*", default=None, help="model aliases")
    parser.add_argument("--modes", nargs="*", default=None, help="override modes")
    parser.add_argument("--device", default=None, help="default torch device")
    parser.add_argument("--batch-size", type=int, default=None, help="encoding batch size")
    parser.add_argument("--cache-dir", default=None, help="score cache dir")
    parser.add_argument("--no-cache", action="store_true", help="disable score cache")
    parser.add_argument("--max-queries", type=int, default=None, help="debug: first N queries")
    parser.add_argument("--report-file", default=None, help="override cluster report filename")
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
    if args.max_queries is not None:
        data = data[: args.max_queries]
    qa_index = load_qa_index(qa_path)

    raw_models: dict = eval_cfg.get("models", {})
    if not raw_models:
        raise SystemExit("config.eval.models is empty")

    selected = args.models or list(raw_models.keys())
    unknown = [m for m in selected if m not in raw_models]
    if unknown:
        raise SystemExit(f"unknown model aliases: {unknown}; available={list(raw_models)}")

    default_device = args.device or eval_cfg.get("device", emb_cfg.get("device", "cuda"))
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
        "default_device": default_device,
        "batch_size": batch_size,
        "score_cache_dir": str(cache_dir) if cache_dir else None,
        "scoring_doc": (
            "For each candidate FAQ, score = mean similarity between query and "
            "[Question, ...Question_cluster (5 variants from qa.json)]. "
            "Candidates are ranked by this average score."
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
    report_file = args.report_file or cluster_cfg.get(
        "report_file", "intent_retrieval_cluster_eval_msa.json"
    )

    for alias in selected:
        model_cfg = normalize_model_cfg(raw_models[alias])
        modes = args.modes or model_cfg["modes"]
        device = resolve_model_device(model_cfg, default_device)

        model_report = {
            "path": model_cfg["path"],
            "backend": model_cfg["backend"],
            "device": device,
            "configured_modes": model_cfg["modes"],
            "modes": {},
        }

        backend = None
        try:
            for mode in modes:
                print(f"Evaluating {alias} / {mode} (cluster-avg) on {device} ...")
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
                cache_tag = cluster_cache_tag(
                    eval_path,
                    qa_path,
                    alias,
                    mode,
                    device,
                    model_cfg["path"],
                    model_cfg,
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
