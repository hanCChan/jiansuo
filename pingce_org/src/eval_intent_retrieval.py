"""Evaluate embedding models on intent eval JSON with per-model retrieval modes."""

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


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path, root: Path = ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def resolve_eval_path(args: argparse.Namespace, cfg: dict) -> Path:
    if args.input:
        return resolve_path(args.input)
    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("input_json"):
        return resolve_path(eval_cfg["input_json"])
    paths = cfg.get("paths", {})
    if paths.get("cluster_retrieval_eval_json"):
        return resolve_path(paths["cluster_retrieval_eval_json"])
    return resolve_path(paths["output_json"])


def load_eval(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for i, item in enumerate(data):
        if not item["positive"] or not item["negative"]:
            raise ValueError(f"cluster {i}: empty pos/neg")
    return data


def metrics_from_scores(scores: np.ndarray, labels: list[int]) -> dict:
    """Return flat retrieval metrics for one query."""
    order = np.argsort(-scores)

    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    pos_ranks = [int(np.where(order == pi)[0][0]) + 1 for pi in pos_idx]
    pos_rank_arr = np.asarray(pos_ranks, dtype=np.float32)

    return {
        "recall@1": float(np.mean(pos_rank_arr <= 1)) if pos_ranks else 0.0,
        "recall@3": float(np.mean(pos_rank_arr <= 3)) if pos_ranks else 0.0,
        "recall@5": float(np.mean(pos_rank_arr <= 5)) if pos_ranks else 0.0,
    }


METRIC_KEYS = ("recall@1", "recall@3", "recall@5")


def aggregate_metrics(rows: list[dict]) -> dict:
    return {k: float(np.mean([float(r[k]) for r in rows])) for k in METRIC_KEYS}


def eval_backend_on_dataset(
    data: list[dict],
    backend,
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
            scores = backend.score(item["query"], candidates)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, scores)

        row = metrics_from_scores(scores, labels)
        row["cluster_id"] = i
        row["query"] = item["query"]
        rows.append(row)
    return rows


def release_backend(backend) -> None:
    import torch

    backend.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_model_cfg(raw: dict | str) -> dict:
    if isinstance(raw, str):
        return {"path": raw, "backend": "dense", "modes": ["dense"]}
    cfg = dict(raw)
    cfg.setdefault("backend", "dense")
    cfg.setdefault("modes", ["dense"])
    return cfg


def resolve_model_device(model_cfg: dict, default_device: str) -> str:
    if model_cfg.get("device"):
        return str(model_cfg["device"])
    devices = model_cfg.get("devices")
    if devices is None:
        return default_device
    if isinstance(devices, str):
        first = devices.split(",")[0].strip()
        if first == "all":
            return default_device
        return first if first.startswith("cuda") else f"cuda:{first}"
    if isinstance(devices, list) and devices:
        first = str(devices[0]).strip()
        return first if first.startswith("cuda") else f"cuda:{first}"
    return default_device


def eval_cache_tag(eval_path: Path, alias: str, mode: str, device: str, model_path: str) -> str:
    stat = eval_path.stat()
    digest = hashlib.sha1(
        f"{eval_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|"
        f"{alias}|{mode}|{device}|{model_path}".encode()
    ).hexdigest()[:12]
    return f"{alias}__{mode}__{digest}"


def print_summary(report: dict) -> None:
    print("\n=== Intent Retrieval Benchmark (multi-mode) ===")
    print(f"eval_set: {report['eval_set']}")
    print(f"queries: {report['num_queries']}")
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
    parser = argparse.ArgumentParser(description="Benchmark embeddings on intent eval JSON")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--input",
        default=None,
        help="eval JSON path (relative to pingce_org root); default from config eval.input_json",
    )
    parser.add_argument("--models", nargs="*", default=None, help="model aliases")
    parser.add_argument(
        "--modes",
        nargs="*",
        default=None,
        help="override modes for all selected models, e.g. dense sparse hybrid",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="default torch device; per-model override via config eval.models.<alias>.device",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="encoding batch size; default from config eval.batch_size",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="optional score cache dir (relative to pingce_org root)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable score cache even if eval.cache_dir is set in config",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="debug: only evaluate the first N query groups",
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="override report output filename under report_dir",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    eval_cfg = cfg.get("eval", {})
    emb_cfg = cfg.get("embedding", {})

    eval_path = resolve_eval_path(args, cfg)
    if not eval_path.is_file():
        raise SystemExit(f"eval set not found: {eval_path}")
    data = load_eval(eval_path)
    if args.max_queries is not None:
        data = data[: args.max_queries]

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
        cache_arg = args.cache_dir or eval_cfg.get("cache_dir")
        if cache_arg:
            cache_dir = resolve_path(cache_arg)

    report = {
        "eval_set": str(eval_path),
        "num_queries": len(data),
        "default_device": default_device,
        "batch_size": batch_size,
        "score_cache_dir": str(cache_dir) if cache_dir else None,
        "metrics_doc": {
            "recall@1": "fraction of positive samples whose rank is <= 1",
            "recall@3": "fraction of positive samples whose rank is <= 3",
            "recall@5": "fraction of positive samples whose rank is <= 5",
        },
        "mode_doc": {
            "bge_m3": {
                "dense": "BGE-M3 dense cosine",
                "sparse": "BGE-M3 lexical matching",
                "colbert": "BGE-M3 multi-vector ColBERT score",
                "hybrid": "weighted dense+sparse+colbert (see hybrid_weights)",
                "dense+sparse": "weighted dense+sparse only",
            },
            "dense": {"dense": "SentenceTransformer dense cosine similarity"},
        },
        "models": {},
    }

    out_dir = resolve_path(eval_cfg.get("report_dir", "output/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

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
                print(f"Evaluating {alias} / {mode} on {device} ...")
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
                cache_tag = eval_cache_tag(eval_path, alias, mode, device, model_cfg["path"])
                per_query = eval_backend_on_dataset(
                    data,
                    backend,
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
    report_name = args.report_file or eval_cfg.get("report_file", "intent_retrieval_eval.json")
    out_path = out_dir / report_name
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport saved -> {out_path}")


if __name__ == "__main__":
    main()
