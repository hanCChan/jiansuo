"""Compare BGE-M3 mode top10 rankings from cached score arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _resolve_mode_dir(cache_root: Path, mode: str) -> Path:
    """Match cache dir exactly; avoid `dense` prefix swallowing `dense+sparse`."""
    token = mode.replace("+", "_plus_")
    exact = sorted(cache_root.glob(f"bge_m3__{mode}__*"))
    if exact:
        return exact[0]
    escaped = sorted(cache_root.glob(f"bge_m3__{token}__*"))
    if escaped:
        return escaped[0]
    # Legacy fallback: pick shortest matching dirname after alias prefix.
    candidates = [
        p
        for p in cache_root.glob("bge_m3__*")
        if p.name.startswith(f"bge_m3__{mode}__")
    ]
    if not candidates:
        raise FileNotFoundError(f"no cache dir for mode {mode}")
    return sorted(candidates, key=lambda p: len(p.name))[0]


def load_mode_scores(cache_root: Path, mode: str, n_queries: int) -> list[np.ndarray]:
    mode_dir = _resolve_mode_dir(cache_root, mode)
    scores = []
    for i in range(n_queries):
        path = mode_dir / f"{i:04d}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        scores.append(np.load(path))
    return scores


def topk_indices(scores: np.ndarray, k: int = 10) -> list[int]:
    return np.argsort(-scores)[:k].tolist()


def compare_modes(
    cache_root: Path,
    n_queries: int,
    modes: list[str],
) -> dict:
    tops = {m: [topk_indices(s) for s in load_mode_scores(cache_root, m, n_queries)] for m in modes}
    result: dict = {"modes": modes, "pairwise": {}}
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            same10 = sum(1 for qa, qb in zip(tops[a], tops[b]) if qa == qb)
            same1 = sum(1 for qa, qb in zip(tops[a], tops[b]) if qa[0] == qb[0])
            diffs = []
            for qi, (qa, qb) in enumerate(zip(tops[a], tops[b])):
                if qa != qb:
                    diffs.append({"cluster_id": qi, f"{a}_top10": qa, f"{b}_top10": qb})
            result["pairwise"][f"{a}_vs_{b}"] = {
                "same_top10_count": same10,
                "same_top1_count": same1,
                "same_top10_ratio": round(same10 / n_queries, 4),
                "same_top1_ratio": round(same1 / n_queries, 4),
                "sample_diffs": diffs[:8],
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BGE-M3 cached mode rankings")
    parser.add_argument(
        "--cache-root",
        default="output/reports/score_cache_multimode",
    )
    parser.add_argument("--n-queries", type=int, default=37)
    parser.add_argument("-o", "--output", default="output/reports/bge_mode_top10_check.json")
    args = parser.parse_args()

    modes = ["dense", "colbert", "dense+sparse", "hybrid"]
    # cache tags use mode names; dense+sparse has plus sign
    result = compare_modes(Path(args.cache_root), args.n_queries, modes)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"BGE mode check -> {out}")
    for key, val in result["pairwise"].items():
        print(
            f"  {key}: same_top10={val['same_top10_count']}/{args.n_queries} "
            f"same_top1={val['same_top1_count']}/{args.n_queries}"
        )


if __name__ == "__main__":
    main()
