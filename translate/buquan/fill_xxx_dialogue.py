#!/usr/bin/env python3
"""Fill XXX in dialogue MSA by local clusters using Kimi 2.6."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BUQUAN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUQUAN))

from idn_msa.kimi_client import KimiClient
from xxx_fill_qa import qa_cluster_result, qa_turn_fill
from xxx_utils import (
    XxxCluster,
    build_all_clusters,
    build_local_context,
    find_xxx_tokens,
    has_xxx,
)

DEFAULT_INPUT = Path("/data1/hcc/jiansuo/dh/dialogue_20260615_BCA_clean_test_msa.json")
DEFAULT_OUTPUT = Path(
    "/data1/hcc/jiansuo/dh/dialogue_20260615_BCA_clean_test_msa_xxx_filled.json"
)
DEFAULT_DEBUG = BUQUAN / "dialogue_test_xxx_fill_debug.jsonl"
DEFAULT_CACHE = BUQUAN / "dialogue_test_xxx_fill_cache.jsonl"
DEFAULT_REPORT = BUQUAN / "dialogue_test_xxx_fill_report.json"
DEFAULT_PROMPT = BUQUAN / "prompts" / "xxx_fill_system.txt"
DEFAULT_BASE_URL = "http://10.16.137.2:8000/v1"
DEFAULT_MODEL = "Kimi-K2.6-CT-FP8KV"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill XXX in dialogue MSA by local cluster.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--debug", type=Path, default=DEFAULT_DEBUG)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--gap-max", type=int, default=3)
    p.add_argument("--context-window", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--assemble-only", action="store_true")
    p.add_argument("--dialogue-id", type=int, default=0, help="Only run one dialogue_id")
    p.add_argument("--max-clusters", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("cluster_id")
            if cid:
                cache[cid] = row
    return cache


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cid in sorted(cache):
            f.write(json.dumps(cache[cid], ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_user_payload(
    cluster: XxxCluster,
    dialogue_turns: list,
    context_window: int,
) -> dict[str, Any]:
    payload = cluster.to_kimi_payload(context_window=context_window)
    payload["local_context_msa"] = build_local_context(
        dialogue_turns, cluster, window=context_window
    )
    return payload


def normalize_cluster_response(
    cluster: XxxCluster,
    raw: dict[str, Any],
) -> dict[str, Any]:
    turns_out: list[dict[str, Any]] = []
    raw_turns = raw.get("turns")
    if not isinstance(raw_turns, list):
        raise KeyError(f"missing turns in response: {raw!r}")

    by_turn = {int(t.turn): t for t in cluster.turns}
    for item in raw_turns:
        if not isinstance(item, dict):
            continue
        turn_num = int(item.get("turn"))
        filled = (item.get("content_msa_filled") or "").strip()
        if not filled:
            continue
        src = by_turn.get(turn_num)
        turns_out.append(
            {
                "turn": turn_num,
                "role": src.role if src else "",
                "content_msa": src.content_msa if src else "",
                "content_msa_filled": filled,
                "confidence": item.get("confidence", "medium"),
                "notes": item.get("notes", ""),
            }
        )

    if len(turns_out) != len(cluster.turns):
        missing = sorted(set(by_turn) - {t["turn"] for t in turns_out})
        if missing:
            raise KeyError(f"missing filled turns: {missing}")

    return {
        "cluster_id": cluster.cluster_id,
        "dialogue_id": cluster.dialogue_id,
        "slot_map": raw.get("slot_map", {}),
        "turns": turns_out,
        "turns_to_fill": [t.to_dict() for t in cluster.turns],
    }


def fill_cluster(
    client: KimiClient,
    system_prompt: str,
    cluster: XxxCluster,
    dialogue_turns: list,
    context_window: int,
) -> dict[str, Any]:
    payload = build_user_payload(cluster, dialogue_turns, context_window)
    raw = client.chat_json(system_prompt, json.dumps(payload, ensure_ascii=False), temperature=0.0)
    normalized = normalize_cluster_response(cluster, raw)
    qa = qa_cluster_result(normalized)
    normalized["cluster_status"] = qa["cluster_status"]
    normalized["turn_qa"] = qa["turn_results"]
    return normalized


def merge_results(
    dialogues: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # turn -> fill meta
    fill_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    status_ctr = {"accepted": 0, "warning": 0, "failed": 0, "skipped": 0}
    pii_like = 0
    remaining_xxx = 0

    for row in cache.values():
        cid = row.get("cluster_id")
        cluster_status = row.get("cluster_status", "failed")
        for turn_row in row.get("turns", []):
            key = (int(row["dialogue_id"]), int(turn_row["turn"]))
            turn_qa = (row.get("turn_qa") or {}).get(str(turn_row["turn"]), {})
            status = turn_qa.get("status", cluster_status)
            fill_by_key[key] = {
                "content_msa_filled": turn_row.get("content_msa_filled", ""),
                "xxx_fill_status": status,
                "xxx_fill_meta": {
                    "cluster_id": cid,
                    "cluster_status": cluster_status,
                    "confidence": turn_row.get("confidence"),
                    "notes": turn_row.get("notes", ""),
                    "slot_map": row.get("slot_map", {}),
                    "qa": turn_qa,
                },
            }
            status_ctr[status] = status_ctr.get(status, 0) + 1
            if turn_qa.get("errors"):
                for err in turn_qa["errors"]:
                    if err.startswith("pii_shape"):
                        pii_like += 1
            if find_xxx_tokens(turn_row.get("content_msa_filled", "")):
                remaining_xxx += 1

    out_dialogues: list[dict[str, Any]] = []
    turns_with_xxx = 0
    for dlg in dialogues:
        did = int(dlg["dialogue_id"])
        new_turns = []
        for turn in dlg.get("turns", []):
            new_turn = dict(turn)
            tnum = int(turn.get("turn", 0))
            msa = (turn.get("content_msa") or "").strip()
            if has_xxx(msa):
                turns_with_xxx += 1
                meta = fill_by_key.get((did, tnum))
                if meta and meta["xxx_fill_status"] in {"accepted", "warning"}:
                    new_turn["content_msa_filled"] = meta["content_msa_filled"]
                    new_turn["xxx_fill_status"] = meta["xxx_fill_status"]
                    new_turn["xxx_fill_meta"] = meta["xxx_fill_meta"]
                else:
                    new_turn["xxx_fill_status"] = meta["xxx_fill_status"] if meta else "failed"
                    new_turn["xxx_fill_meta"] = meta["xxx_fill_meta"] if meta else {"reason": "not_filled"}
            new_turns.append(new_turn)
        new_dlg = dict(dlg)
        new_dlg["turns"] = new_turns
        out_dialogues.append(new_dlg)

    report = {
        "turns_with_xxx": turns_with_xxx,
        "filled_accepted_or_warning": status_ctr.get("accepted", 0) + status_ctr.get("warning", 0),
        "status_counts": status_ctr,
        "remaining_xxx": remaining_xxx,
        "pii_like_generated": pii_like,
        "clusters_cached": len(cache),
    }
    return out_dialogues, report


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dialogues = json.loads(args.input.read_text(encoding="utf-8"))
    by_dialogue, clusters = build_all_clusters(dialogues, gap_max=args.gap_max)

    if args.dialogue_id:
        clusters = [c for c in clusters if c.dialogue_id == args.dialogue_id]
    if args.max_clusters:
        clusters = clusters[: args.max_clusters]

    cache = load_cache(args.cache) if args.resume else {}
    pending = [c for c in clusters if c.cluster_id not in cache]
    logging.info(
        "clusters_total=%s cached=%s pending=%s",
        len(clusters),
        len(cache),
        len(pending),
    )

    system_prompt = load_system_prompt(args.system_prompt)

    if not args.assemble_only and pending:
        client = KimiClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            enable_thinking=args.enable_thinking,
        )

        def _work(cluster: XxxCluster) -> dict[str, Any]:
            return fill_cluster(
                client,
                system_prompt,
                cluster,
                by_dialogue[cluster.dialogue_id],
                args.context_window,
            )

        debug_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
            futures = {pool.submit(_work, cluster): cluster for cluster in pending}
            for idx, fut in enumerate(as_completed(futures), start=1):
                cluster = futures[fut]
                try:
                    row = fut.result()
                    cache[cluster.cluster_id] = row
                    debug_rows.append(row)
                    logging.info(
                        "Done %s/%s %s status=%s turns=%s",
                        idx,
                        len(pending),
                        cluster.cluster_id,
                        row.get("cluster_status"),
                        len(row.get("turns", [])),
                    )
                except Exception as exc:  # noqa: BLE001
                    logging.error("Failed %s: %s", cluster.cluster_id, exc)
                    debug_rows.append(
                        {
                            "cluster_id": cluster.cluster_id,
                            "dialogue_id": cluster.dialogue_id,
                            "cluster_status": "failed",
                            "error": str(exc),
                            "turns_to_fill": [t.to_dict() for t in cluster.turns],
                        }
                    )
                if idx % 10 == 0:
                    save_cache(args.cache, cache)
                    append_jsonl(args.debug, debug_rows)
                    debug_rows = []

        if debug_rows:
            append_jsonl(args.debug, debug_rows)
        save_cache(args.cache, cache)

    out_dialogues, report = merge_results(dialogues, cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out_dialogues, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["input"] = str(args.input)
    report["output"] = str(args.output)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Wrote %s", args.output)
    logging.info("Report: %s", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
