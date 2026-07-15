#!/usr/bin/env python3
"""Translate dialogue JSON content (Indonesian) to MSA with hard QA.

Flow:
  1. Load original dialogue JSON
  2. Extract unique content strings (+ speaker role for Kimi context)
  3. Skip whitelist tokens -> content_msa = content
  4. Translate via Kimi (content field only), hard_qa, up to 3 retries
  5. Merge accepted/skipped translations back as content_msa
  6. Record QA failures without writing content_msa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idn_msa.config_loader import load_config
from idn_msa.dialogue_mask import (
    load_skip_whitelist,
    mask_sensitive_tokens,
    should_skip_content,
)
from idn_msa.dialogue_pipeline import process_dialogue_items_with_retry
from idn_msa.expand import TranslationItem
from idn_msa.hard_qa import apply_postprocess, hard_qa_item
from idn_msa.kimi_client import KimiClient
from idn_msa.mask import annotate_item
from idn_msa.translation_cache import TranslationCache

DEFAULT_INPUT = Path(
    "/data1/hcc/jiansuo/dh/dialogue_20260615_BCA_clean_test(1).json"
)
DEFAULT_OUTPUT = Path(
    "/data1/hcc/jiansuo/dh/dialogue_20260615_BCA_clean_test_msa.json"
)
DEFAULT_EXTRACT = Path("/data1/hcc/jiansuo/dh/dialogue_test_content_extract.jsonl")
DEFAULT_CACHE = Path("/data1/hcc/jiansuo/dh/dialogue_test_translation_cache.jsonl")
DEFAULT_DEBUG = Path("/data1/hcc/jiansuo/dh/dialogue_test_translation_debug.jsonl")
DEFAULT_FAILED = Path("/data1/hcc/jiansuo/dh/dialogue_test_translation_failed.jsonl")
DEFAULT_BASE_URL = "http://10.16.137.2:8000/v1"
DEFAULT_MODEL = "Kimi-K2.6-CT-FP8KV"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate dialogue content to MSA.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--extract", type=Path, default=DEFAULT_EXTRACT)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--debug", type=Path, default=DEFAULT_DEBUG)
    p.add_argument("--failed", type=Path, default=DEFAULT_FAILED)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--wave-size", type=int, default=500)
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Kimi thinking via chat_template_kwargs.enable_thinking",
    )
    p.add_argument("--max-items", type=int, default=0, help="Debug: cap untranslated unique strings")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-translate only unique contents listed in --failed (ignores prior failed debug rows)",
    )
    p.add_argument("--assemble-only", action="store_true", help="Only merge cache/skip into output JSON")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def stable_item_id(content: str) -> str:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"dlg_{digest}"


def iter_turn_refs(dialogues: list[dict]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for dlg in dialogues:
        dialogue_id = dlg["dialogue_id"]
        for turn in dlg.get("turns", []):
            refs.append(
                {
                    "dialogue_id": dialogue_id,
                    "turn": turn.get("turn"),
                    "role": turn.get("role", ""),
                    "row_id": turn.get("row_id"),
                    "content": turn.get("content", ""),
                }
            )
    return refs


def write_extract(refs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in refs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_unique_contents(refs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map content -> first ref metadata (role etc.)."""
    unique: dict[str, dict[str, Any]] = {}
    for ref in refs:
        content = (ref.get("content") or "").strip()
        if content not in unique:
            unique[content] = {
                "speaker_role": ref.get("role") or "unknown",
                "sample_dialogue_id": ref["dialogue_id"],
                "sample_turn": ref.get("turn"),
            }
    return unique


def make_dialogue_item(
    content: str,
    meta: dict[str, Any],
    wave_idx: int,
    cfg,
    skip_cfg: dict[str, Any],
) -> TranslationItem:
    item = TranslationItem(
        item_id=stable_item_id(content),
        group_id=f"dlg_wave_{wave_idx:04d}",
        role="dialogue",
        candidate_index=0,
        source_idn=content,
    )
    item.qa["speaker_role"] = meta.get("speaker_role", "unknown")

    skip, reason = should_skip_content(content, skip_cfg)
    if skip:
        item.qa["skip_reason"] = reason
        item.msa_raw = content
        item.final_status = "skipped"
        item.qa["hard_pass"] = True
        return item

    sens_masked, sens_pairs = mask_sensitive_tokens(content)
    masked, entities, terms, actions = annotate_item(sens_masked, cfg)
    item.masked_idn = masked
    item.entities_found = entities
    item.terms_found = terms
    item.actions_found = actions
    item.qa["sensitive_pairs"] = sens_pairs
    return item


def load_failed_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_failed_sources(path: Path) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    if not path.exists():
        return sources
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            content = (row.get("content") or "").strip()
            if content and content not in seen:
                seen.add(content)
                sources.append(content)
    return sources


def failed_meta_by_content(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        content = (row.get("content") or "").strip()
        if not content or content in meta:
            continue
        meta[content] = {
            "speaker_role": row.get("role") or "unknown",
            "sample_dialogue_id": row.get("dialogue_id"),
            "sample_turn": row.get("turn"),
        }
    return meta


def preflight_accept_failed_retries(
    failed_rows: list[dict[str, Any]],
    cfg,
    skip_cfg: dict[str, Any],
    cache: TranslationCache,
    cache_path: Path,
    debug_path: Path,
) -> list[str]:
    """Re-QA prior failed translations; accept/cache those that now pass."""
    from idn_msa.dialogue_mask import restore_sensitive_tokens
    from idn_msa.placeholder_restore import restore_kimi_placeholders

    still_pending: list[str] = []
    debug_rows: list[dict[str, Any]] = []
    accepted = 0
    skipped = 0

    for idx, row in enumerate(failed_rows, start=1):
        content = (row.get("content") or "").strip()
        if not content:
            continue

        skip, reason = should_skip_content(content, skip_cfg)
        if skip:
            skipped += 1
            debug_rows.append(
                {
                    "item_id": stable_item_id(content),
                    "source_idn": content,
                    "speaker_role": row.get("role") or "unknown",
                    "msa_raw": content,
                    "final_status": "skipped",
                    "qa": {"skip_reason": reason, "hard_pass": True},
                }
            )
            continue

        item = make_dialogue_item(
            content,
            failed_meta_by_content([row])[content],
            0,
            cfg,
            skip_cfg,
        )
        item.msa_raw = row.get("msa_raw") or ""
        if item.msa_raw:
            sens_pairs = item.qa.get("sensitive_pairs") or []
            text = restore_sensitive_tokens(item.msa_raw, sens_pairs)
            text, _ = restore_kimi_placeholders(item.source_idn, text, cfg)
            item.msa_raw = text
            apply_postprocess(item, cfg)
            item.qa.update(hard_qa_item(item, cfg))
            if item.qa.get("hard_pass"):
                item.final_status = "accepted"
                cache.set(content, item.msa_raw)
                accepted += 1
                debug_rows.append(
                    {
                        "item_id": item.item_id,
                        "source_idn": content,
                        "speaker_role": item.qa.get("speaker_role"),
                        "msa_raw": item.msa_raw,
                        "final_status": "accepted",
                        "qa": item.qa,
                    }
                )
                continue

        still_pending.append(content)

        if idx % 20 == 0 or idx == len(failed_rows):
            logging.info("Retry preflight progress: %s/%s", idx, len(failed_rows))

    if debug_rows:
        append_jsonl(debug_path, debug_rows)
        cache.save(cache_path)
    logging.info(
        "Retry preflight: accepted_existing=%s skipped=%s still_need_api=%s",
        accepted,
        skipped,
        len(still_pending),
    )
    return still_pending


def load_processed_sources(debug_path: Path) -> set[str]:
    done: set[str] = set()
    if not debug_path.exists():
        return done
    with debug_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            src = row.get("source_idn")
            status = row.get("final_status")
            if src and status in {"accepted", "skipped", "failed"}:
                done.add(src)
    return done


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _persist_wave_items(
    items: list[TranslationItem],
    cache: TranslationCache,
    cache_path: Path,
    debug_path: Path,
) -> None:
    """Save accepted/failed progress even if the wave crashes mid-run."""
    for item in items:
        if item.final_status == "accepted" and item.msa_raw:
            cache.set(item.source_idn, item.msa_raw)
        elif item.final_status == "pending":
            item.final_status = "failed"
            item.qa["needs_manual_review"] = True

    debug_rows = [
        {
            "item_id": item.item_id,
            "source_idn": item.source_idn,
            "speaker_role": item.qa.get("speaker_role"),
            "msa_raw": item.msa_raw,
            "final_status": item.final_status,
            "qa": item.qa,
        }
        for item in items
    ]
    append_jsonl(debug_path, debug_rows)
    cache.save(cache_path)


def build_translation_map(
    refs: list[dict[str, Any]],
    cache: TranslationCache,
    skip_cfg: dict[str, Any],
    debug_path: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    translations: dict[str, str] = {}
    failures: list[dict[str, Any]] = []

    latest_debug: dict[str, dict[str, Any]] = {}
    if debug_path.exists():
        with debug_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    src = row.get("source_idn")
                    if src:
                        latest_debug[src] = row

    for ref in refs:
        content = (ref.get("content") or "").strip()
        if not content:
            continue

        skip, reason = should_skip_content(content, skip_cfg)
        if skip:
            translations[content] = content
            continue

        msa = cache.get(content)
        if msa:
            translations[content] = msa
            continue

        dbg = latest_debug.get(content)
        if dbg and dbg.get("final_status") == "accepted" and dbg.get("msa_raw"):
            translations[content] = dbg["msa_raw"]
            continue

        if dbg and dbg.get("final_status") == "failed":
            failures.append(
                {
                    "dialogue_id": ref["dialogue_id"],
                    "turn": ref.get("turn"),
                    "role": ref.get("role"),
                    "row_id": ref.get("row_id"),
                    "content": content,
                    "qa": dbg.get("qa", {}),
                    "msa_raw": dbg.get("msa_raw"),
                }
            )

    # dedupe failures by content
    seen: set[str] = set()
    uniq_failures: list[dict[str, Any]] = []
    for row in failures:
        key = row["content"]
        if key in seen:
            continue
        seen.add(key)
        uniq_failures.append(row)
    return translations, uniq_failures


def merge_into_dialogues(
    dialogues: list[dict],
    translations: dict[str, str],
) -> list[dict]:
    out: list[dict] = []
    for dlg in dialogues:
        new_dlg = dict(dlg)
        turns = []
        for turn in dlg.get("turns", []):
            new_turn = dict(turn)
            content = (turn.get("content") or "").strip()
            if content in translations:
                new_turn["content_msa"] = translations[content]
            turns.append(new_turn)
        new_dlg["turns"] = turns
        out.append(new_dlg)
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = load_config()
    skip_cfg = load_skip_whitelist(cfg.config_dir)

    dialogues = json.loads(args.input.read_text(encoding="utf-8"))
    refs = iter_turn_refs(dialogues)
    write_extract(refs, args.extract)
    logging.info("Wrote extract rows=%s -> %s", len(refs), args.extract)

    unique = collect_unique_contents(refs)
    cache = TranslationCache(args.cache)

    if args.retry_failed:
        failed_rows = load_failed_rows(args.failed)
        failed_meta = failed_meta_by_content(failed_rows)
        unique.update(failed_meta)
        pending = preflight_accept_failed_retries(
            failed_rows,
            cfg,
            skip_cfg,
            cache,
            args.cache,
            args.debug,
        )
    else:
        skip_count = sum(1 for c in unique if should_skip_content(c, skip_cfg)[0])
        pending = [c for c in unique if not should_skip_content(c, skip_cfg)[0] and not cache.get(c)]
        logging.info(
            "unique_content=%s skip_whitelist=%s cached=%s pending=%s",
            len(unique),
            skip_count,
            len(unique) - skip_count - len(pending),
            len(pending),
        )

    if args.max_items:
        pending = pending[: args.max_items]

    if not args.assemble_only and pending:
        if args.resume and not args.retry_failed:
            done = load_processed_sources(args.debug)
            done.update(src for src in unique if cache.get(src))
            pending = [c for c in pending if c not in done]
            logging.info("Resume: remaining pending=%s", len(pending))

        client = KimiClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            enable_thinking=args.enable_thinking,
        )

        for wave_idx, start in enumerate(range(0, len(pending), args.wave_size), start=1):
            wave_src = pending[start : start + args.wave_size]
            items = [
                make_dialogue_item(
                    src,
                    unique.get(src, {"speaker_role": "unknown"}),
                    wave_idx,
                    cfg,
                    skip_cfg,
                )
                for src in wave_src
            ]
            if not items:
                continue

            logging.info(
                "Wave %s: translating %s unique strings (%s/%s)",
                wave_idx,
                len(items),
                start + len(wave_src),
                len(pending),
            )
            try:
                process_dialogue_items_with_retry(
                    client=client,
                    items=items,
                    cfg=cfg,
                    batch_size=args.batch_size,
                    concurrency=args.concurrency,
                    cache=cache,
                )
            finally:
                _persist_wave_items(items, cache, args.cache, args.debug)

            accepted = sum(1 for i in items if i.final_status == "accepted")
            failed = sum(1 for i in items if i.final_status == "failed")
            logging.info(
                "Wave %s done: accepted=%s failed=%s cache=%s",
                wave_idx,
                accepted,
                failed,
                len(cache),
            )

    translations, failures = build_translation_map(refs, cache, skip_cfg, args.debug)
    merged = merge_into_dialogues(dialogues, translations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if failures:
        args.failed.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
            encoding="utf-8",
        )
    elif args.failed.exists():
        args.failed.unlink()

    filled = sum(1 for r in refs if (r.get("content") or "").strip() in translations)
    logging.info(
        "Wrote output -> %s | turns_with_content_msa=%s/%s | failed_unique=%s",
        args.output,
        filled,
        len(refs),
        len(failures),
    )

    if failures and not args.assemble_only:
        logging.warning("Some strings failed QA; see %s", args.failed)


if __name__ == "__main__":
    main()
