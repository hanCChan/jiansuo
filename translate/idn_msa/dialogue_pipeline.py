from __future__ import annotations

import logging
from collections import Counter

from .config_loader import PipelineConfig
from .dialogue_mask import restore_sensitive_tokens
from .dialogue_translate import repair_dialogue_batch, translate_dialogue_batch
from .expand import TranslationItem
from .hard_qa import apply_postprocess, hard_qa_item
from .kimi_client import KimiClient
from .placeholder_restore import restore_kimi_placeholders
from .translation_cache import TranslationCache

MAX_RETRY_ROUNDS = 3
logger = logging.getLogger(__name__)


def _finalize_item_msa(item: TranslationItem, cfg: PipelineConfig) -> None:
    if item.final_status == "skipped":
        return
    sens_pairs = item.qa.get("sensitive_pairs") or []
    text = restore_sensitive_tokens(item.msa_raw, sens_pairs)
    text, _ = restore_kimi_placeholders(item.source_idn, text, cfg)
    item.msa_raw = text


def _merge_qa(item: TranslationItem, cfg: PipelineConfig) -> None:
    apply_postprocess(item, cfg)
    item.qa.update(hard_qa_item(item, cfg))


def _apply_cache(items: list[TranslationItem], cache: TranslationCache | None) -> int:
    if not cache:
        return 0
    hits = 0
    for item in items:
        cached = cache.get(item.source_idn)
        if cached:
            item.msa_raw = cached
            hits += 1
    return hits


def _log_qa_round(round_idx: int, items: list[TranslationItem], failed: list[TranslationItem]) -> None:
    err_ctr: Counter[str] = Counter()
    for item in failed:
        for err in item.qa.get("hard_errors", []):
            err_ctr[err.split(":")[0]] += 1
    top = ", ".join(f"{k}={v}" for k, v in err_ctr.most_common(5))
    logger.info(
        "Dialogue QA round %s: accepted=%s failed=%s total=%s top_errors=[%s]",
        round_idx + 1,
        len(items) - len(failed),
        len(failed),
        len(items),
        top or "none",
    )


def process_dialogue_items_with_retry(
    client: KimiClient,
    items: list[TranslationItem],
    cfg: PipelineConfig,
    batch_size: int,
    concurrency: int,
    cache: TranslationCache | None = None,
) -> None:
    """Hard QA only, up to 3 rounds: translate once, then repair-only retries."""
    pending = [item for item in items if item.final_status != "skipped"]
    cache_hits = _apply_cache(pending, cache)
    if cache_hits:
        logger.info("Dialogue cache hits: %s/%s", cache_hits, len(pending))

    for round_idx in range(MAX_RETRY_ROUNDS):
        if not pending:
            break

        if round_idx == 0:
            need_translate = [item for item in pending if not item.msa_raw]
            if need_translate:
                translate_dialogue_batch(
                    client,
                    need_translate,
                    cfg,
                    batch_size=batch_size,
                    concurrency=concurrency,
                )
        else:
            error_by_id = {
                item.item_id: "; ".join(item.qa.get("hard_errors", [])) or "hard QA failed"
                for item in pending
            }
            repair_dialogue_batch(client, pending, cfg, error_by_id, concurrency=concurrency)

        failed: list[TranslationItem] = []
        for item in pending:
            _finalize_item_msa(item, cfg)
            _merge_qa(item, cfg)
            item.retry_rounds = round_idx
            if item.qa.get("hard_pass", False):
                item.final_status = "accepted"
                if cache:
                    cache.set(item.source_idn, item.msa_raw)
            else:
                item.final_status = "pending"
                failed.append(item)

        _log_qa_round(round_idx, pending, failed)
        if not failed:
            break
        pending = failed

    for item in items:
        if item.final_status in {"accepted", "skipped"}:
            continue
        if item.final_status == "pending":
            item.final_status = "failed"
            item.qa["needs_manual_review"] = True
