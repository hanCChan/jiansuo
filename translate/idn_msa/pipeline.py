from __future__ import annotations

import logging
from collections import Counter

from .expand import TranslationItem
from .hard_qa import apply_postprocess, hard_qa_item
from .kimi_client import KimiClient
from .semantic_qa import backtranslate, judge_meaning, judge_relation
from .translate import repair_batch, translate_batch
from .config_loader import PipelineConfig
from .translation_cache import TranslationCache

MAX_RETRY_ROUNDS = 3
logger = logging.getLogger(__name__)


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


def _update_cache(items: list[TranslationItem], cache: TranslationCache | None) -> None:
    if not cache:
        return
    for item in items:
        if item.final_status == "accepted" and item.msa_raw:
            cache.set(item.source_idn, item.msa_raw)


def _log_qa_round(round_idx: int, items: list[TranslationItem], failed: list[TranslationItem]) -> None:
    err_ctr: Counter[str] = Counter()
    for item in failed:
        for err in item.qa.get("hard_errors", []):
            err_ctr[err.split(":")[0]] += 1
    top = ", ".join(f"{k}={v}" for k, v in err_ctr.most_common(5))
    logger.info(
        "QA round %s: accepted=%s failed=%s total=%s top_errors=[%s]",
        round_idx + 1,
        len(items) - len(failed),
        len(failed),
        len(items),
        top or "none",
    )


def process_items_with_retry(
    client: KimiClient,
    items: list[TranslationItem],
    cfg: PipelineConfig,
    batch_size: int,
    concurrency: int,
    enable_semantic_qa: bool,
    enable_relation_qa: bool,
    enable_backtranslation: bool,
    cache: TranslationCache | None = None,
    relation_sample_limit: int = 20,
) -> None:
    untranslated = list(items)
    cache_hits = _apply_cache(untranslated, cache)
    if cache_hits:
        logger.info("Translation cache hits: %s/%s", cache_hits, len(untranslated))

    for round_idx in range(MAX_RETRY_ROUNDS):
        if not untranslated:
            break

        repair_hint = None
        if round_idx > 0:
            repair_hint = (
                "Fix QA failures. Keep item_id/group_id/role unchanged. "
                "Do not rewrite positive/negative intent. Preserve entities and terms."
            )

        need_translate = (
            [item for item in untranslated if not item.msa_raw]
            if round_idx == 0
            else list(untranslated)
        )
        if need_translate:
            translate_batch(
                client,
                need_translate,
                cfg,
                batch_size=batch_size,
                concurrency=concurrency,
                repair_hint=repair_hint,
            )

        failed: list[TranslationItem] = []
        for item in untranslated:
            _merge_qa(item, cfg)

            if enable_semantic_qa:
                meaning = judge_meaning(client, item)
                item.qa.update(meaning)

            if enable_backtranslation:
                bt = backtranslate(client, item)
                item.qa["backtranslation"] = bt.get("back_idn", "")

            hard_ok = item.qa.get("hard_pass", False)
            semantic_ok = item.qa.get("semantic_pass", True) if enable_semantic_qa else True
            item.retry_rounds = round_idx

            if hard_ok and semantic_ok:
                item.final_status = "accepted"
                if cache:
                    cache.set(item.source_idn, item.msa_raw)
            else:
                item.final_status = "pending"
                failed.append(item)

        _log_qa_round(round_idx, untranslated, failed)

        if not failed:
            break

        error_by_id = {}
        for item in failed:
            errors = (
                item.qa.get("hard_errors", [])
                + item.qa.get("critical_errors", [])
                + item.qa.get("entity_errors", [])
                + item.qa.get("term_errors", [])
            )
            error_by_id[item.item_id] = "; ".join(map(str, errors)) or "semantic QA failed"

        repair_batch(client, failed, cfg, error_by_id, concurrency=concurrency)
        untranslated = failed

    _update_cache(items, cache)

    if enable_relation_qa:
        query_item = next(i for i in items if i.role == "query")
        for idx, candidate in enumerate(i for i in items if i.role != "query"):
            if candidate.role == "negative" and relation_sample_limit and idx >= relation_sample_limit:
                candidate.qa.setdefault("relation_preserved", True)
                candidate.qa.setdefault("relation_risk_level", "low")
                candidate.qa.setdefault("relation_skipped", True)
                continue
            relation = judge_relation(client, query_item, candidate)
            candidate.qa.update(relation)
            if not relation.get("relation_preserved", True):
                candidate.final_status = "failed"
                candidate.qa["semantic_pass"] = False

    for item in items:
        if item.final_status == "accepted":
            continue
        if item.final_status == "pending":
            item.final_status = "failed"
            item.qa["needs_manual_review"] = True
