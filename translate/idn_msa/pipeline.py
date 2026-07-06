from __future__ import annotations

from .expand import TranslationItem
from .hard_qa import apply_postprocess, hard_qa_item
from .kimi_client import KimiClient
from .semantic_qa import backtranslate, judge_meaning, judge_relation
from .translate import repair_single, translate_batch
from .config_loader import PipelineConfig

MAX_RETRY_ROUNDS = 3


def _merge_qa(item: TranslationItem, cfg: PipelineConfig) -> None:
    apply_postprocess(item, cfg)
    item.qa.update(hard_qa_item(item, cfg))


def process_items_with_retry(
    client: KimiClient,
    items: list[TranslationItem],
    cfg: PipelineConfig,
    batch_size: int,
    enable_semantic_qa: bool,
    enable_relation_qa: bool,
    enable_backtranslation: bool,
    relation_sample_limit: int = 20,
) -> None:
    untranslated = list(items)

    for round_idx in range(MAX_RETRY_ROUNDS):
        if not untranslated:
            break

        repair_hint = None
        if round_idx > 0:
            repair_hint = (
                "Fix QA failures. Keep item_id/group_id/role unchanged. "
                "Do not rewrite positive/negative intent. Preserve entities and terms."
            )

        translate_batch(
            client,
            untranslated,
            cfg,
            batch_size=batch_size,
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
            else:
                item.final_status = "pending"
                failed.append(item)

        if not failed:
            break

        for item in failed:
            errors = (
                item.qa.get("hard_errors", [])
                + item.qa.get("critical_errors", [])
                + item.qa.get("entity_errors", [])
                + item.qa.get("term_errors", [])
            )
            repair_single(
                client,
                item,
                cfg,
                "; ".join(map(str, errors)) or "semantic QA failed",
            )

        untranslated = failed

    query_item = next(i for i in items if i.role == "query")
    if enable_relation_qa:
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
