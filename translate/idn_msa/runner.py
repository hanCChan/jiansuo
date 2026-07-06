from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config_loader import PipelineConfig
from .expand import TranslationItem, assemble_group_output_debug, assemble_group_output_simple, expand_record
from .hard_qa import check_structure
from .kimi_client import KimiClient
from .mask import annotate_item
from .pipeline import process_items_with_retry
from .translation_cache import TranslationCache

logger = logging.getLogger(__name__)


def precheck_source(record: dict[str, Any], group_idx: int) -> list[str]:
    errors: list[str] = []
    if not record.get("query", "").strip():
        errors.append("empty_query")
    positives = record.get("positive", [])
    negatives = record.get("negative", [])
    if not positives:
        errors.append("empty_positive")
    if not negatives:
        errors.append("empty_negative")
    if len(set(positives)) != len(positives):
        errors.append("duplicate_positive")
    return errors


def prepare_items(record: dict[str, Any], group_idx: int, cfg: PipelineConfig) -> list[TranslationItem]:
    items = expand_record(record, group_idx)
    for item in items:
        masked, entities, terms, actions = annotate_item(item.source_idn, cfg)
        item.masked_idn = masked
        item.entities_found = entities
        item.terms_found = terms
        item.actions_found = actions
    return items


def run_group(
    client: KimiClient,
    record: dict[str, Any],
    group_idx: int,
    cfg: PipelineConfig,
    batch_size: int,
    concurrency: int,
    enable_semantic_qa: bool,
    enable_relation_qa: bool,
    enable_backtranslation: bool,
    relation_sample_limit: int,
    cache: TranslationCache | None = None,
) -> dict[str, Any]:
    pre_errors = precheck_source(record, group_idx)
    if pre_errors:
        raise ValueError(f"source precheck failed for group {group_idx}: {pre_errors}")

    items = prepare_items(record, group_idx, cfg)
    expected = {
        "query": 1,
        "positive": len(record.get("positive", [])),
        "negative": len(record.get("negative", [])),
    }
    struct_errors = check_structure(items, expected)
    if struct_errors:
        raise ValueError(f"structure error for group {group_idx}: {struct_errors}")

    process_items_with_retry(
        client=client,
        items=items,
        cfg=cfg,
        batch_size=batch_size,
        concurrency=concurrency,
        enable_semantic_qa=enable_semantic_qa,
        enable_relation_qa=enable_relation_qa,
        enable_backtranslation=enable_backtranslation,
        relation_sample_limit=relation_sample_limit,
        cache=cache,
    )

    group_id = items[0].group_id
    debug = assemble_group_output_debug(group_id, items)
    simple = assemble_group_output_simple(items)
    return {
        "query_id": group_id,
        "group_index": group_idx,
        "simple": simple,
        "debug": debug,
        "eval_ready": not debug["qa"]["failed_items"],
    }


def write_msa_eval_json(simple_records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(simple_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_qa_report(debug_records: list[dict[str, Any]], report_path: Path) -> None:
    groups = [record["debug"] for record in debug_records]
    summary = {
        "total_groups": len(groups),
        "accepted_groups": sum(1 for g in groups if not g["qa"]["failed_items"]),
        "failed_groups": sum(1 for g in groups if g["qa"]["failed_items"]),
        "eval_ready_groups": sum(1 for record in debug_records if record.get("eval_ready")),
        "groups": [
            {
                "query_id": group["query_id"],
                "hard_pass": group["qa"]["hard_pass"],
                "semantic_pass": group["qa"]["semantic_pass"],
                "relation_preserved": group["qa"]["relation_preserved"],
                "failed_items": group["qa"]["failed_items"],
                "repair_rounds": group["qa"]["repair_rounds"],
                "eval_ready": not group["qa"]["failed_items"],
            }
            for group in groups
        ],
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
