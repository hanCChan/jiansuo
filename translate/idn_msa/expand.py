from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranslationItem:
    item_id: str
    group_id: str
    role: str  # query | positive | negative
    candidate_index: int
    source_idn: str
    entities_found: list[str] = field(default_factory=list)
    terms_found: list[str] = field(default_factory=list)
    actions_found: list[str] = field(default_factory=list)
    masked_idn: str = ""
    msa_raw: str = ""
    msa_clean: str = ""
    msa_norm_for_embedding: str = ""
    qa: dict[str, Any] = field(default_factory=dict)
    retry_rounds: int = 0
    final_status: str = "pending"


def expand_record(record: dict[str, Any], group_idx: int) -> list[TranslationItem]:
    group_id = f"q_{group_idx:06d}"
    items: list[TranslationItem] = []

    items.append(
        TranslationItem(
            item_id=f"{group_id}_query",
            group_id=group_id,
            role="query",
            candidate_index=0,
            source_idn=record["query"],
        )
    )

    for i, text in enumerate(record.get("positive", [])):
        items.append(
            TranslationItem(
                item_id=f"{group_id}_p_{i:03d}",
                group_id=group_id,
                role="positive",
                candidate_index=i,
                source_idn=text,
            )
        )

    for i, text in enumerate(record.get("negative", [])):
        items.append(
            TranslationItem(
                item_id=f"{group_id}_n_{i:04d}",
                group_id=group_id,
                role="negative",
                candidate_index=i,
                source_idn=text,
            )
        )

    return items


def assemble_group_output(group_id: str, items: list[TranslationItem]) -> dict[str, Any]:
    query_item = next(i for i in items if i.role == "query")
    positives = [i for i in items if i.role == "positive"]
    negatives = [i for i in items if i.role == "negative"]

    hard_pass = all(i.qa.get("hard_pass", False) for i in items)
    semantic_pass = all(i.qa.get("semantic_pass", True) for i in items)
    relation_pass = all(i.qa.get("relation_preserved", True) for i in items if i.role != "query")
    max_retry = max((i.retry_rounds for i in items), default=0)
    failed = [i.item_id for i in items if i.final_status != "accepted"]

    return {
        "id": group_id.replace("q_", ""),
        "query_id": group_id,
        "query_idn": query_item.source_idn,
        "query_msa": query_item.msa_raw,
        "query_msa_norm_for_embedding": query_item.msa_norm_for_embedding,
        "positive": [
            {
                "id": p.item_id,
                "idn": p.source_idn,
                "msa": p.msa_raw,
                "msa_norm_for_embedding": p.msa_norm_for_embedding,
                "qa": p.qa,
                "final_status": p.final_status,
            }
            for p in positives
        ],
        "negative": [
            {
                "id": n.item_id,
                "idn": n.source_idn,
                "msa": n.msa_raw,
                "msa_norm_for_embedding": n.msa_norm_for_embedding,
                "qa": n.qa,
                "final_status": n.final_status,
            }
            for n in negatives
        ],
        "qa": {
            "hard_pass": hard_pass,
            "semantic_pass": semantic_pass,
            "relation_preserved": relation_pass,
            "repair_rounds": max_retry,
            "failed_items": failed,
        },
    }
