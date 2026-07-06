from __future__ import annotations

import json
from typing import Any

from .expand import TranslationItem
from .kimi_client import KimiClient

MEANING_JUDGE_SYSTEM = """你是翻译质量审查员。请判断印尼语源句和阿拉伯语 MSA 译文是否语义一致。
只输出 JSON：
{
  "score": 1,
  "meaning_preserved": true,
  "critical_errors": [],
  "minor_errors": [],
  "entity_errors": [],
  "term_errors": [],
  "should_retry": false
}
"""

RELATION_JUDGE_SYSTEM = """你是金融客服检索数据质检员。请判断翻译前后 query-candidate 的语义关系是否保持一致。
只输出 JSON：
{
  "relation_preserved": true,
  "risk_level": "low",
  "reason": "...",
  "should_retry": false
}
"""

BACKTRANSLATE_SYSTEM = """你是翻译器。请将阿拉伯语 MSA 金融客服文本回译为印尼语。
要求：忠实回译，不解释。只输出 JSON: {"back_idn": "..."}
"""


def judge_meaning(client: KimiClient, item: TranslationItem) -> dict[str, Any]:
    payload = {
        "source_idn": item.source_idn,
        "translation_msa": item.msa_raw,
        "entities_found": item.entities_found,
        "terms_found": item.terms_found,
        "actions_found": item.actions_found,
        "role": item.role,
    }
    result = client.chat_json(
        MEANING_JUDGE_SYSTEM,
        json.dumps(payload, ensure_ascii=False),
    )
    score = int(result.get("score", 0))
    meaning_preserved = bool(result.get("meaning_preserved", False))
    semantic_pass = (
        score >= 4
        and meaning_preserved
        and not result.get("critical_errors")
        and not result.get("entity_errors")
    )
    result["semantic_pass"] = semantic_pass
    return result


def judge_relation(
    client: KimiClient,
    query_item: TranslationItem,
    candidate_item: TranslationItem,
) -> dict[str, Any]:
    payload = {
        "label": candidate_item.role,
        "source_query_idn": query_item.source_idn,
        "source_candidate_idn": candidate_item.source_idn,
        "msa_query": query_item.msa_raw,
        "msa_candidate": candidate_item.msa_raw,
    }
    result = client.chat_json(
        RELATION_JUDGE_SYSTEM,
        json.dumps(payload, ensure_ascii=False),
    )
    result.setdefault("relation_preserved", True)
    return result


def backtranslate(client: KimiClient, item: TranslationItem) -> dict[str, Any]:
    payload = {"translation_msa": item.msa_raw}
    result = client.chat_json(
        BACKTRANSLATE_SYSTEM,
        json.dumps(payload, ensure_ascii=False),
    )
    return result
