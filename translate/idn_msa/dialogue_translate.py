from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from .config_loader import PipelineConfig
from .expand import TranslationItem
from .kimi_client import KimiClient
from .mask import restore_text

logger = logging.getLogger(__name__)

_TRANSLATION_KEYS = (
    "translation_msa",
    "content_msa",
    "msa",
    "translation",
    "arabic",
    "output",
    "text",
)


def _pick_translation(row: dict) -> str | None:
    for key in _TRANSLATION_KEYS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _pick_repair_translation(result: dict) -> str | None:
    direct = _pick_translation(result)
    if direct:
        return direct
    items = result.get("items")
    if isinstance(items, list) and items:
        return _pick_translation(items[0])
    return None


def _apply_translation_or_repair(
    client: KimiClient,
    item: TranslationItem,
    cfg: PipelineConfig,
    error_reason: str,
) -> None:
    repair_dialogue_single(client, item, cfg, error_reason)


DIALOGUE_TRANSLATE_SYSTEM_PROMPT = """你是专业的银行客服对话翻译器。任务：将印尼语客服对话 utterance 译为现代标准阿拉伯语（MSA）。

这是电话客服对话逐句翻译，不是改写、不是摘要、不是润色。

严格要求：
1. 只翻译输入的 content 字段语义，逐句忠实翻译。
2. 禁止新增信息、禁止删减信息、禁止解释、禁止合并多句、禁止拆分句子。
3. 禁止根据 speaker_role 改写语气风格；role 仅供语境参考，不得因此增删词句。
4. 保持 item_id 完全不变。
5. 输出必须是合法 JSON，禁止 Markdown、注释、思考过程。
6. 使用现代标准阿拉伯语（MSA），禁止方言。
7. 必须保留所有 <ENT_...> 与 <SENS_...> 占位符，字符级原样保留，不得翻译或替换。
8. 产品名/品牌名保持拉丁写法：myBCA、BCA、BCA ID、BCA Mobile、KlikBCA、Halo BCA、ATM、OTP、PIN、QRIS 等。
9. 术语保持一致：
   - password → كلمة المرور
   - reset password → إعادة تعيين كلمة المرور
   - login / masuk → تسجيل الدخول
   - terblokir / blokir → محظور / مقفل（按语境）
   - membuka blokir / buka blokir → فك الحظر / إلغاء الحظر
   - kartu debit / kartu ATM → بطاقة الخصم / بطاقة ATM
   - kartu kredit → بطاقة الائتمان
   - rekening / akun → الحساب
   - nomor HP / handphone → رقم الهاتف المحمول
   - verifikasi wajah → التحقق من الوجه
   - kode OTP → رمز التحقق (OTP)
   - PIN → الرقم السري (PIN)
10. 不要把 password 与 PIN 互相替换。
11. 人名可音译或保留，但不要编造新名字。
12. 允许与原文长度接近；不要扩写成长段落。

输出 JSON 格式：
{
  "items": [
    {
      "item_id": "...",
      "translation_msa": "..."
    }
  ]
}
"""

DIALOGUE_REPAIR_SYSTEM_PROMPT = """你是银行客服对话翻译修复器。根据 hard QA 错误原因修正阿拉伯语 MSA 译文。

要求：
1. 只输出 JSON: {"translation_msa": "..."}
2. 忠实翻译，禁止润色、扩写、摘要
3. 保留 <ENT_...> 与 <SENS_...> 占位符
4. 禁止 Markdown 与解释
"""


def _chunked(items: list[TranslationItem], batch_size: int) -> Iterable[list[TranslationItem]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _speaker_role(item: TranslationItem) -> str:
    return str(item.qa.get("speaker_role") or item.role or "unknown")


def _translate_one_batch(
    client: KimiClient,
    batch: list[TranslationItem],
    cfg: PipelineConfig,
    repair_hint: str | None,
    batch_idx: int,
) -> None:
    payload = [
        {
            "item_id": item.item_id,
            "speaker_role": _speaker_role(item),
            "content": item.masked_idn,
        }
        for item in batch
    ]
    user_prompt = json.dumps(
        {
            "task": "translate_dialogue_content_idn_to_msa",
            "repair_hint": repair_hint,
            "items": payload,
        },
        ensure_ascii=False,
    )
    logger.info("Dialogue translate batch %s (%s items)", batch_idx, len(batch))
    result = client.chat_json(DIALOGUE_TRANSLATE_SYSTEM_PROMPT, user_prompt)
    rows = result.get("items", [])
    if not isinstance(rows, list):
        raise ValueError(f"Kimi response missing items list: {result!r}")

    translations: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = row.get("item_id")
        text = _pick_translation(row)
        if item_id and text:
            translations[str(item_id)] = text

    missing = [item for item in batch if item.item_id not in translations]
    if missing:
        logger.warning(
            "Batch %s: Kimi returned %s/%s items; repairing missing individually",
            batch_idx,
            len(batch) - len(missing),
            len(batch),
        )
        for item in missing:
            _apply_translation_or_repair(
                client,
                item,
                cfg,
                "Kimi batch JSON missing translation_msa; retranslate faithfully.",
            )
            text = item.msa_raw
            if text:
                translations[item.item_id] = text

    for item in batch:
        if item.item_id not in translations:
            raise KeyError(
                f"missing translation for {item.item_id} after batch+repair; "
                f"batch_keys={list(translations)[:5]}"
            )
        item.msa_raw = restore_text(translations[item.item_id], cfg)


def translate_dialogue_batch(
    client: KimiClient,
    items: list[TranslationItem],
    cfg: PipelineConfig,
    batch_size: int = 20,
    concurrency: int = 1,
    repair_hint: str | None = None,
) -> None:
    batches = list(_chunked(items, batch_size))
    if concurrency <= 1 or len(batches) <= 1:
        for idx, batch in enumerate(batches, start=1):
            _translate_one_batch(client, batch, cfg, repair_hint, idx)
        return

    workers = min(concurrency, len(batches))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_translate_one_batch, client, batch, cfg, repair_hint, idx): idx
            for idx, batch in enumerate(batches, start=1)
        }
        for future in as_completed(futures):
            future.result()


def repair_dialogue_single(
    client: KimiClient,
    item: TranslationItem,
    cfg: PipelineConfig,
    error_reason: str,
) -> None:
    user_prompt = json.dumps(
        {
            "item_id": item.item_id,
            "speaker_role": _speaker_role(item),
            "content": item.masked_idn,
            "previous_translation_msa": item.msa_raw,
            "error_reason": error_reason,
        },
        ensure_ascii=False,
    )
    result = client.chat_json(DIALOGUE_REPAIR_SYSTEM_PROMPT, user_prompt)
    text = _pick_repair_translation(result)
    if not text:
        raise KeyError(f"repair response missing translation field: {result!r}")
    item.msa_raw = restore_text(text, cfg)


def repair_dialogue_batch(
    client: KimiClient,
    items: list[TranslationItem],
    cfg: PipelineConfig,
    error_by_id: dict[str, str],
    concurrency: int = 8,
) -> None:
    if concurrency <= 1 or len(items) <= 1:
        for item in items:
            repair_dialogue_single(
                client,
                item,
                cfg,
                error_by_id.get(item.item_id, "hard QA failed"),
            )
        return

    workers = min(concurrency, len(items))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                repair_dialogue_single,
                client,
                item,
                cfg,
                error_by_id.get(item.item_id, "hard QA failed"),
            )
            for item in items
        ]
        for future in as_completed(futures):
            future.result()
