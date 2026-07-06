from __future__ import annotations

import json
from typing import Iterable

from .expand import TranslationItem
from .kimi_client import KimiClient
from .mask import restore_text
from .config_loader import PipelineConfig

TRANSLATE_SYSTEM_PROMPT = """你是专业的金融客服语料翻译器。任务是将印尼语银行客服 query/candidate 翻译为现代标准阿拉伯语 MSA。

严格要求：
1. 只翻译语义，不新增、不删除、不解释。
2. 保持 item_id、group_id、role 完全不变。
3. positive 和 negative 只是数据标签，不要根据 query 去改写 candidate。
4. 每个 candidate 必须独立忠实翻译，不能为了让 positive 更像 query 或让 negative 更不像 query 而改写。
5. 输出必须是合法 JSON，不能输出 Markdown、解释、注释。
6. 使用现代标准阿拉伯语，避免方言词。
7. 保留所有 <ENT_...> 占位符，不要翻译或替换它们。
8. 术语翻译保持一致：
   - password → كلمة المرور
   - reset password → إعادة تعيين كلمة المرور
   - login → تسجيل الدخول
   - blocked / terblokir → محظور / مقفل حسب السياق
   - membuka blokir → فك الحظر / إلغاء الحظر
   - kartu debit → بطاقة الخصم
   - kartu kredit → بطاقة الائتمان
   - rekening / akun → الحساب
   - nomor HP → رقم الهاتف المحمول
   - verifikasi wajah → التحقق من الوجه
   - kode OTP → رمز التحقق لمرة واحدة (OTP)
   - PIN → الرقم السري (PIN)
9. 不要把 myBCA、m-BCA、KlikBCA、BCA ID 互相替换。
10. 不要把 password 翻成 PIN，也不要把 PIN 翻成 password。

输出 JSON 格式：
{
  "items": [
    {
      "item_id": "...",
      "group_id": "...",
      "role": "...",
      "translation_msa": "..."
    }
  ]
}
"""


REPAIR_SYSTEM_PROMPT = """你是金融客服语料翻译修复器。请根据错误原因修正阿拉伯语 MSA 译文。
要求：
1. 只输出 JSON: {"translation_msa": "..."}
2. 不要 Markdown，不要解释
3. 保留 <ENT_...> 占位符直到后续 restore
4. 独立忠实翻译，不根据 query 或 label 改写
"""


def _chunked(items: list[TranslationItem], batch_size: int) -> Iterable[list[TranslationItem]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def translate_batch(
    client: KimiClient,
    items: list[TranslationItem],
    cfg: PipelineConfig,
    batch_size: int = 20,
    repair_hint: str | None = None,
) -> None:
    for batch in _chunked(items, batch_size):
        payload = [
            {
                "item_id": item.item_id,
                "group_id": item.group_id,
                "role": item.role,
                "source_idn_masked": item.masked_idn,
            }
            for item in batch
        ]
        user_prompt = json.dumps(
            {
                "task": "translate_idn_to_msa",
                "repair_hint": repair_hint,
                "items": payload,
            },
            ensure_ascii=False,
        )
        result = client.chat_json(TRANSLATE_SYSTEM_PROMPT, user_prompt)
        translations = {row["item_id"]: row["translation_msa"] for row in result.get("items", [])}
        for item in batch:
            if item.item_id not in translations:
                raise KeyError(f"missing translation for {item.item_id}")
            restored = restore_text(translations[item.item_id], cfg)
            item.msa_raw = restored


def repair_single(
    client: KimiClient,
    item: TranslationItem,
    cfg: PipelineConfig,
    error_reason: str,
) -> None:
    user_prompt = json.dumps(
        {
            "item_id": item.item_id,
            "role": item.role,
            "source_idn_masked": item.masked_idn,
            "previous_translation_msa": item.msa_raw,
            "error_reason": error_reason,
        },
        ensure_ascii=False,
    )
    result = client.chat_json(REPAIR_SYSTEM_PROMPT, user_prompt)
    item.msa_raw = restore_text(result["translation_msa"], cfg)
