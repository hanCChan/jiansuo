#!/usr/bin/env python3
"""Fill XXX placeholders in translated dialogue MSA using Kimi 2.6."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idn_msa.dialogue_mask import SENSITIVE_TOKEN_RE
from idn_msa.kimi_client import KimiClient

DEFAULT_INPUT = Path("/data1/hcc/jiansuo/dh/dialogue_20260615_BCA_clean_test_msa.json")
DEFAULT_OUTPUT = Path("/data1/hcc/jiansuo/dh/dialogue_20260615_BCA_clean_test_msa_filled.json")
DEFAULT_CACHE = Path(__file__).resolve().parent / "xxx_fill_cache.jsonl"
DEFAULT_DEBUG = Path(__file__).resolve().parent / "xxx_fill_debug.jsonl"
DEFAULT_BASE_URL = "http://10.16.137.2:8000/v1"
DEFAULT_MODEL = "Kimi-K2.6-CT-FP8KV"

SYSTEM_PROMPT = """你是银行客服对话脱敏还原助手。任务：根据印尼语原句和阿拉伯语译文语境，将译文中的 XXX 占位符补全为合理的合成数值/文本。

背景：
- XXX / XXXXXX 等是脱敏占位符，可能代表：次数、金额、账号片段、手机号片段、日期、验证码位数等。
- 印尼语原句与阿拉伯语译文语义一致，可互相参照理解每个 XXX 的含义。

严格要求：
1. 只补全 XXX 占位符，其余译文逐字保留（含 BCA、myBCA、OTP、PIN 等拉丁产品名）。
2. 生成随机合成值（非真实个人信息），供训练使用：
   - 卡号：16 位随机数字
   - 手机号：08xx + 8 位数字
   - 账号/片段：数字长度与 X 个数大致相当
   - 姓名：常见阿拉伯语合成姓名
   禁止用泛化短语（如「المعلومات المذكورة」）替代数字型 PII。
3. 结合上下文推断类型：
   - 「XXX kali / مرات」→ 次数，如 3
   - 「XXX ribu / ألف」→ 金额，如 50
   - 「nomor / رقم / rekening / حساب」→ 数字串，长度与 X 个数大致相当
   - 「tanggal / تاريخ」→ 日期格式
4. 按原文中 XXX 出现顺序依次补全。
5. 输出合法 JSON，禁止 Markdown。

输出格式：
{
  "content_msa_filled": "补全后的阿拉伯语句子",
  "fills": [
    {"index": 1, "token": "XXX", "value": "3", "type": "count"}
  ]
}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill XXX placeholders in dialogue MSA.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--debug", type=Path, default=DEFAULT_DEBUG)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--assemble-only", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def stable_id(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]


def find_xxx_tokens(text: str) -> list[str]:
    return SENSITIVE_TOKEN_RE.findall(text)


def load_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            src = row.get("source_idn")
            filled = row.get("content_msa_filled")
            if src and filled:
                cache[src] = filled
    return cache


def save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for src in sorted(cache):
            f.write(
                json.dumps(
                    {"source_idn": src, "content_msa_filled": cache[src]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def append_debug(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_items(dialogues: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for dlg in dialogues:
        for turn in dlg.get("turns", []):
            content = (turn.get("content") or "").strip()
            msa = (turn.get("content_msa") or "").strip()
            if not content or not msa or not find_xxx_tokens(msa):
                continue
            if content not in items:
                items[content] = {
                    "source_idn": content,
                    "content_msa": msa,
                    "speaker_role": turn.get("role") or "unknown",
                    "xxx_count": len(find_xxx_tokens(msa)),
                }
    return items


def build_user_prompt(item: dict[str, Any]) -> str:
    return json.dumps(
        {
            "source_idn": item["source_idn"],
            "content_msa_with_xxx": item["content_msa"],
            "speaker_role": item["speaker_role"],
            "xxx_tokens_in_order": find_xxx_tokens(item["content_msa"]),
        },
        ensure_ascii=False,
    )


def pick_filled_text(result: dict[str, Any]) -> str | None:
    for key in ("content_msa_filled", "msa_filled", "filled", "translation_msa", "text"):
        val = result.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def validate_fill(original_msa: str, filled: str) -> list[str]:
    errors: list[str] = []
    if not filled.strip():
        errors.append("empty_filled")
    orig_cnt = len(find_xxx_tokens(original_msa))
    left_cnt = len(find_xxx_tokens(filled))
    if left_cnt >= orig_cnt:
        errors.append(f"xxx_not_reduced:{left_cnt}/{orig_cnt}")
    return errors


def fill_one(client: KimiClient, item: dict[str, Any]) -> dict[str, Any]:
    result = client.chat_json(SYSTEM_PROMPT, build_user_prompt(item), temperature=0.0)
    filled = pick_filled_text(result)
    if not filled:
        raise KeyError(f"missing content_msa_filled in response: {result!r}")
    errors = validate_fill(item["content_msa"], filled)
    return {
        "item_id": stable_id(item["source_idn"]),
        "source_idn": item["source_idn"],
        "content_msa": item["content_msa"],
        "content_msa_filled": filled,
        "fills": result.get("fills", []),
        "status": "accepted" if not errors else "failed",
        "errors": errors,
    }


def merge_into_dialogues(
    dialogues: list[dict[str, Any]],
    filled_by_source: dict[str, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dlg in dialogues:
        new_dlg = dict(dlg)
        turns = []
        for turn in dlg.get("turns", []):
            new_turn = dict(turn)
            content = (turn.get("content") or "").strip()
            if content in filled_by_source:
                new_turn["content_msa_filled"] = filled_by_source[content]
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

    dialogues = json.loads(args.input.read_text(encoding="utf-8"))
    items = collect_items(dialogues)
    cache = load_cache(args.cache)

    pending = [src for src in items if src not in cache]
    logging.info(
        "unique_with_xxx=%s cached=%s pending=%s",
        len(items),
        len(cache),
        len(pending),
    )

    if args.max_items:
        pending = pending[: args.max_items]

    enable_thinking = args.enable_thinking

    if not args.assemble_only and pending:
        client = KimiClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            enable_thinking=enable_thinking,
        )

        def _work(src: str) -> dict[str, Any]:
            return fill_one(client, items[src])

        done_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
            futures = {pool.submit(_work, src): src for src in pending}
            for idx, fut in enumerate(as_completed(futures), start=1):
                src = futures[fut]
                try:
                    row = fut.result()
                    if row["status"] == "accepted":
                        cache[src] = row["content_msa_filled"]
                    done_rows.append(row)
                    filled_text = row.get("content_msa_filled") or ""
                    logging.info(
                        "Done %s/%s status=%s xxx_left=%s",
                        idx,
                        len(pending),
                        row["status"],
                        len(find_xxx_tokens(filled_text)),
                    )
                except Exception as exc:  # noqa: BLE001
                    logging.error("Failed %s: %s", stable_id(src), exc)
                    done_rows.append(
                        {
                            "item_id": stable_id(src),
                            "source_idn": src,
                            "status": "error",
                            "errors": [str(exc)],
                        }
                    )
                if idx % 20 == 0:
                    save_cache(args.cache, cache)
                    append_debug(args.debug, done_rows)
                    done_rows = []

        if done_rows:
            append_debug(args.debug, done_rows)
        save_cache(args.cache, cache)

    filled_map = dict(cache)
    out_dialogues = merge_into_dialogues(dialogues, filled_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out_dialogues, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    turns_filled = sum(
        1
        for dlg in out_dialogues
        for t in dlg.get("turns", [])
        if (t.get("content_msa_filled") or "").strip()
    )
    turns_with_xxx = sum(
        1
        for dlg in dialogues
        for t in dlg.get("turns", [])
        if find_xxx_tokens((t.get("content_msa") or ""))
    )
    logging.info(
        "Wrote %s | turns_with_xxx=%s | turns_with_content_msa_filled=%s | cache=%s",
        args.output,
        turns_with_xxx,
        turns_filled,
        len(cache),
    )


if __name__ == "__main__":
    main()
