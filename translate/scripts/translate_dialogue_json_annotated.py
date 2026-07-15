#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【学习用注释版】与 translate_dialogue_json.py 逻辑一致，逐段/逐行中文注释。
生产跑批请用无注释的 translate_dialogue_json.py。

整体角色：对话 JSON 翻译「总调度器」(orchestrator)
  - 不负责具体翻译 prompt / QA 规则细节
  - 负责：读 JSON → 去重 → 白名单 → 分批调 Kimi → 合并写回 content_msa

依赖模块速查：
  config_loader      → 加载 entities/glossary/action_polarity 等 yaml
  dialogue_mask      → 白名单跳过 + XXX 敏感掩码/还原
  mask               → 实体 <ENT_xx> 掩码 + 术语/动作标注
  dialogue_translate   → 组 prompt、批量调 Kimi 翻译/修复
  dialogue_pipeline  → 3 轮 translate+repair+hard_qa 循环
  hard_qa            → 阿拉伯语比例、术语、极性等硬规则
  kimi_client        → OpenAI 兼容 API 封装
  translation_cache  → 原文 exact match 缓存，避免重复计费
  expand             → TranslationItem 数据结构
  placeholder_restore→ 修复 Kimi 幻觉的 <ENT_*> 占位符
"""

from __future__ import annotations  # 允许用 str 代替尚未定义的类名做类型注解

import argparse   # 解析命令行参数 --input --resume 等
import hashlib    # 用 content 生成稳定的 item_id
import json       # 读写 JSON / JSONL
import logging    # 打日志到 stdout
import sys        # 修改模块搜索路径
from pathlib import Path  # 跨平台路径
from typing import Any    # 宽松类型注解

# 定位 translate/ 包根目录，使 import idn_msa 生效
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------- 下游模块 import（按调用顺序理解）----------

from idn_msa.config_loader import load_config
# ↑ 读取 config/*.yaml，返回 PipelineConfig（实体表、术语表、极性规则等）

from idn_msa.dialogue_mask import (
    load_skip_whitelist,      # 读 dialogue_skip_whitelist.yaml
    mask_sensitive_tokens,    # XXX → <SENS_00>，翻译后再还原
    should_skip_content,      # 判断是否整句跳过（白名单）
)
from idn_msa.dialogue_pipeline import process_dialogue_items_with_retry
# ↑ 核心：一轮 translate + 最多两轮 repair，每轮后 hard_qa

from idn_msa.expand import TranslationItem
# ↑ 单条翻译任务的数据容器（原文、译文、QA、状态）

from idn_msa.hard_qa import apply_postprocess, hard_qa_item
# ↑ 归一化 unicode + 执行硬 QA，返回 hard_pass / hard_errors

from idn_msa.kimi_client import KimiClient
# ↑ 调 Kimi HTTP API，强制 JSON 输出

from idn_msa.mask import annotate_item
# ↑ 实体掩码 + 抽 terms_found / actions_found（供 QA 用）

from idn_msa.translation_cache import TranslationCache
# ↑ 内存 dict + jsonl 持久化，key=印尼原文 exact string

# ---------- 默认路径（test 集；training 可命令行覆盖）----------

DEFAULT_INPUT = Path("/data1/hcc/jiansuo/dh/dialogue_20260615_BCA_clean_test(1).json")
DEFAULT_OUTPUT = Path("/data1/hcc/jiansuo/dh/dialogue_20260615_BCA_clean_test_msa.json")
DEFAULT_EXTRACT = Path("/data1/hcc/jiansuo/dh/dialogue_test_content_extract.jsonl")   # 展平后的 turn 清单
DEFAULT_CACHE = Path("/data1/hcc/jiansuo/dh/dialogue_test_translation_cache.jsonl")   # 成功译文缓存
DEFAULT_DEBUG = Path("/data1/hcc/jiansuo/dh/dialogue_test_translation_debug.jsonl")   # 每条 QA 明细
DEFAULT_FAILED = Path("/data1/hcc/jiansuo/dh/dialogue_test_translation_failed.jsonl") # QA 失败 unique
DEFAULT_BASE_URL = "http://10.16.137.2:8000/v1"  # 内网 Kimi 服务
DEFAULT_MODEL = "Kimi-K2.6-CT-FP8KV"


def parse_args() -> argparse.Namespace:
    """解析 CLI；所有路径/并发参数可覆盖默认值。"""
    p = argparse.ArgumentParser(description="Translate dialogue content to MSA.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)       # 输入对话 JSON
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)     # 输出带 content_msa 的 JSON
    p.add_argument("--extract", type=Path, default=DEFAULT_EXTRACT) # 调试：展平 extract
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)       # 译文缓存 jsonl
    p.add_argument("--debug", type=Path, default=DEFAULT_DEBUG)       # 调试 jsonl
    p.add_argument("--failed", type=Path, default=DEFAULT_FAILED)     # 失败清单
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)            # Kimi API 地址
    p.add_argument("--api-key", default="EMPTY")                    # 内网常为空
    p.add_argument("--model", default=DEFAULT_MODEL)                  # 模型名
    p.add_argument("--batch-size", type=int, default=30)            # 每次 prompt 里几条（送 Kimi）
    p.add_argument("--concurrency", type=int, default=8)              # 并发批次数
    p.add_argument("--wave-size", type=int, default=500)              # 每波最多翻多少 unique
    p.add_argument("--enable-thinking", action="store_true")        # Kimi 思考模式
    p.add_argument("--max-items", type=int, default=0)                # 调试：限制条数
    p.add_argument("--resume", action="store_true")                   # 跳过 debug 里已处理的
    p.add_argument("--retry-failed", action="store_true")             # 只重跑 failed 文件里的 unique
    p.add_argument("--assemble-only", action="store_true")            # 不调 API，只 cache+白名单合并
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def stable_item_id(content: str) -> str:
    """同一 content 永远同一 ID，便于 Kimi JSON 对齐与 debug 追踪。"""
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"dlg_{digest}"


def iter_turn_refs(dialogues: list[dict]) -> list[dict[str, Any]]:
    """
    把嵌套 dialogues 展平为 turn 级引用列表。
    一条 ref = 一个 turn 在输出 JSON 里的坐标 + content。
    """
    refs: list[dict[str, Any]] = []
    for dlg in dialogues:
        dialogue_id = dlg["dialogue_id"]
        for turn in dlg.get("turns", []):
            refs.append({
                "dialogue_id": dialogue_id,
                "turn": turn.get("turn"),
                "role": turn.get("role", ""),       # Agent / Customer，送 Kimi 作语境
                "row_id": turn.get("row_id"),
                "content": turn.get("content", ""), # 印尼语原文
            })
    return refs


def write_extract(refs: list[dict[str, Any]], path: Path) -> None:
    """把展平结果写成 JSONL，方便用 jq / pandas 检查。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in refs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_unique_contents(refs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    7435 turn → ~5789 unique content（大量重复寒暄句只翻一次）。
    返回：content -> 首次出现时的 metadata（speaker_role 等）。
    """
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
    """
    把一条 unique content 变成 TranslationItem（翻译流水线的工作单元）。

    分支 A：白名单 → skipped，msa_raw=原文，不调 Kimi
    分支 B：需翻译 → 掩码敏感词/实体，留给 dialogue_pipeline
    """
    item = TranslationItem(
        item_id=stable_item_id(content),
        group_id=f"dlg_wave_{wave_idx:04d}",  # 仅用于日志分组
        role="dialogue",
        candidate_index=0,
        source_idn=content,                    # 印尼语原文（cache key）
    )
    item.qa["speaker_role"] = meta.get("speaker_role", "unknown")

    # --- 模块 dialogue_mask.should_skip_content ---
    skip, reason = should_skip_content(content, skip_cfg)
    if skip:
        item.qa["skip_reason"] = reason
        item.msa_raw = content               # 白名单：MSA 字段直接复制拉丁/原文
        item.final_status = "skipped"
        item.qa["hard_pass"] = True
        return item

    # --- 模块 dialogue_mask：敏感 XXX 掩码 ---
    sens_masked, sens_pairs = mask_sensitive_tokens(content)
    # --- 模块 mask.annotate_item：实体掩码 + 术语/动作标注 ---
    masked, entities, terms, actions = annotate_item(sens_masked, cfg)
    item.masked_idn = masked                 # 送 Kimi 的实际文本（含 <ENT>/<SENS>）
    item.entities_found = entities           # hard_qa 检查实体是否保留
    item.terms_found = terms                 # password/pin/otp 等
    item.actions_found = actions             # memblokir / reset_password 等
    item.qa["sensitive_pairs"] = sens_pairs  # <SENS_00>↔XXX 还原表
    return item


def load_failed_rows(path: Path) -> list[dict[str, Any]]:
    """读取 failed.jsonl 全部行（含 dialogue_id/turn/qa/msa_raw）。"""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_failed_sources(path: Path) -> list[str]:
    """只取 failed 里的 unique content 字符串列表。"""
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
    """从 failed 行恢复 speaker_role 等 meta，供 make_dialogue_item 使用。"""
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
    """
    【--retry-failed 专用】在调 Kimi 之前本地预检：
      1) 现白名单 → skipped
      2) 旧 msa_raw + 新 hard_qa 通过 → 直接 accepted 写 cache
      3) 其余 → 返回 still_pending，交给后续 API 波次
    """
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
            debug_rows.append({
                "item_id": stable_item_id(content),
                "source_idn": content,
                "speaker_role": row.get("role") or "unknown",
                "msa_raw": content,
                "final_status": "skipped",
                "qa": {"skip_reason": reason, "hard_pass": True},
            })
            continue

        item = make_dialogue_item(content, failed_meta_by_content([row])[content], 0, cfg, skip_cfg)
        item.msa_raw = row.get("msa_raw") or ""   # 复用上次 Kimi 输出
        if item.msa_raw:
            sens_pairs = item.qa.get("sensitive_pairs") or []
            text = restore_sensitive_tokens(item.msa_raw, sens_pairs)
            text, _ = restore_kimi_placeholders(item.source_idn, text, cfg)
            item.msa_raw = text
            apply_postprocess(item, cfg)          # unicode 归一化
            item.qa.update(hard_qa_item(item, cfg))  # 用新规则重跑 QA
            if item.qa.get("hard_pass"):
                item.final_status = "accepted"
                cache.set(content, item.msa_raw)
                accepted += 1
                debug_rows.append({
                    "item_id": item.item_id,
                    "source_idn": content,
                    "speaker_role": item.qa.get("speaker_role"),
                    "msa_raw": item.msa_raw,
                    "final_status": "accepted",
                    "qa": item.qa,
                })
                continue

        still_pending.append(content)

        if idx % 20 == 0 or idx == len(failed_rows):
            logging.info("Retry preflight progress: %s/%s", idx, len(failed_rows))

    if debug_rows:
        append_jsonl(debug_path, debug_rows)
        cache.save(cache_path)
    logging.info(
        "Retry preflight: accepted_existing=%s skipped=%s still_need_api=%s",
        accepted, skipped, len(still_pending),
    )
    return still_pending


def load_processed_sources(debug_path: Path) -> set[str]:
    """--resume：debug 里已有 accepted/skipped/failed 的 source 不再重翻。"""
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
    """追加写 JSONL（debug/cache 增量持久化）。"""
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
    """
    每波结束必调（finally）：即使崩溃也落盘。
    pending → 标记 failed + needs_manual_review
    """
    for item in items:
        if item.final_status == "accepted" and item.msa_raw:
            cache.set(item.source_idn, item.msa_raw)
        elif item.final_status == "pending":
            item.final_status = "failed"
            item.qa["needs_manual_review"] = True

    debug_rows = [{
        "item_id": item.item_id,
        "source_idn": item.source_idn,
        "speaker_role": item.qa.get("speaker_role"),
        "msa_raw": item.msa_raw,
        "final_status": item.final_status,
        "qa": item.qa,
    } for item in items]
    append_jsonl(debug_path, debug_rows)
    cache.save(cache_path)


def build_translation_map(
    refs: list[dict[str, Any]],
    cache: TranslationCache,
    skip_cfg: dict[str, Any],
    debug_path: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """
    为每个 turn 的 content 查译文，优先级：
      1) 白名单 → 原文
      2) cache
      3) debug 里最新 accepted
      4) debug 里最新 failed → 记入 failures（不写 content_msa）
    """
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
                        latest_debug[src] = row  # 同 source 后者覆盖前者

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
            failures.append({
                "dialogue_id": ref["dialogue_id"],
                "turn": ref.get("turn"),
                "role": ref.get("role"),
                "row_id": ref.get("row_id"),
                "content": content,
                "qa": dbg.get("qa", {}),
                "msa_raw": dbg.get("msa_raw"),
            })

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
    """浅拷贝 dialogues，给每个 turn 挂上 content_msa（若有译文）。"""
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
    """主入口：串联 加载 → 待翻队列 → 分波 Kimi → 合并输出。"""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = load_config()                              # PipelineConfig
    skip_cfg = load_skip_whitelist(cfg.config_dir)   # 白名单 yaml

    dialogues = json.loads(args.input.read_text(encoding="utf-8"))
    refs = iter_turn_refs(dialogues)                 # 7435 条 turn 引用
    write_extract(refs, args.extract)
    logging.info("Wrote extract rows=%s -> %s", len(refs), args.extract)

    unique = collect_unique_contents(refs)           # ~5789 unique
    cache = TranslationCache(args.cache)             # 启动时加载已有 cache

    if args.retry_failed:
        # 重跑模式：只处理 failed.jsonl 里的 unique
        failed_rows = load_failed_rows(args.failed)
        failed_meta = failed_meta_by_content(failed_rows)
        unique.update(failed_meta)
        pending = preflight_accept_failed_retries(
            failed_rows, cfg, skip_cfg, cache, args.cache, args.debug,
        )
    else:
        # 正常模式：unique 中「非白名单且不在 cache」的待翻
        skip_count = sum(1 for c in unique if should_skip_content(c, skip_cfg)[0])
        pending = [c for c in unique if not should_skip_content(c, skip_cfg)[0] and not cache.get(c)]
        logging.info(
            "unique_content=%s skip_whitelist=%s cached=%s pending=%s",
            len(unique), skip_count, len(unique) - skip_count - len(pending), len(pending),
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

        # 分波：避免一次 pending 过大，且每波 finally 持久化
        for wave_idx, start in enumerate(range(0, len(pending), args.wave_size), start=1):
            wave_src = pending[start : start + args.wave_size]
            items = [
                make_dialogue_item(src, unique.get(src, {"speaker_role": "unknown"}), wave_idx, cfg, skip_cfg)
                for src in wave_src
            ]
            if not items:
                continue

            logging.info("Wave %s: translating %s unique strings (%s/%s)",
                           wave_idx, len(items), start + len(wave_src), len(pending))
            try:
                # ↓ 模块 dialogue_pipeline：translate → QA → repair ×2
                process_dialogue_items_with_retry(
                    client=client, items=items, cfg=cfg,
                    batch_size=args.batch_size, concurrency=args.concurrency, cache=cache,
                )
            finally:
                _persist_wave_items(items, cache, args.cache, args.debug)

            accepted = sum(1 for i in items if i.final_status == "accepted")
            failed = sum(1 for i in items if i.final_status == "failed")
            logging.info("Wave %s done: accepted=%s failed=%s cache=%s",
                           wave_idx, accepted, failed, len(cache))

    # 无论是否调 API，最后都合并输出（assemble-only 只跑这一段）
    translations, failures = build_translation_map(refs, cache, skip_cfg, args.debug)
    merged = merge_into_dialogues(dialogues, translations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if failures:
        args.failed.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8")
    elif args.failed.exists():
        args.failed.unlink()

    filled = sum(1 for r in refs if (r.get("content") or "").strip() in translations)
    logging.info(
        "Wrote output -> %s | turns_with_content_msa=%s/%s | failed_unique=%s",
        args.output, filled, len(refs), len(failures),
    )
    if failures and not args.assemble_only:
        logging.warning("Some strings failed QA; see %s", args.failed)


if __name__ == "__main__":
    main()
