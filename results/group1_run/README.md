# Group 1 翻译结果（供 GPT / embedding 分析）

## Embedding 主输入

**`cluster_retrieval_intent_eval_msa_strict.json`** — strict 100%，4954 negatives，可直接用于 embedding baseline。

与下列文件内容一致（已同步）：

- `cluster_retrieval_intent_eval_msa_repaired_partial.json`

**不要使用** 旧版空 strict 输出 `translate/output/cluster_retrieval_intent_eval_msa.json` 的历史快照；当前第 1 组最终产物见本目录。

> strict eval 要求 query / positive / 全部 negative 均通过 hard QA。经规则修复 + 定点 repair 后已达 **4956/4956（100%）**。

## 流水线阶段与通过率

| 阶段 | 通过率 | 说明 |
|------|--------|------|
| 1. 旧规则首次 run | **56.38%** | QA 误杀严重（latin_leakage、entity_drift 等），非 Kimi 整体翻差 |
| 2. 当前规则重评（无重翻） | **96.21%** | 同批译文，partial ready |
| 3. 定点 repair + 规则收尾 | **100%** | 最后 6 条失败项修复；双极性/实体别名等规则补齐 |

## 定点 repair 口径（`repair_summary.json`）

```json
{
  "failed_before": 188,
  "rule_false_positive": 28,
  "sent_to_repair": 160,
  "repair_accepted": 169,
  "repair_failed": 19,
  "dropped": 19,
  "by_repair_status": {
    "accepted": 169,
    "failed": 11,
    "rule_still_fail": 8
  }
}
```

准确表述：

> 188 条失败经 triage 后，**160 条**送 Kimi repair，**28 条**走规则复核（其中 8 条 `rule_still_fail`）；repair 阶段 accepted **169**、dropped **19**。后续规则增强 + round2 repair 将 strict 从 99.88% 推至 **100%**（剩余 6 条边缘 negative 已修复，不再丢弃）。

## 最终导出（`strict_export_summary.json` / `final_export_summary.json`）

```json
{
  "total_items": 4956,
  "passed": 4956,
  "failed": 0,
  "pass_rate_pct": 100.0,
  "strict_eval_ready": true,
  "negative_in_output": 4954,
  "negative_expected": 4954
}
```

## 文件索引

| 文件 | 用途 |
|------|------|
| `cluster_retrieval_intent_eval_msa_strict.json` | **embedding 主输入（100%）** |
| `cluster_retrieval_intent_eval_msa_repaired_partial.json` | 与 strict 同步 |
| `strict_export_summary.json` | strict 导出统计 |
| `sanity_check_report.json` | embedding 前结构检查 |
| `repaired_items_all.jsonl` | 全部 accepted repair（含 round2） |
| `repair_summary.json` | 首轮 repair 汇总 |
| `gpt_analysis_summary.json` | 旧规则 56% 分析 |
| `reeval_current_rules_summary.json` | 重评 96.21% |
| `cluster_retrieval_intent_eval_msa_debug.jsonl` | 完整 debug |

## 复现命令

```bash
cd translate

# 定点 repair（首轮）
python3 scripts/run_targeted_repair.py \
  --failed ../results/group1_run/failed_after_reeval.jsonl \
  --partial ../results/group1_run/cluster_retrieval_intent_eval_msa_partial_v2.json \
  --output-dir ../results/group1_run \
  --concurrency 8

# strict 导出（合并 repaired_items_all.jsonl）
python3 scripts/export_strict_from_repaired.py \
  --debug ../results/group1_run/cluster_retrieval_intent_eval_msa_debug.jsonl \
  --repaired ../results/group1_run/repaired_items_all.jsonl \
  --input ../shuju/cluster_retrieval_intent_eval.json \
  --group-index 1 \
  --output-strict ../results/group1_run/cluster_retrieval_intent_eval_msa_strict.json \
  --output-failed ../results/group1_run/strict_still_failed.jsonl \
  --summary-out ../results/group1_run/strict_export_summary.json

# embedding 前检查
python3 scripts/sanity_check_partial.py \
  --input ../results/group1_run/cluster_retrieval_intent_eval_msa_strict.json
```

## 后续 36 组策略

1. 全局去重 candidate（4992 唯一句，97.3% 跨组重复）
2. 已翻译 candidate 走 global cache 命中
3. 每组只补翻新增 query
4. hard QA 用当前规则；query/positive 失败强修
5. negative 失败默认 partial drop，非核心 hard negative 不跑完整 repair
