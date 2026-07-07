# Group 1 翻译结果（供 GPT / embedding 分析）

## 推荐用于 embedding 的文件

**`cluster_retrieval_intent_eval_msa_repaired_partial.json`**（4935 negatives，query+positive 通过）

备选：`cluster_retrieval_intent_eval_msa_partial_v2.json`（4786 negatives，规则修复后、未做 Kimi repair）

## 流水线阶段与通过率

| 阶段 | 通过 | 说明 |
|------|------|------|
| 旧规则首次 run | 56.4% | QA 误杀为主 |
| 当前规则重评（无重翻） | **96.2%** | 同批译文 |
| 规则增强后重评 | **96.6%** | ATM/KTP/双极性等 |
| 定点 repair 后 | **+149 negatives** | 160 条送修，169 接受，19 丢弃 |

## 定点 repair 结果（`repair_summary.json`）

```json
{
  "failed_before": 188,
  "rule_false_positive": 28,
  "sent_to_repair": 160,
  "repair_accepted": 169,
  "repair_failed": 19,
  "dropped": 19
}
```

- **P0**：160 条（占位符/产品名丢失/核心实体）
- **RULE**：28 条（ATM/KTP 合理阿语化、双极性同句）→ 重跑 QA 自动通过
- **P2_DROP**：repair 仍失败的 19 条 negative 丢弃

## 文件索引

| 文件 | 用途 |
|------|------|
| `cluster_retrieval_intent_eval_msa_repaired_partial.json` | **embedding 主输入** |
| `failed_after_reeval_triage.jsonl` | 188 条 triage（P0/RULE） |
| `repaired_items.jsonl` | repair 成功项 |
| `dropped_items.jsonl` | 丢弃项 |
| `repair_summary.json` / `merge_summary.json` | 汇总 |
| `cluster_retrieval_intent_eval_msa_debug.jsonl` | 完整 debug |

## 复现命令

```bash
cd translate
python3 scripts/run_targeted_repair.py \
  --failed ../results/group1_run/failed_after_reeval.jsonl \
  --partial ../results/group1_run/cluster_retrieval_intent_eval_msa_partial_v2.json \
  --output-dir ../results/group1_run \
  --concurrency 8
```
