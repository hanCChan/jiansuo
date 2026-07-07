# Group 1 翻译结果（供 GPT 分析）

第 1 组（`q_000001`）印尼语 → MSA 翻译流水线输出，2026-07-06 运行。

## 推荐阅读顺序

1. **`reeval_current_rules_summary.json`** — 用**当前 main 规则**对同一批译文重评（无需重调 Kimi）
2. **`gpt_analysis_summary.json`** — 旧规则下的原始统计（56.4% 通过率，供对比）
3. **`cluster_retrieval_intent_eval_msa_partial.json`** — **可用于 embedding 实验**的 partial eval
4. **`failed_after_reeval.jsonl`** — 重评后仍失败的 188 条
5. **`cluster_retrieval_intent_eval_msa_debug.jsonl`** — 完整 debug（~3MB）

## 关键数字对比

| 指标 | 旧规则（首次 run） | 当前规则重评（同批译文） |
|------|-------------------|-------------------------|
| 通过率 | 56.4% (2794/4956) | **96.2% (4768/4956)** |
| strict eval | 不可用 `[]` | 仍不可用（188 条失败） |
| partial eval | 无 | **可用**（query+positive 通过，4766 negatives） |

结论：**大量失败是 QA 误杀，不是 Kimi 整体翻错。**

## 文件说明

| 文件 | 用途 |
|------|------|
| `reeval_current_rules_summary.json` | 重评汇总：错误分布、partial 是否可用 |
| `cluster_retrieval_intent_eval_msa_partial.json` | partial eval JSON（embedding 先用这个） |
| `failed_after_reeval.jsonl` | 重评后仍失败 item，供定点 repair |
| `qa_report.json` | 首次 run 的 group QA（旧规则） |
| `cluster_retrieval_intent_eval_msa_debug.jsonl` | 完整 idn/msa/qa |
| `cluster_retrieval_intent_eval_msa.json` | strict eval（空 `[]`） |

## 重评命令（本地复现）

```bash
cd translate
python3 scripts/reevaluate_debug_with_current_qa.py \
  --debug ../results/group1_run/cluster_retrieval_intent_eval_msa_debug.jsonl \
  --summary-out ../results/group1_run/reeval_current_rules_summary.json \
  --partial-out ../results/group1_run/cluster_retrieval_intent_eval_msa_partial.json \
  --failed-out ../results/group1_run/failed_after_reeval.jsonl
```

## 给 GPT 的分析 prompt 示例

```
请对比 results/group1_run/ 下：
- gpt_analysis_summary.json（旧规则 56%）
- reeval_current_rules_summary.json（新规则 96%）
说明哪些失败是规则误杀、哪些是真翻译问题，并读 failed_after_reeval.jsonl 给出 repair 建议。
```
