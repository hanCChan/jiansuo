# Group 1 翻译结果（供 GPT 分析）

第 1 组（`q_000001`）印尼语 → MSA 翻译流水线输出，2026-07-06 运行。

## 文件说明

| 文件 | 用途 |
|------|------|
| `gpt_analysis_summary.json` | **先看这个**：通过率、错误分布、失败/成功样例 |
| `qa_report.json` | group 级 QA 汇总，含全部 `failed_items` id 列表 |
| `cluster_retrieval_intent_eval_msa_debug.jsonl` | 完整 debug（单行 JSON，~3MB，含每条 idn/msa/qa） |
| `cluster_retrieval_intent_eval_msa.json` | 主 eval 输出（本 run 为空 `[]`，因 eval_ready=False） |

## 关键结论（摘要）

- 4956 条 item，**2794 通过 / 2162 失败**（56.4%）
- 主输出为空：设计为 **全组 QA 全过** 才写入 eval JSON
- 主要失败：`latin_leakage`、`arabic_ratio_too_low`、实体/术语/极性规则误杀

## 给 GPT 的分析建议

1. 读 `gpt_analysis_summary.json` 了解整体
2. 按 `top_hard_errors` 分类看 `failure_samples_by_error_type`
3. 需要逐条细节时解析 `debug.jsonl` 里的 `positive[]` / `negative[]`
4. 文字版根因分析见：`translate/docs/GROUP1_RUN_ANALYSIS.md`

## debug.jsonl 结构（单行）

```json
{
  "query_id": "q_000001",
  "eval_ready": false,
  "simple": { "query", "positive", "negative" },
  "debug": {
    "query_idn", "query_msa",
    "positive": [{ "id", "idn", "msa", "qa", "final_status" }],
    "negative": [...],
    "qa": { "failed_items": [...] }
  }
}
```
