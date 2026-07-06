# jiansuo

印尼语 BCA 银行 FAQ 检索评测数据：IDN → MSA 翻译流水线，以及后续 embedding 检索评估。

## 目录结构

```text
jiansuo/
  shuju/
    cluster_retrieval_intent_eval.json      # 原始印尼语评测数据
  translate/
    config/                                 # 实体、术语、动作极性等 QA 规则
    idn_msa/                                # 翻译 + QA 流水线代码
    run_pipeline.py                         # 主入口
    output/                                 # 本地生成（不提交 Git）
  examples/
    msa_eval_sample.json                    # 主输出格式样例
    qa_report_sample.json                   # QA 汇总样例
```

## 输入格式

`shuju/cluster_retrieval_intent_eval.json`：

```json
[
  {
    "query": "Indonesian query",
    "positive": ["Indonesian positive"],
    "negative": ["Indonesian negative 1", "Indonesian negative 2"]
  }
]
```

## 输出格式

### 1. 主数据（给 embedding 评估用）

`translate/output/cluster_retrieval_intent_eval_msa.json`

**与原始数据结构完全一致**，只有语言从印尼语变为 MSA：

```json
[
  {
    "query": "MSA query",
    "positive": ["MSA positive"],
    "negative": ["MSA negative 1", "MSA negative 2"]
  }
]
```

样例见：`examples/msa_eval_sample.json`

### 2. Debug 数据（排错 / 质检追溯）

`translate/output/cluster_retrieval_intent_eval_msa_debug.jsonl`

每行一个 query group，包含 `idn`、`msa`、`qa`、`final_status` 等复杂字段。

### 3. QA 汇总

`translate/output/qa_report.json`

group 级通过率、失败项、重试轮次。样例见：`examples/qa_report_sample.json`

> **原则**：复杂 QA 信息不进入 embedding 评估主输入。

## 第 1 组试运行结论（2026-07-06）

详见 [`translate/docs/GROUP1_RUN_ANALYSIS.md`](translate/docs/GROUP1_RUN_ANALYSIS.md)。

- 进程跑完约 2 小时，**主输出为空**（2162/4956 条 QA 失败，`eval_ready=False`）
- 全库仅 **4992 条唯一印尼语句**（97.3% 跨组重复），应使用 `--cache translation_cache.jsonl` 做全局去重补翻
- 主要失败：`latin_leakage`、过严 `arabic_ratio`、`entity_drift` / `action_polarity` 误杀（已在代码中修复）

从 debug 导出 partial 结果：

```bash
python3 translate/scripts/export_partial_from_debug.py \
  --debug translate/output/cluster_retrieval_intent_eval_msa_debug.jsonl \
  --output translate/output/cluster_retrieval_intent_eval_msa_partial.json \
  --cache-out translate/output/translation_cache.jsonl
```

## 运行命令

```bash
cd translate

# 冒烟测试：先跑 1 个 query group
python3 run_pipeline.py --max-groups 1 --batch-size 40 --concurrency 8

# 续跑后续组（复用全局 cache，每组 mostly 只补翻 query）
python3 run_pipeline.py --start-index 1 --max-groups 36 --resume

# 正式跑全量（37 组）
python3 run_pipeline.py --start-index 0 --max-groups 37 --batch-size 20

# 开启完整 QA（更慢）
python3 run_pipeline.py \
  --max-groups 37 \
  --batch-size 20 \
  --enable-semantic-qa \
  --enable-relation-qa \
  --relation-sample-limit 0

# 断点续跑
python3 run_pipeline.py --max-groups 37 --resume
```

默认 Kimi API：

```text
base_url: http://10.16.137.2:8000/v1
model:    Kimi-K2.6-CT-FP8KV
```

## QA 流水线概要

1. 结构冻结 + ID 展开
2. 实体 mask（BCA、myBCA、OTP、PIN 等）
3. Kimi 结构化翻译为 MSA
4. 硬规则 QA：JSON 结构、Arabic ratio、实体/术语/动作极性
5. 可选语义 QA、关系 QA、回译检查
6. 最多 3 轮自动 repair

仅 QA 全部通过的 group 会写入主输出 `cluster_retrieval_intent_eval_msa.json`。

## 模型权重

`model/` 目录存放本地 embedding 模型（约 12GB），已在 `.gitignore` 中忽略，需本地下载。

## 下一步：embedding 评估

读取主输出即可，无需解析 debug 字段：

```python
import json

with open("translate/output/cluster_retrieval_intent_eval_msa.json") as f:
    data = json.load(f)

for row in data:
    query = row["query"]
    positives = row["positive"]
    negatives = row["negative"]
    # encode & compute Recall@K / MRR
```
