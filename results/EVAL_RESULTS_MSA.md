# MSA Intent Retrieval Benchmark — 结果汇总与配置核对

## 评测设定

| 项 | 值 |
|---|---|
| 数据集 | `results/full_run/cluster_retrieval_intent_eval_msa_full.json` |
| Query 数 | 37（印尼语 FAQ query 机器译为 MSA 阿拉伯语） |
| 候选规模 | 每组 1 positive + 4954 negative（全库检索） |
| 指标 | recall@1 / @3 / @5（正样本排名） |
| 任务 | Query（阿拉伯语用户问法）→ 检索匹配 FAQ Question |

---

## 总榜（按 recall@1 排序）

| 排名 | 模型 | 模式 | recall@1 | recall@3 | recall@5 | 耗时(s) | 备注 |
|:---:|------|------|---------:|---------:|---------:|--------:|------|
| 1 | **bge_m3** | hybrid | **0.6486** | 0.8108 | 0.8108 | 174 | 当前最佳 |
| 2 | bge_m3 | dense | 0.5946 | 0.7568 | 0.8378 | 67–94 | |
| 2 | bge_m3 | colbert | 0.5946 | 0.7838 | 0.8378 | 115 | |
| 2 | bge_m3 | dense+sparse | 0.5946 | 0.7838 | 0.8378 | 175 | |
| 2 | arabic_english_bge_m3 | dense | 0.5946 | 0.7568 | 0.8378 | 84 | 与 bge_m3 持平 |
| 6 | multilingual_e5_large_instruct | dense | 0.5135 | 0.7838 | 0.8378 | 92 | |
| 6 | gte_multilingual_base | dense | 0.5135 | 0.6486 | 0.7297 | 42 | |
| 6 | qwen3_embedding_4b | dense | 0.5135 | 0.7838 | 0.8378 | 342 | |
| 9 | snowflake_arctic_l_v2 | dense | 0.4865 | 0.7838 | 0.8378 | 83 | |
| 9 | gate_arabert_v1 | dense | 0.4865 | 0.7027 | 0.8918 | 67 | recall@5 高 |
| 11 | embeddinggemma_300m | dense | 0.4595 | 0.7297 | 0.7568 | 152 | |
| 12 | qwen3_embedding_0_6b | dense | 0.4324 | 0.7027 | 0.8378 | 119 | **MTEB SOTA 系列但表现偏低** |
| 13 | arabic_triplet_matryoshka_v2 | dense | 0.4054 | 0.6486 | 0.7568 | 66 | 阿拉伯专用模型最低 |
| 14 | bge_m3 / gte | sparse | 0.3243 | ~0.54–0.57 | ~0.59–0.62 | 60–70 | 纯 sparse 不适合本任务 |
| 15 | gte_multilingual_base | hybrid | 0.3514 | 0.5405 | 0.5946 | 99 | hybrid 反而低于 dense |

原始 JSON：`results/eval_run/eval_summary.json`

---

## 各模型配置核对（代码 vs 官网）

| 模型 | 我们的配置 | 官网推荐 | 是否对齐 | 说明 |
|------|-----------|---------|:------:|------|
| **bge_m3** (dense) | `backend=dense`, 无 query 前缀, fp16 | FlagEmbedding dense encode，无需 instruct | ✅ | 8模型跑法正确；multimode 用官方 FlagEmbedding 后端 |
| **bge_m3** (hybrid) | `hybrid_weights: [0.4, 0.2, 0.4]` | 官方示例常用等权或可调 | ⚠️ | 权重未 grid search；hybrid 已是最佳说明配置基本合理 |
| **multilingual_e5** | `Instruct: Given a user question in Modern Standard Arabic, retrieve the matching FAQ question\nQuery: {q}` | E5 要求 query 加 task instruct，document 不加 | ✅ | **配置最到位**，针对阿拉伯语 FAQ 定制了 instruct |
| **snowflake_arctic** | `query: {q}`，document 无前缀 | 官方要求 query 加 `query:` 前缀 | ✅ | |
| **embeddinggemma** | query: `task: search result \| query: {q}`；doc: `encode_document()` | 官方 Retrieval prompt 为 `task: search result`；doc 为 `title: none \| text:` | ✅ | 使用了模型内置 encode_query/encode_document |
| **qwen3 0.6B / 4B** | `prompt_name="query"` → 默认 *"Given a **web search** query, retrieve relevant **passages**..."* | 官方明确建议 **按任务自定义 instruct**，多语言场景建议写英文 instruct | ❌ | **未按任务定制**；FAQ question-question 检索与 web passage 检索语义不同 |
| **gte** (hybrid) | `dense=1.0, sparse=0.3` | 官方 `compute_scores` 默认 `sparse_weight=0.1` | ⚠️ | sparse 权重偏高，且 sparse 本身弱，导致 hybrid 低于 dense |
| **gte** (dense) | GTEEmbeddidng, fp16, max_length=8192 | 官方脚本一致 | ✅ | |
| **gate_arabert / arabic_triplet** | plain encode，无 instruct | 训练目标为通用阿拉伯语句义相似，非 FAQ 检索 | ⚠️ | 无官方 retrieval instruct；任务形式不匹配 |
| **arabic_english_bge_m3** | plain dense | 阿英混合微调版 BGE-M3 | ✅ | 与通用 bge_m3 完全相同，说明阿英微调对此 MSA FAQ 任务无额外收益 |

配置实现见：`pingce_org/src/embedding_backends.py`  
配置文件：`pingce_org/config.yaml`, `config_multimode.yaml`, `config_gte.yaml`, `config_qwen3_4b.yaml`

---

## 为什么「SOTA 模型」反而效果差？（分析）

### 1. MTEB 排名 ≠ 本任务表现（最主要）

- **Qwen3-Embedding** 在 MTEB multilingual 榜单靠前，但本任务是：
  - 极窄领域（银行 FAQ）
  - 短文本 question-to-question 匹配
  - 数据经 IDN→MSA 机器翻译，存在噪声
- MTEB 测的是 passage retrieval、分类、聚类等混合任务，和「4954 负样本中找 1 个 FAQ 问句」分布差异很大。

### 2. Instruct / Prompt 未按官网建议定制（Qwen3 明显）

- Qwen3 官方 README：
  > *"using instructions typically yields an improvement of 1% to 5%... create tailored instructions specific to their tasks"*
- 我们用的是默认 **web search → passages**，而 E5 已定制为 **MSA FAQ question retrieval**。
- 这解释了：**E5 配置更贴合任务，Qwen3-0.6B 虽为 SOTA 系列但 recall@1 仅 0.43**；Qwen3-4B 靠模型容量拉到 0.51，仍未超过 BGE-M3。

### 3. 阿拉伯语「专用」模型不一定适合 FAQ 检索

- `arabic_triplet_matryoshka_v2`（0.40）和 `gate_arabert_v1`（0.49）低于 BGE-M3。
- 这些模型优化的是通用阿拉伯语句义相似/三元组，而候选池是 4954 条高度同质化的 FAQ 问句，需要细粒度区分能力。
- BGE-M3 的多语言检索预训练 + hybrid 融合更适合这种「大海捞针」。

### 4. Sparse / Hybrid 模式任务不适配

- 纯 sparse（BGE/GTE 均 0.32）说明：**关键词匹配对阿拉伯语 FAQ paraphrase 帮助有限**。
- GTE hybrid（0.35）< GTE dense（0.51）：sparse 权重 0.3 高于官方默认 0.1，放大了弱 sparse 信号的负面影响。

### 5. 评测协议本身偏难

- 全库 4955 候选，非 sampled negatives。
- 许多模型 recall@3/@5 高（0.78–0.84）但 recall@1 低，说明能排进前列但 Top-1 区分不够。

### 6. 不是「没跑对」的情况

以下模型经核对 **已按官网设置**：
- E5 instruct ✅
- Snowflake query prefix ✅
- EmbeddingGemma encode_query/document ✅
- BGE-M3 multimode FlagEmbedding 后端 ✅
- GTE dense 编码路径 ✅

**明确未按官网最佳实践的是 Qwen3 instruct 未定制**；GTE hybrid 权重偏离默认。

---

## 建议补跑实验（优先级）

1. **Qwen3-4B 定制 instruct**（预期 +1~5%）：
   ```
   Instruct: Given a user question in Modern Standard Arabic, retrieve the matching FAQ question
   Query: {query}
   ```
2. **GTE hybrid 调权重**：`sparse_weight=0.1`（官方默认）或只报 dense。
3. **BGE-M3 + Qwen3 reranker** 二阶段（官方推荐 pipeline）。
4. 等 `qa_msa.json` 翻译完成后跑 cluster 第二种评测。

---

## 文件索引

| 文件 | 内容 |
|------|------|
| `results/eval_run/eval_summary.json` | 所有模型汇总指标 |
| `results/eval_run/intent_retrieval_eval_msa_gte.json` | GTE 三模式详情 |
| `results/eval_run/intent_retrieval_eval_msa_qwen3_4b.json` | Qwen3-4B 详情 |
| `pingce_org/output/reports/intent_retrieval_eval_msa.json` | 8 模型 dense |
| `pingce_org/output/reports/intent_retrieval_eval_msa_multimode.json` | BGE-M3 五模式 |

---

## 给 GPT 的分析提示（可直接粘贴）

```
以下是印尼语银行 FAQ 检索评测结果（query 已译为 MSA 阿拉伯语，37 query × 4955 全库候选）。
请分析：
1. 为什么 MTEB SOTA 的 Qwen3-Embedding-0.6B (recall@1=0.43) 低于 BGE-M3 hybrid (0.65)？
2. 配置核对显示 Qwen3 用了默认 web-search instruct 而非 FAQ retrieval instruct，E5 则定制了阿拉伯语 instruct——这是否是主要原因？
3. 阿拉伯语专用模型 arabic_triplet (0.40) 为何最差？
4. 提出具体的 prompt 修改和重跑建议。
```
