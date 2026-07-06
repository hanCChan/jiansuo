# 第 1 组翻译运行分析报告

运行时间：2026-07-06 15:58 – 17:53（约 **1 小时 55 分**）

命令：

```bash
python3 run_pipeline.py --max-groups 1 --batch-size 40 --concurrency 8 --enable-thinking
```

## 1. 是否「翻译完了」？

**进程已跑完**，但 **未产出可用于 embedding 评估的主数据**。

| 输出文件 | 大小 | 说明 |
|----------|------|------|
| `cluster_retrieval_intent_eval_msa.json` | **2 字节 `[]`** | 空：该组 `eval_ready=False` |
| `cluster_retrieval_intent_eval_msa_debug.jsonl` | ~3.0 MB | 含完整 debug / QA 信息 |
| `qa_report.json` | ~59 KB | group 级 QA 汇总 |

设计原则：**仅当组内全部 item QA 通过**，才写入主输出。本组有 2162 条失败，故主输出为空。

## 2. 量化结果

| 指标 | 数值 |
|------|------|
| 总 item 数 | 4956 |
| QA 通过 | **2794（56.4%）** |
| QA 失败 | **2162（43.6%）** |
| query 状态 | **失败** |
| positive 状态 | **通过** |
| negative 通过 | 2793 / 4954 |

重试：跑满 **3 轮**（translate → QA → repair → 再 translate …）。

## 3. 失败原因分布（hard QA）

| 错误类型 | 次数 | 占比 | 说明 |
|----------|------|------|------|
| `latin_leakage` | 3285* | 66% | 拉丁字母未在白名单（*含一条多错误） |
| `arabic_ratio_too_low` | 652 | 13% | 阿拉伯字符占比 < 55% |
| `entity_drift` | 119 | 2% | 实体替换规则误杀 |
| `action_polarity_error` | 82 | 2% | 极性词 substring 误匹配 |
| `term_confusion` | 63 | 1% | 术语期望未满足 |
| `indonesian_residue` | 10 | <1% | 印尼语残留 |

**仅因 `latin_leakage` 失败**：1383 条  
**仅因 `arabic_ratio` 失败**：34 条  
**两者兼有**：562 条  

### 典型失败样例

1. **`latin_leakage`**：`A Card Flazz` → 译文保留 `Card`/`Flazz`，未加入白名单  
   - 源：`Di mana beli A Card Flazz, apakah di cabang BCA?`  
   - 译：`أين يمكن شراء بطاقة A Card Flazz، هل في فرع BCA؟`

2. **`arabic_ratio_too_low`**：产品名拉丁字母拉低阿拉伯语占比（0.521）

3. **`entity_drift`**：`KlikBCA Individu` 译文中含 `KlikBCA` 子串被误判为 drift

4. **query 失败**（3 条错误叠加）：
   - `entity_drift:BCA ID->myBCA`：源句同时提到 BCA ID 与 myBCA，译文合法却被拦
   - `term_confusion:unblock` / `action_polarity_error:membuka_blokir`：「فك حظر」中 `حظر` 触发 `memblokir` 的 forbidden 子串匹配

### 译文质量（通过样本）

positive 译文质量可接受：

- 源：`BCA ID terblokir salah password, bagaimana?`
- 译：`BCA ID محظور بسبب كلمة مرور خاطئة، ماذا أفعل؟`

## 4. 发现的系统性问题

### P0 — 主输出为空，2 小时无 eval 可用数据

QA 采用 **全有或全无** gate；43.6% 失败导致整组丢弃，但 debug 里已有 2794 条可用译文。

### P0 — 未做全局去重，效率极低

全库 183372 条 occurrence，**仅 4992 条唯一印尼语句**。  
第 1 组跑完已覆盖 **4956 条唯一句**；剩余 36 组只需各补翻 **1 条 query**。

当前 pipeline 若跑满 37 组，会重复翻译同一 FAQ **~37 倍**。

### P1 — hard QA 规则过严 / 有 bug

1. **拉丁白名单不全**：`Flazz`、`Card`、`Online`、`ATS` 等产品名未列入  
2. **`entity_drift` 误杀**：源句含多个实体时，译文同时出现不应算 drift  
3. **`action_polarity` 子串匹配**：`فك الحظر` 中的 `حظر` 误触发 `memblokir` forbidden  
4. **`arabic_ratio` 未排除白名单拉丁字符**：产品名拉低比例

### P1 — repair 串行，极慢

`repair_single` 逐条 API（~1 次/秒），第 1 轮 repair 约 **40 分钟**；batch 翻译仅 ~12 分钟。

### P2 — thinking 模式收益不明

`--enable-thinking` 已开启，但未见明显 reasoning 分离；`enable_thinking` 对此 vLLM 部署效果有限。

### P2 — 缺少 QA 进度日志

失败条数、错误分布需跑完后解析 debug，无法实时感知。

## 5. 建议下一步

用修复后的 hard QA 规则对同一批 debug 译文重新评估，通过率可从 **56.4% → ~79%**（仍约 1046 条失败，多为未白名单化的产品词 `latin_leakage`）。

1. **启用全局 translation cache**（按 `source_idn` 去重，4992 条唯一句）  
2. **修复 hard QA**（白名单、entity_drift、action 子串匹配、arabic_ratio）  
3. **repair 并发化**  
4. **从 debug 导出 partial eval** 或放宽 `--export-on-partial` 供 interim 评估  
5. **重跑第 1 组**（关闭 thinking 或修复 QA 后），预计 **10–20 分钟**（有 cache + 并发 repair）

## 6. 从 debug 恢复 partial 数据

```bash
cd translate
python3 scripts/export_partial_from_debug.py \
  --debug output/cluster_retrieval_intent_eval_msa_debug.jsonl \
  --output output/cluster_retrieval_intent_eval_msa_partial.json
```

仅导出 `final_status=accepted` 的 item（本组约 2794 条可用译文）。
