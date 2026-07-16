# XXX 语义补全 — test 集全量结果与错误分析

运行时间：2026-07-15 18:07–18:23（约 16 分钟）  
输入：`dh/dialogue_20260615_BCA_clean_test_msa.json`  
输出：`dh/dialogue_20260615_BCA_clean_test_msa_xxx_filled.json`

## 总体统计

| 指标 | 数值 |
|------|------|
| 含 XXX 的 turn | 849 |
| 局部簇（API 调用次数） | 493 |
| 写入 `content_msa_filled` | 789（93.0%） |
| accepted | 749 |
| warning | 40 |
| failed | 58 |
| 补全后仍剩 XXX | 0 |
| cache 簇数 | 492（缺 `739_c0`） |

报告文件：`dialogue_test_xxx_fill_report.json`  
日志：`logs/xxx_fill_full_20260715_180707.log`

## 簇划分说明

- 在**每个对话内部**，把 `content_msa` 含 XXX 的 turn 按 turn 编号排序
- 相邻两个 XXX turn 编号差 ≤ 3（`gap_max=3`）则合并为一簇
- 1 簇 = 1 次 Kimi API；不跨对话合并
- 全量 493 簇，平均约 1.72 turn/簇

## QA 规则摘要（`xxx_fill_qa.py`）

**failed（有 error）：**

- `empty_filled` / `xxx_remaining`
- `non_xxx_preservation_low`：骨架相似度 < 0.55
- `pii_shape_long_digit_sequence`：非 PII hint 却出现 ≥10 位连续数字
- `pii_generic_phrase_instead_of_digits`：PII 类用泛化短语而非合成数字
- `pii_name_has_long_digits` / `pii_shape_email_shape` / `missing_entity:*`

**warning：** 骨架相似度 0.55–0.72（仍写入 `content_msa_filled`）

**骨架相似度：** 去掉 XXX 后比较原文与补全；原文 ≥3 个 XXX 时额外去掉补全中的数字再比。

## 失败原因分布

| 类型 | turn 数 | 说明 |
|------|---------|------|
| `non_xxx_preservation_low` | 50 | 短句（1–2 个 XXX）填数字后骨架差异大 |
| `pii_shape_long_digit_sequence` | 8 | hint 为 unknown/date 等，却填长数字 |
| API 漏 turn（`739_c0`） | 2 | 模型未返回 turn 3、5 |

### 真实样例

**preservation 失败 — dialogue 1012 turn 84**

- 原文：`XXX XXX أريد`
- 补全：`453 843 أريد`
- ratio = 0.50，hint = `pii_generic`

**preservation 失败 — dialogue 1007 turn 92**

- 原文：`لـ XXX`
- 补全：`لـ 501`
- ratio = 0.50

**长数字规则失败 — dialogue 323 turn 10**

- 原文：`…هل يمكنك إخباري XXXXXXX`
- 补全：`…3273781310`
- hint = `unknown`，ratio 很高仍 failed

**API 失败 — dialogue 739 cluster `739_c0`**

- 日志：`missing filled turns: [3, 5]`
- turn 9 在 `739_c1` 中 accepted

## 同簇部分成功示例

`1012_c7`：turn 82 accepted（长句多 XXX），turn 84 failed（短句 `XXX XXX أريد`）。

## 后续建议

1. 放宽 1–2 个 XXX 短 turn 的 preservation 规则（去掉补全数字后再比，或降低阈值）
2. 改进 `unknown` 但语境像账号/手机的 hint 分类
3. 重跑 `739_c0` 及 58 个 failed turn

## 复现命令

```bash
cd translate/buquan
./run_fill_dialogue_test.sh full --resume
```
