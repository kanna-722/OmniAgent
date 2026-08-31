# OmniAgent 七层对比评测

本目录只评估三个精确任务单元，不对“全能 Agent”打一个含义不清的总分：

1. 单次 ReAct 正常收敛或达到递归上限后的受控终止；
2. 固定企业政策语料的 Top-K Chunk 检索、排序、去重和故障降级；
3. 用户事实的近期历史、长期记忆、跨会话召回、更新、隔离与上下文预算。

V1 首次可比基线为 `main@284305d`。该提交只把 `main@69c6ab6` 的模型 API 和 `.env` 配置同步到候选版本使用的服务格式，不包含 ReAct、RAG 或 Memory 优化。候选版本为 `improve@e3fc22b`。

## 七层职责

| 层 | 本框架中的落点 |
| --- | --- |
| 任务定义 | `datasets/v1/manifest.json` 中的三个 task unit |
| 成功标准 | `compare.py` 中的 Suite 独立门禁，Case 输出 `PASS/PARTIAL/FAIL` |
| 数据集 | 20 Agent、60 RAG、50 Memory Case；Gold/Boundary/Failure/Adversarial 四层 |
| 环境执行器 | 两个 detached Worktree、独立子进程、固定 Fake 轨迹和外部超时 |
| 评分器 | 规则断言、Recall/MRR/NDCG、可观察轨迹；不采集隐藏 CoT |
| 报告门禁 | JSONL 原始结果、绝对值/差值、退化 Case、Markdown 上线结论 |
| 反馈闭环 | 去重失败候选经人工复核后用 `promote_failure` 晋升回归集 |

## 数据快照

`datasets/v1/cases.py` 是 Case 生成源，以下文件是提交到 Git 的固定快照：

- `cases.jsonl`：统一 Case Schema；
- `qrels.json`：RAG Query 到相关 Chunk 的等级标注；
- `policies.json`：12 份政策、60 个 Chunk；
- `checksums.sha256`：快照 SHA256。

修改生成源后执行：

```powershell
python -m evals.datasets.v1.build_artifacts
```

执行器会在运行前校验快照，文件与校验和不一致时以退出码 `2` 停止。

## 使用方式

Mock 全量门禁不读取真实 Key，也不会调用外部 API：

```powershell
python -m evals.compare --baseline-ref 284305d --candidate-ref e3fc22b --suite all --mode mock
```

也可以使用 `--suite agent`、`rag` 或 `memory` 单独运行。退出码定义为：

- `0`：所选 Suite 门禁全部通过；
- `1`：评测完成，但至少一个门禁失败；
- `2`：环境、Worktree、数据快照或执行器失败。

`--mode real` 当前会明确返回退出码 `2`。真实适配器尚未完成前，框架拒绝把 Mock 结果标成真实结果。Mock 报告只能说明固定评测集和确定性执行器中的代码行为，不能表述成生产效果。

真实模式启用前必须补齐以下适配器，并保证两个 Ref 使用完全相同的模型参数和独立资源：

- Agent：6 个正常冒烟 Case 的统一模型调用适配器和 Judge 结构化评分；循环 Case 仍只使用 Fake；
- RAG：政策快照导入、独立 Chroma Collection、真实 Embedding/Rerank 以及 10 次预热后 P50/P95；
- Memory：独立 MySQL 数据标识、独立 Chroma Collection、写入/更新/跨 Dialog/隔离清理；
- finally 清理、配额检查、数据污染检查和人工争议复核。

## 产物

每次运行写入 `evals/results/<run_id>/`：

```text
environment.json
main/raw.jsonl
improve/raw.jsonl
summary.json
comparison.md
failures.jsonl
failure_candidates.jsonl
```

`environment.json` 不记录 Secret，只记录提交、运行模式、Python/平台和数据 SHA256。Mock 与未来的 Real 报告必须分开，禁止平均或合并成一个分数。

## 失败反馈闭环

失败记录按 `suite + case_id + failure_code` 生成指纹并去重。人工确认根因和正确期望后才能晋升：

```powershell
python -m evals.promote_failure `
  --source evals/results/<run_id>/failure_candidates.jsonl `
  --fingerprint <sha256> `
  --reviewer <name> `
  --reason "已确认期望和根因"
```

禁止直接把所有失败自动写入 Gold Set。错误标注应通过新数据集版本或显式变更记录修正。
