# V1 数据集说明

- Agent：20 个确定性轨迹 Case（Gold 6、Boundary 4、Failure 6、Adversarial 4）。
- RAG：12 份企业政策、60 个 Chunk、60 个 Query（30/12/9/9）。
- Memory：50 个多轮场景（20/12/8/10）。
- Production Replay：V1 没有脱敏线上数据，不声明生产指标。
- 公共 Benchmark：仅映射 BFCL、BEIR/MIRACL、LongMemEval 的任务或指标概念，不声明公共榜单成绩。

`regressions.jsonl` 在首次人工晋升前可以不存在或为空。评测失败不会自动进入回归集。
