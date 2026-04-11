# OneShot-ID: Identity-Consistent face Generation & Verification

本项目利用 InstantID 和 SDXL 实现基于单张参考图的身份一致性人脸图像生成，并提供可量化的身份相似度验证。

## 1. 项目结构

```
OneShot-ID/
├── app/                    # CLI 入口
│   ├── run_generate.py     # 仅生成候选图
│   ├── run_validate.py     # 仅校验已有候选目录
│   └── run_pipeline.py     # 端到端：生成 → 校验 → 筛选 → 报表
├── configs/                # 默认配置（设备、模型路径、阈值与轮次）
├── outputs/                # 运行输出（见下文目录结构）
├── src/                    # 核心库
│   ├── generation/         # InstantID 生成
│   ├── prompts/            # Prompt 变体库
│   ├── validation/         # InsightFace 校验
│   ├── reporting/          # CSV / JSON / Markdown 报表
│   ├── pipeline.py         # 流水线编排
│   └── config.py           # 配置加载
├── docs/
│   ├── project_documentation_zh.md   # 中文系统说明（技术与运维）
│   └── project_documentation_en.md   # 英文系统说明（同上）
├── requirements.txt
└── tests/
```

更完整的技术说明见 [docs/project_documentation_zh.md](docs/project_documentation_zh.md) 与 [docs/project_documentation_en.md](docs/project_documentation_en.md)。

## 2. 环境设置（本地）

```bash
conda activate py312_cu121
pip install -r requirements.txt
```

默认在配置中将 GPU 设为 `cuda:1`（可在 `configs/default.yaml` 的 `runtime.device` 中修改）。

## 3. 模型路径

请确保权重位于配置中的路径（可按机器调整 `configs/default.yaml`）：

- **InsightFace**：`models.insightface_dir`（需 `antelopev2` 等）
- **InstantID**：`models.instantid_dir`（ControlNet + `ip-adapter.bin` 等）
- **SDXL**：`models.sdxl_path`（本项目示例为 RealVisXL V4.0 Lightning）

## 4. 配置文件说明（`configs/default.yaml`）

| 区块 | 作用 |
| --- | --- |
| `runtime` | `device`（如 `cuda:1`）、`seed` |
| `models` | `insightface_dir`、`instantid_dir`、`sdxl_path` |
| `generation` | `min_keep` / `max_keep`（最终保留 8–12 张）、`candidates_per_round`、`max_rounds`（不足时重试轮数） |
| `validation` | `similarity_threshold`（余弦相似度阈值，默认 0.45） |
| `output` | `base_dir`（通常为 `outputs`，其下为 `runs/<run_name>/`） |

## 5. 命令行工具

### 5.1 端到端（推荐）

```bash
python app/run_pipeline.py --input data/my_face.jpg --output_name test_run_01
python app/run_pipeline.py --input data/my_face.jpg --config configs/default.yaml --seed 123
```

### 5.2 仅生成候选图

```bash
python app/run_generate.py --input data/my_face.jpg --config configs/default.yaml --run-name my_gen --seed 42
```

### 5.3 仅校验候选目录

```bash
python app/run_validate.py \
  --reference data/my_face.jpg \
  --candidate-dir outputs/runs/my_gen/candidates \
  --config configs/default.yaml \
  --report-dir outputs/runs/my_gen/reports
```

校验阶段会写入 `validation_results.csv`、`validation_summary.json` 与 **`validation_report.md`**（Markdown 人类可读汇总）。

## 6. 输出目录结构

单次运行在 `outputs/runs/<run_name>/` 下：

```
outputs/runs/<run_name>/
├── candidates/              # 各轮生成的候选图
│   └── candidate_manifest.jsonl
├── kept/                      # 最终保留（通过阈值且入选 8–12 张）
├── rejected/                  # 未通过阈值，或通过但超出 max_keep 的副本
└── reports/
    ├── validation_results.csv
    ├── validation_summary.json
    └── validation_report.md
```

## 7. 最终保留策略（8–12 张）

- 目标：至少 `min_keep`（默认 8）、至多 `max_keep`（默认 12）张 **通过** 相似度阈值的图。
- 每轮最多生成 `candidates_per_round`（默认 12）个 prompt 变体；若已通过数仍低于 `min_keep`，会用 **可重试** 的 prompt 继续生成，最多 `max_rounds`（默认 3）轮。
- 若通过数 **超过** `max_keep`：按相似度从高到低保留前 `max_keep` 张，其余通过样本复制到 `rejected/`。
- 流水线成功退出表示最终 `kept/` 数量 ≥ `min_keep`；否则为失败（需换参考图、调阈值或增轮次等）。

## 8. 失败与多脸行为（摘要）

- **参考图无人脸**：`run_pipeline` / `run_generate` / `run_validate` 在参考端无法提取人脸时会报错退出。
- **候选图无人脸**：`failed_no_face`，图会进入 `rejected/`。
- **相似度低于阈值**：`failed_low_similarity`；多脸且低于阈值时为 `failed_multi_face_low_similarity`。
- **多脸**：检测上取 **面积最大** 的一张脸做嵌入与相似度；多脸会作为异常信息参与状态与报表（详见中文/英文系统文档）。

## 9. 运行测试

```bash
pytest tests/
```

## 10. 报表与文档

- **CSV**：逐图 `status`、`failure_reason`、`similarity`、`face_count` 等。
- **JSON**：批次汇总（通过/失败数、相似度统计、`failure_reasons` 分布）。
- **Markdown（`validation_report.md`）**：汇总表、失败原因统计、逐图表格，便于直接阅读或归档。

详细字段与流程说明见 [docs/project_documentation_zh.md](docs/project_documentation_zh.md) 与 [docs/project_documentation_en.md](docs/project_documentation_en.md)。
