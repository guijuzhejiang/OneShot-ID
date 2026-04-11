# 单张人脸参考图驱动的身份一致性生成与验证系统说明文档

更完整的英文对照说明见同目录下的 [project_documentation_en.md](project_documentation_en.md)。

## 1. 技术方案选型

为了实现单张参考图驱动的高质量身份一致性生成，本系统选择了 **InstantID** 作为核心架构。

### 1.1 为什么选择 InstantID？

- **非微调方案**：相比于 LoRA 或 DreamBooth 需要多张图且耗时训练，InstantID 支持单图 Zero-shot 身份提取。
- **强身份约束**：InstantID 通过集成的 IP-Adapter 提取人脸特征嵌入（Embedding），并结合 ControlNet 控制脸部关键点，能够极好地保持面部核心特征。
- **兼容性**：与 SDXL 生态深度集成，支持各种精调底座模型（如本项目使用的 RealVisXL）。

### 1.2 后端逻辑

- **生成后端**：使用 `diffusers` 库封装 InstantID Pipeline。通过将参考图的人脸 Embedding 作为全局 Context 注入，实现对身份的显式控制。
- **验证后端**：基于 **InsightFace (AntelopeV2)** 模型。在生成结束后，对每一张图重新进行人脸检测与特征提取，并计算其与原始参考图的余弦相似度（Cosine Similarity）。

## 2. 身份一致性保障机制

系统通过以下“三级约束”确保身份不漂移：

1. **特征提取级**：InstantID 解析参考图获取身份嵌入，作为生成时的核心身份注入点。
2. **结构控制级**：利用人脸关键点（Facial Landmarks）作为 ControlNet 条件，通过控制面部拓扑结构防止五官比例失真。
3. **闭环验证级**：后置验证环节通过可量化的相似度分数筛选掉漂移严重的样本，形成反馈闭环。

## 3. 姿态与表情控制

本系统设计了 `prompt_bank.py` 模块，包含 12 种预置的变化维度：

- **Pose（姿态）**：通过自然语言描述（如 "slight left turn", "head tilted up"）结合基准关键点，引导扩散模型生成多角度结果。
- **Expression（表情）**：利用关键提示词（如 "subtle smile", "serious", "closed eyes"）在保持身份特征的前提下改变情绪表达。
- **Camera（镜头）**：通过 "extreme close-up" 或 "medium shot" 控制景深及构图。

## 4. 身份验证流程

1. **Reference Embedding**：提取原始参考图的身份向量。
2. **Batch Processing**：对候选目录中的图像迭代：
   - 使用 InsightFace 检测人脸；**多脸时取面积最大的人脸** 计算嵌入与相似度，并在结果中反映人脸数量与异常状态。
   - 若未检测到脸，标记为 `failed_no_face`。
   - 若相似度 ≥ 阈值（默认 0.45），标记为 `passed`；否则为 `failed_low_similarity`；多脸且仍低于阈值时为 `failed_multi_face_low_similarity`。
3. **Statistical Analysis**：汇总均值、标准差、最大/最小值等（见 `validation_summary.json`）。
4. **报表输出**：逐图 CSV、汇总 JSON、以及人类可读的 **`validation_report.md`**（标题为 “Validation report”，含汇总与逐图表）。

## 5. 配置项说明（`configs/default.yaml`）

| 配置路径 | 含义 |
| --- | --- |
| `runtime.device` | 推理设备，默认 **`cuda:1`** |
| `runtime.seed` | 随机种子基准（流水线多轮会在此基础上偏移） |
| `models.insightface_dir` / `instantid_dir` / `sdxl_path` | 三类权重的本地路径 |
| `generation.min_keep` | 最终至少保留几张通过阈值的图（默认 8） |
| `generation.max_keep` | 最终至多保留几张（默认 12）；超出则按相似度截断 |
| `generation.candidates_per_round` | 每轮最多生成的 prompt 数（默认 12） |
| `generation.max_rounds` | 不足 `min_keep` 时最多追加几轮生成（默认 3） |
| `validation.similarity_threshold` | 余弦相似度阈值（默认 0.45） |
| `output.base_dir` | 输出根目录（默认 `outputs`，其下为 `runs/<run_name>/`） |

## 6. CLI 使用方法

三类入口脚本均在 `app/` 下，均可通过 `--config` 指定 YAML。

### 6.1 端到端：`run_pipeline`

```bash
python app/run_pipeline.py --input <参考图路径> [--output_name <运行名>] [--seed <整数>] [--config configs/default.yaml]
```

完成：参考图分析 → 多轮生成与校验 → 按策略拷贝到 `kept/` / `rejected/` → 写入 `reports/` 下 **`validation_report.md`** 等同名报表文件。

### 6.2 仅生成：`run_generate`

```bash
python app/run_generate.py --input <参考图> [--config ...] [--run-name <名>] [--seed <整数>]
```

仅填充 `outputs/runs/<run_name>/candidates/` 与 `candidate_manifest.jsonl`，不执行流水线筛选与 `kept/` 逻辑。

### 6.3 仅校验：`run_validate`

```bash
python app/run_validate.py \
  --reference <参考图> \
  --candidate-dir <候选目录> \
  [--config ...] \
  [--report-dir <目录>] \
  [--run-name <报表标题>]
```

默认报表目录为 `<candidate-dir>/../reports/`。输出 **`validation_results.csv`**、**`validation_summary.json`**、**`validation_report.md`**。

## 7. 输出目录结构

```
outputs/runs/<run_name>/
├── candidates/
│   └── candidate_manifest.jsonl
├── kept/                 # 最终保留（8–12 张策略内）
├── rejected/             # 未通过、或通过但超出 max_keep 的副本
└── reports/
    ├── validation_results.csv
    ├── validation_summary.json
    └── validation_report.md
```

## 8. 最终保留策略（8–12 张与重试）

- 系统在 **prompt_id 维度** 上维护每变体最新一次校验结果；目标是 **不同 prompt 下** 至少 `min_keep` 个 `passed`。
- 若当前 `passed` 数 < `min_keep`：使用 **可重试** prompt 集合继续生成下一批（每轮数量受 `candidates_per_round` 限制），最多 `max_rounds` 轮。
- 全部轮次结束后，对 **所有 `passed` 结果** 做最终挑选：若数量 > `max_keep`，按 **相似度降序**（相同时可按 `prompt_id`）取前 `max_keep` 张放入 `kept/`，其余 `passed` 复制到 `rejected/`；未通过阈值的图像也在 `rejected/`。
- 若最终 `kept` 仍少于 `min_keep`，`run_pipeline` 以失败状态结束（退出码非 0）。

## 9. 多脸处理规则

- 验证时对每张候选图做人脸检测；若 **多于一张脸**，选取 **边界框面积最大** 的人脸作为代表计算与参考图的相似度。
- 若最大脸仍低于阈值，状态为 `failed_multi_face_low_similarity`，`failure_reason` 中会包含人脸数量与分数说明，便于区分“单纯分低”与“多脸场景下的低分”。
- 报表中 `face_count` 列可用来审计多脸样本频率。

## 10. 报表解读

- **`validation_results.csv`**：一行一图；关注 `status`、`failure_reason`、`similarity`、`face_count`、`prompt_id`。适合表格软件筛选或二次统计。
- **`validation_summary.json`**：`passed` / `failed` 计数，`mean_similarity` 等聚合指标，`failure_reasons` 为失败原因直方图，适合程序消费或快速看分布。
- **`validation_report.md`**：与 JSON 同内容的可读版，含 **Summary**、**Failure reason breakdown**、**Per-image results**；交付或评审时可直接打开该文件（文件名即流水线中的 Markdown 校验报告）。

## 11. 系统失效点分析与改进

### 11.1 易失效场景

- **大角度侧脸**：当脸部旋转超过约 45° 时，InstantID 的身份还原与特征提取能力会下降。
- **遮挡物**：眼镜、头发大面积遮挡会干扰 Embedding。
- **背景干扰**：复杂背景中可能产生额外人脸，验证端虽取最大脸，仍可能拉低一致性或误判构图。

### 11.2 身份漂移（Identity Drift）来源

- **Prompt 冲突**：过强的表情描述可能扭曲肌肉结构。
- **底座模型权重**：写实类底座的审美倾向可能稀释参考脸特征。

### 11.3 下一步提升建议

- **加入 Face-Parsing**：面部解析掩码与背景保护。
- **多后端融合**：PuLID、IP-Adapter FaceID 等联合推理。
- **动态阈值**：按人脸分布自适应阈值或分类器。
