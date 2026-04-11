# OneShot-ID 项目需求与规范 (Project Requirements & Specifications)

## 1. 项目目标 (Goal)
从零实现一个单张人脸参考图驱动的身份一致性生成与验证系统。
- **输入**：1 张人物参考图。
- **输出**：8–12 张同一身份的图像（涵盖不同姿态、角度、表情、光照和背景变化）。
- **验证**：通过可量化的身份验证逻辑（Face Detection, Alignment, Embedding, Cosine Similarity）评估身份一致性。

## 2. 核心约束 (Constraints)
- **非纯 Prompt 方案**：必须包含显式的身份条件约束，不能仅靠 Prompt 描述“像同一个人”。
- **身份一等公民**：身份一致性（Identity Consistency）必须作为系统设计的核心。
- **闭合环路**：支持“生成后再验证”的自动化流程。
- **高度可配置化**：模型后端、相似度阈值、生成数量、随机种子、输出路径等均需支持配置。
- **运行环境**：使用 conda 环境 `py312_cu121`。
- **硬件设备**：默认使用本地的 1 号 GPU（cuda:1，具备 24G 显存）。
- **语言要求**：优先使用 Python 3.12 实现。

## 3. 已知背景与模型路径 (Known Context & Paths)
- **技术选型**：
  - 核心架构：InstantID
  - 生成模型底座：SDXL 模型
  - 人脸分析：InsightFace (AntelopeV2)
- **本地模型绝对路径**：
  - **InsightFace**: `/media/zzg/GJ_disk01/pretrained_model/stable-diffusion-webui/models/insightface/models`
  - **InstantID**: `/media/zzg/GJ_disk01/pretrained_model/InstantX/InstantID`
  - **SDXL 模型**: `/media/zzg/GJ_disk01/pretrained_model/stable-diffusion-webui/models/Stable-diffusion/SDXL/realvisxlV40_v40LightningBakedvae.safetensors`

## 4. 验证逻辑要求 (Validation Requirements)
- **对齐与提取**：对生成的每一张图进行人脸检测、对齐并提取 512 维 Embedding。
- **相似度计算**：计算生成的图像 Embedding 与参考图 Embedding 之间的余弦相似度（Cosine Similarity）。
- **统计信息**：输出批量统计结果（均值、最小值、最大值、标准差）。
- **异常处理**：
  - 若检测到多张脸：优先选择最大人脸，并记录异常。
  - 若未检测到人脸：记录失败原因。
  - 低于阈值的样本标记为失败。

## 5. 交付物 (Deliverables)
- 完整的模块化代码结构（符合 app/, src/, configs/ 等规范）。
- 生成的图片文件。
- 相似度结果报表（CSV/JSON/Markdown 格式）。
- 详细的中英文项目说明文档与运行指南。
