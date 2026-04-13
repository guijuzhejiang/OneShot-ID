# OneShot-ID 成果物与客户问答

## 1）生成成果物（run：`test0`）

本次运行生成的所有成果物都在：

- `outputs/runs/test0/`

目录含义如下：

- `outputs/runs/test0/candidates/`  
  存放**所有生成的人物图像**（生成阶段产出的全部候选图）。

- `outputs/runs/test0/kept/`  
  存放**满足人物身份一致性要求**的图像（被接受的最终输出）。

- `outputs/runs/test0/rejected/`  
  存放**不满足人物身份一致性要求**的图像（被拒绝的输出）。

- `outputs/runs/test0/faces/`  
  存放从候选图中截取出的**人脸裁剪图**（默认取检测到的最大人脸），用于复核与问题定位。

- `outputs/runs/test0/reports/`  
  存放记录**每张生成图像与参考图像相似度**的报表文件，包括：
  - `validation_results.csv`
  - `validation_report.md`

## 2）客户问题：身份一致性与失效模式

### Q：你们如何确保身份一致性（how you ensured identity consistency）？

我们使用 InstantID 作为身份一致性保持的核心机制。关键点在于：不只依赖 prompt 去“描述像同一个人”，而是把参考人脸以**显式的身份条件**注入到生成过程中。这样生成的人在面部结构与整体身份特征上会更加稳定。

### Q：系统从哪里开始失效（where the system starts to fail）？

当我们把变化幅度推得过大时，系统更容易开始失效，尤其是强姿态变化（如明显侧脸、抬头/低头角度）、夸张表情，或加入手、帽子、口罩等遮挡。在这些情况下，可用于身份判断的可见人脸线索会变弱，模型更容易偏离参考身份。

### Q：你们的 setup 里身份漂移的原因是什么（what causes identity drift in your setup）？

在我们的设置中，身份漂移主要来自：**大幅姿态变化**，以及**重遮挡/强表情变化**。当 prompt 同时叠加“侧脸 + 强表情 + 遮挡”等要求时，模型需要在“保持身份”和“满足变化”之间做权衡，这也是漂移通常开始出现的地方。

### Q：如果用于生产系统，你会如何提升身份稳定性（how you would improve identity stability in a production system）？

在生产系统中，为了提升身份稳定性，我们会加入生成后的验证环节。我们使用 InsightFace 分别对参考图与生成图提取人脸 embedding，然后计算余弦相似度。这样就能用**量化指标**来衡量身份一致性，而不是只依赖人工主观判断。

### Q：为什么某些输出应该被接受或拒绝（why certain outputs should be accepted or rejected）？

接受/拒绝逻辑非常直接：如果相似度分数高于阈值，则认为输出与参考身份足够接近，保存到 `kept/`；如果低于阈值，则说明身份漂移过大，保存到 `rejected/`。这给流水线提供了清晰的质量门槛，使最终输出更可靠、更适合生产交付。

