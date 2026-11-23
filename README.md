# SRPA-Net

## 代码结构

```text
SRPA-Net/
├── train.py                                # 模型训练与验证主程序
├── run_experiments.py                      # 实验准备与多配置顺序执行入口
├── infer_scene.py                          # 整景高光谱影像分块推理与拼接脚本
├── configs/                                # 实验配置与光谱元信息
│   ├── selected_spectral_detail.json       # 优选配置（全局光谱可靠性 + 光谱细节补偿）
│   ├── factorial_2x2.json                  # 2×2 消融实验配置
│   ├── innovations_v2.json                 # 结构探索配置（局部可靠性/空间适配/光谱细节）
│   ├── experiment.json                     # 基础四组消融配置
│   ├── sensor_metadata_224.json            # 224 波段中心波长、FWHM 与传感器元信息
│   └── text_embeddings.pt                  # SpecAware 预缓存属性文本嵌入
├── environment/                            # 运行环境依赖
│   ├── environment.yml                     # Conda 环境配置
│   └── requirements.txt                    # Python 依赖包列表
├── scripts/                                # 训练启动与检查点评估脚本
│   ├── train_selected.sh                   # 默认优选配置训练启动脚本
│   └── evaluate_locked_checkpoint.py       # 已锁定最佳检查点的验证与测试评估脚本
├── src/                                    # 核心模型、数据与训练控制模块
│   ├── __init__.py
│   ├── model.py                            # SRPANet 主体、光谱可靠性估计器与条件适配器
│   ├── innovations.py                      # 局部可靠性、多尺度空间适配与光谱细节补偿模块
│   ├── mechanism_controls.py               # 静态波段权重与细节分支单因素机制对照模块
│   ├── weighting_location.py               # 原始输入/聚合位置加权对照（解耦梯度策略）
│   ├── shared_weighting_location.py        # 原始输入/聚合位置加权对照（共享梯度策略）
│   ├── data.py                             # 224 波段溢油图块数据集读取与元信息子集工具
│   ├── runtime.py                          # GhostBatchNorm、混合精度回退与融合优化器控制
│   ├── experiment_controls.py              # 学习率调度、Dice/Tversky 损失与训练控制函数
│   └── factorial.py                        # 2×2 消融结果汇总分析工具
├── tests/                                  # 模块机制、初始化一致性与推理单元测试
│   ├── test_mechanisms.py                  # 基础可靠性与适配器机制测试
│   ├── test_innovations.py                 # 创新模块前向与梯度测试
│   ├── test_innovations_initialization.py  # 多变体共享初始化一致性测试
│   ├── test_factorial.py                   # 2×2 消融初始化与路径检查
│   ├── test_detail_routing.py              # 光谱细节分支权重与上下文路由测试
│   ├── test_mechanism_controls.py          # 机制对照变体初始化与梯度测试
│   ├── test_weighting_location.py          # 加权位置对照测试
│   ├── test_shared_weighting_location.py   # 共享梯度加权位置对照测试
│   ├── test_loss_schedule_controls.py      # 损失函数与学习率调度测试
│   ├── test_overlap_curriculum.py          # 重叠损失课程调度测试
│   ├── test_tversky_controls.py            # Tversky/Dice 损失数值测试
│   ├── test_overnight_controls.py          # 训练控制与可靠性调度测试
│   ├── test_pretrained_token_scale.py      # 预训练编码器 token 尺度一致性测试
│   ├── test_runtime.py                     # GhostBatchNorm 等运行时控制测试
│   ├── test_runtime_cuda.py                # CUDA 加速配置等价性测试
│   └── test_scene_stitching.py             # 整景影像分块推理与拼接测试
└── vendor/                                 # 上游依赖源码
    ├── SpecAware_README.md                 # SpecAware 上游说明
    ├── specaware/                          # SpecAware 编码器、动态光谱嵌入与 UPerNet 解码器
    │   ├── __init__.py
    │   ├── HyperEmbedding.py
    │   ├── SpectralEmbedding.py
    │   ├── models_SpecAware_encoder.py
    │   ├── models_SpecAware_upernet.py
    │   ├── aurora/                         # 面积与傅里叶位置编码工具
    │   ├── UperNet/                        # MultiLevelNeck 与 UPerNet 解码器
    │   └── util/                           # 2D sincos 位置编码工具
    └── benchmark/                          # 基准数据读取、数据增强与对比模型定义
        ├── run_benchmark.py
        ├── models.py
        └── original_augmentations.py
```
