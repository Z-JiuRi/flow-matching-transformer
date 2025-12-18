# README.md

# Flow-Matching Conditional Generation

基于Transformer和Flow-Matching的条件生成框架。

## 项目结构

.
├── configs/
│   └── config.yaml          # 配置文件
├── core/
│   ├── init.py
│   ├── trainer.py           # 训练器
│   ├── tester.py            # 测试器
│   └── inferencer.py        # 推理器
├── models/
│   ├── init.py
│   ├── normalization.py     # RMSNorm, PreNorm
│   ├── embeddings.py        # 各种嵌入层
│   ├── attention.py         # 注意力机制
│   ├── transformer.py       # Transformer模型
│   └── flow_matching.py     # Flow-Matching核心
├── utils/
│   ├── init.py
│   ├── helpers.py           # 通用工具函数
│   └── data_utils.py        # 数据处理工具
├── scripts/
│   └── generate_dummy_data.py  # 生成测试数据
├── main.py                  # 主程序
├── requirements.txt
└── README.md


## 安装

```bash
pip install -r requirements.txt
```

数据格式
<data_id>_linear1.pth: (H, W) 的tensor，对应layer=0
<data_id>_linear2.pth: (H, W) 的tensor，对应layer=1
<data_id>.pt: 一维条件向量
使用方法
1. 生成测试数据
```bash
python scripts/generate_dummy_data.py --output_dir ./data --num_samples 100
```
2. 训练
```bash
python main.py --mode train --config configs/config.yaml
```
恢复训练：
```bash
python main.py --mode train --config configs/config.yaml --resume checkpoints/epoch_50.pt
```
3. 测试
```bash
python main.py --mode test --config configs/config.yaml --checkpoint checkpoints/best.pt
```
4. 推理
```bash
python main.py --mode inference \
    --config configs/config.yaml \
    --checkpoint checkpoints/best.pt \
    --condition data/sample_0000.pt \
    --layer 0 \
    --output outputs/generated.pth
```
配置说明
主要配置项在 configs/config.yaml 中：

data: 数据相关配置（路径、尺寸、条件维度等）
model: 模型配置（Transformer参数、Flow-Matching参数）
training: 训练配置（批大小、学习率、优化器等）
inference: 推理配置（采样步数、引导强度等）
模型架构
输入处理: 将 (H, W) 的tensor展平为序列
条件嵌入:
时间步嵌入 (正弦编码 + MLP)
条件向量嵌入 (MLP)
Layer嵌入 (Embedding)
Transformer: 使用AdaLN注入条件的Transformer块
Flow-Matching: 学习从噪声到数据的速度场
采样方法
支持三种ODE求解器：

euler: 欧拉方法
heun: Heun方法（改进的欧拉）
rk4: 四阶龙格-库塔方法

这个框架包含了完整的训练、测试和推理流程。主要特点：

1. **模型架构**：使用Transformer + AdaLN进行条件注入
2. **Flow-Matching**：实现了标准的flow-matching训练和多种ODE求解器
3. **归一化**：使用RMSNorm和PreNorm
4. **条件处理**：支持条件向量和layer索引的联合条件
5. **训练功能**：EMA、混合精度、梯度裁剪、学习率调度
6. **推理功能**：支持插值、变体生成、求解器比较等
