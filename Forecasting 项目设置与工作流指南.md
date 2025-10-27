本文档旨在介绍 TOTEM forecasting 项目的结构、标准化的数据处理与模型训练流程，并详细说明近期为提升脚本稳健性与易用性所做的增强。

## 1. 项目结构概览

forecasting 目录是本项目的核心，其结构组织如下，旨在分离数据、源码、脚本与模型：

```plaintext
forecasting/
│
├── 📜 combine_datasets_for_forecaster.py  # 合并已提取的预测数据
├── 📜 combine_datasets_for_vqvae.py       # 合并用于 VQVAE 训练的基础数据
│
├── 📁 data_provider/                      # 数据加载与预处理模块
│   ├── data_factory.py                     # 数据集工厂，按需创建加载器
│   └── data_loader.py                      # PyTorch Dataset 定义 (e.g., Dataset_Custom)
│
├── 📁 dataset/                            # 原始数据集 (CSV 格式)
│   ├── ETT-small/
│   ├── electricity/
│   ├── traffic/
│   └── weather/
│
├── 📜 extract_forecasting_data.py         # 使用 VQVAE 提取时序数据的 code
├── 📜 generalist_eval.py                  # 通用模型评估脚本
│
├── 📁 lib/                                # 核心库
│   ├── 📁 models/                         # 模型定义 (VQVAE, RevIN, Transformer等)
│   │   ├── core.py
│   │   ├── decode.py
│   │   └── vqvae.py
│   └── 📁 utils/                          # 实用工具
│       ├── checkpoint.py                   # 检查点管理
│       ├── env.py                          # 环境变量与路径辅助 (核心)
│       └── timefeatures.py                 # 时间特征工程
│
├── 📜 save_revin_data.py                  # 为 ReVIN 创建预处理数据
│
├── 📁 scripts/                            # 自动化运行脚本与配置文件
│   ├── *.json                              # 模型与实验配置文件
│   └── *.sh                                # 编排数据处理与训练流程的 Shell 脚本
│
├── 📜 train_forecaster.py                 # 训练最终的预测模型
└── 📜 train_vqvae.py                      # 训练 VQVAE 模型
```

### 关键目录与文件说明

- lib/utils/env.py : 核心环境文件 。提供 get_default_output_dir() 和 ensure_dir() 等关键函数，用于统一输出路径和确保目录存在。所有增强脚本均依赖此文件来管理路径。
- scripts/*.sh : 最佳实践示例 。这些脚本展示了从数据预处理到模型训练的完整流程，是理解多步骤操作的绝佳参考。
- combine_*.py : 数据合并脚本 。我们将这些脚本增强为自动化、稳健的数据处理工具。
- extract_forecasting_data.py : 特征提取脚本 。同样经过路径统一与进度显示增强。

## 2. 核心脚本增强说明

我们对数据处理流程中的三个关键脚本进行了标准化改造，显著提升了其 易用性 、 稳健性 和 一致性 。

### 通用增强

- 统一路径管理 ：所有脚本的 --root_path 和 --save_path 参数均增加了回退逻辑。若不提供，则默认使用 lib/utils/env.py 中定义的标准输出目录（如 output/data/ ），避免了硬编码路径。
- 稳健的目录创建 ：所有保存操作前，都会调用 ensure_dir() 检查并创建目标目录，防止因路径不存在而执行失败。
- 清晰的执行进度 ：在所有耗时的数据迭代引发中加入了 tqdm 进度条，直观展示处理进度。
- 跨平台路径兼容 ：所有文件路径拼接均使用 os.path.join() ，确保在不同操作系统（Windows/Linux）下均能正确执行。

### combine_datasets_for_vqvae.py 的特别增强

为使其在面对不规范数据时表现更稳健，我们加入了以下校验：

1. 友好的缺失文件提示 ：脚本会首先检查所有需要合并的文件是否存在，若有缺失，则一次性列出所有缺失文件并终止，而不是逐个报错。
2. 数据加载与类型校验 ：
   - 加载 .npy 文件时强制使用 allow_pickle=True ，兼容多种数据格式。
   - 加载后强制转换为 np.float32 ，确保后续模型训练的输入类型一致。
   - 若加载或转换失败，会抛出明确的错误信息，直指问题文件。
3. 维度一致性检查 ：在拼接前，脚本会检查所有数据集的时间维度（ shape[1] ）是否一致。若不一致，则报错并列出各数据集的维度，提示用户检查 pred_len 等参数是否统一。

## 3. 标准化工作流与操作指令

以下是推荐的数据处理与模型训练流程，所有命令均可在 forecasting/ 目录下执行。

### 步骤 1：为各数据集生成 ReVIN 数据

此步骤为每个独立的数据集（如 weather, electricity）进行预处理。

```bash
# 示例：处理 electricity 数据集
python save_revin_data.py \
  --data custom \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --save_path "output/data/electricity/" \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 321
```

注意 ：请为 scripts/*.sh 中列出的所有数据集重复此操作。 save_path 已按数据集名称组织。

### 步骤 2：合并所有数据集以训练 VQVAE

使用我们增强后的脚本，无需任何参数即可运行。它会自动从 output/data/ 读取并合并到 output/data/all/ 。

```bash
# 自动合并所有数据集
python combine_datasets_for_vqvae.py
```


此脚本现在包含强大的错误检查功能。如果中途失败，请根据提示检查缺失文件或数据维度。

### 步骤 3：训练通用 VQVAE 模型

使用上一步合并的 all 数据集来训练一个通用的 VQVAE。

```bash
python train_vqvae.py \
  --config_path forecasting/scripts/all.json \
  --base_path "output/data" \
  --save_path "output/saved_models/all/" \
  --batchsize 4096 \
  --comet_log
```


说明 ： --base_path 指向合并数据的根目录。 --save_path 指向模型保存位置。

### 步骤 4：使用通用 VQVAE 提取时序 Code

为每个数据集和不同的 Tin/Tout 组合提取离散的 code 。

```
# 示例：为 electricity 数据集提取 Tin=96, Tout=96 的 code
python extract_forecasting_data.py \
  --data custom \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 321 \
  --save_path "" \
  --trained_vqvae_model_path '/path/to/your/trained/vqvae.pt'
```


说明 ： --save_path 留空后，脚本将自动保存到 output/data/electricity/Tin96_Tout96/ ，这是我们实现的核心便利功能。请将 /path/to/your/trained/vqvae.pt 替换为步骤 3 中训练好的模型路径。

### 步骤 5：合并提取的 Code 以训练预测器

再次使用我们增强的脚本，无需参数即可自动合并所有 Tin/Tout 组合的数据。

```
# 自动合并所有已提取的 code
python combine_datasets_for_forecaster.py
```

它会自动从 output/data/ 下的各个子目录读取，并保存到 output/data/all/ 。

### 步骤 6：训练最终的预测模型

使用上一步合并的数据训练 Transformer 预测模型。

```
python train_forecaster.py --file_save_path ""
```

说明 ： train_forecaster.py 也已集成了默认路径逻辑，它会自动从 output/data/all/ 加载数据，并将检查点保存到 output/checkpoints/ 。
