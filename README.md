# 🧠 BERT预训练数据分类器

该文档介绍了如何利用BERT来选择预训练数据集的训练框架以及测试方法。提供了项目的目录结构说明。用户需要通过创建conda环境进行环境搭建；**训练**时，需自行准备高质量且带标签的训练数据（我们也开放了检查点 `ckpt`）；**测试**时，需加载自己的 `ckpt` 对训练数据进行分类。

## 🛠️ 环境搭建

使用以下命令创建所需的conda环境并安装依赖：

```bash
conda create -n bertclassifier python=3.10
conda activate bertclassifier
pip install -r requirement.txt
```

## 🏋️‍♂️ 训练脚本

使用以下命令运行训练脚本：

```bash
python3 -u run_full.py \
  --data_path // 训练数据位置 \
  --train_data // 训练数据名称 \
  --model bert-base-uncased \ # 采用的模型，如 bert-base-uncased 等
  --batch_size // 批量大小 \
  --epoch // 训练轮数 \
  --lr // 学习率 \
  --ckpt_name // 检查点名称 \
  --num_classes 2 # 分类数，2 表示高质量和非高质量
```

### 参数说明

- `--data_path`: 训练数据的存放路径
- `--train_data`: 训练数据的文件名
- `--model`: 选择的预训练模型，例如 `bert-base-uncased`
- `--batch_size`: 每批训练的样本数
- `--epoch`: 训练的轮数
- `--lr`: 学习率
- `--ckpt_name`: 保存检查点的名称
- `--num_classes`: 分类类别数（默认2）

## 📂 分类脚本

配置好要分类文件的路径及 `ckpt` 的路径后，运行以下命令进行分类：

```bash
python3 -u run_predict.py
```

## 📁 项目目录结构

```
├── dataset.py         # 准备数据加载器
├── model.py           # 模型定义
├── README.md          # 项目说明
├── requirement.txt    # 项目依赖包说明
├── run_full.py        # 训练主函数
├── run_predict.py     # 分类主函数
├── trainers.py        # 训练器
├── utils.py           # 辅助函数
│
├── ckpt/              # 检查点存放处
└── data/              # 训练数据存放处
```

## 📌 注意事项

- 确保你的训练数据质量高且标注准确，以提升分类器的性能。
- 在训练前，可以根据需要调整超参数（如 `batch_size`、`epoch`、`lr`）以获得最佳效果。
- 我们已开放部分检查点 (`ckpt`)，用户可以根据需要进行加载和测试。
