# Urban2Vec 投诉预测项目

## 📚 引用

本项目基于 **Urban2Vec** 论文思想实现：

> **原论文**: *Urban2Vec: Learning Urban Neighborhood Representation with Graph Attention Neural Network* 

> **本项目改进**: 将 POI 文本信息替换为投诉预测任务（仅借鉴使用其社区图像学习方法）

---

## 📖 项目概述

本项目包含了完整的 **"街景图像 → 12345投诉预测"** 流程，包含以下 6 个核心脚本：

* **`0_adapt_data.py` - 文件名解析器**：从图片文件名提取元数据
* **`1_build_pairs.py` - 训练对构建器**：基于地理距离构建 K 近邻训练对
* **`2_train_stage1.py` - 街景嵌入训练**：训练街景图像嵌入模型
* **`3_extract_embedding.py` - 社区特征提取器**：提取社区级嵌入向量
* **`4_train_predictor.py` - 多模型回归**：训练投诉预测模型
* **`5_pipeline.py` - 一键流程**：支持训练和预测两种模式

目前在google colab上进行调试，工作文件已保存为[12345_v1.ipynb](12345_v1.ipynb)

---

## 📂 数据格式要求

### 1. 图片文件结构

```text
街景图片/
├── 社区1_images/
│   ├── 0_社区1_110105005000_116.4_39.9_0_0.png
│   ├── 0_社区1_110105005000_116.4_39.9_90_0.png
│   ├── 0_社区1_110105005000_116.4_39.9_180_0.png
│   └── 0_社区1_110105005000_116.4_39.9_270_0.png
├── 社区2_images/
│   └── ...
└── ...
```



**2. 投诉记录文件说明：**

* `total_complaints`: 总投诉量
* `house_management`: 房屋管理类投诉
* `traffic_municipal`: 交通市政类投诉
* `public_service`: 公共服务类投诉
* `social_affairs`: 社会事务类投诉
* `market_economy`: 市场经济类投诉
* `urban_rural`: 城乡规划类投诉
* `OTHER`: 其他投诉


```csv
community_name,total_complaints,house_management,traffic_municipal,public_service,social_affairs,market_economy,urban_rural,OTHER
社区1,125,30,25,20,15,10,5,20
社区2,89,20,18,15,12,8,4,12
...
```
---

## 🚀 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install -r requirements.txt

# 注意：PyTorch和XGBoost可能需要单独安装
# 例如安装CPU版本的PyTorch：
pip install torch torchvision
```

### 2. 训练全流程

**方式 1：使用一键流程脚本**

```bash
python 5_pipeline.py --mode train \
  --image_root ./街景图片/ \
  --label_path ./complaints.csv \
  --output_dir ./output
```

**方式 2：分步执行**

```bash
# Stage 0: 数据准备（生成元数据）
!python 0_adapt_data.py \
  --image_root "./data/raw/images/" \
  --output "./data/processed/metadata.csv"

# Stage 1: 构建训练对（生成 pickle 文件）
!python 1_build_pairs.py \
  --metadata "./data/processed/metadata.csv" \
  --output_dir "./data/processed/"

# Stage 2: 训练街景嵌入模型（生成 .tar 模型文件）⭐ 你漏了这步！
!python 2_train_stage1.py \
  --train_pairs "./data/processed/train_pair_knn.pickle" \
  --val_pairs "./data/processed/val_pair_knn.pickle" \
  --image_root "./data/raw/images/" \
  --output_dir "./models/checkpoints/"

# Stage 3: 提取社区嵌入向量（使用 Stage 2 的模型）
!python 3_extract_embedding.py \
  --image_root "./data/raw/images/" \
  --model_path "./models/checkpoints/street_view_best.tar" \
  --output "./embeddings/train/community_embeddings.csv"

# Stage 4: 训练投诉预测模型（如果样本够的话）
!python 4_train_predictor.py \
  --embeddings "./embeddings/train/community_embeddings.csv" \
  --complaints "./data/raw/complaints.csv" \
  --output_dir "./results/evaluation/"
```

### 3. 预测新社区

```bash
# 运行前需准备新社区图片目录（格式与训练数据相同）
python 5_pipeline.py --mode predict \
  --new_community_dir ./新社区图片/ \
  --model_path ./models/checkpoints/street_view_best.tar \
  --output_dir ./results/predictions/
```

---

## 🏗️ 项目结构

```text
urban2vec_complaint/
├── 0_adapt_data.py          # 脚本0：文件名解析器
├── 1_build_pairs.py         # 脚本1：训练对构建器
├── 2_train_stage1.py        # 脚本2：街景嵌入训练
├── 3_extract_embedding.py   # 脚本3：社区特征提取器
├── 4_train_predictor.py     # 脚本4：多模型回归
├── 5_pipeline.py            # 脚本5：一键流程
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖包列表
├── README.md                # 本文档
└── src/                     # 核心模块
    ├── models.py            # 模型定义
    ├── datasets.py          # 数据集类
    └── utils.py             # 工具函数
```

---

## ⚙️ 配置说明

编辑 `config.yaml` 文件可以调整所有参数：

```yaml
# Stage1训练配置
stage1_training:
  embedding_dim: 50    # 嵌入维度：50或200
  batch_size: 32
  num_epochs: 20
  learning_rate: 0.0005
  margin: 5.0          # 对比学习边界值

# 训练对构建配置
pair_building:
  k_neighbors: 6       # K近邻数量
  train_ratio: 0.8     # 训练集比例

# 投诉预测配置
prediction:
  cv_folds: 5          # 交叉验证折数
  models: ["Ridge", "SVR", "RandomForest", "XGBoost"]
```

---

## 🔑 关键技术说明

1. **方位角去重（街景图未拼接前暂时方案）**

   * 同一采样点（相同 `point_id`）的不同方位角图片不互为最近邻。
   * 避免同一点不同角度的图片成为正样本对。
2. **社区级聚合**

   * 同一社区所有图片的嵌入向量进行平均池化（Mean Pooling）。
   * 4 个方位角的图片全部参与平均（街景图未拼接前暂时方案）。
3. **多模型对比**

   * 支持 4 种回归模型：Ridge, SVR, RandomForest, XGBoost。
   * 自动选择最佳模型并保存。
4. **全面评估**

   * 5 折交叉验证。
   * 评估指标：R², MAE, RMSE。
   * 可视化：预测 vs 真实散点图、特征重要性、PCA 降维、残差图。
5. **健壮性设计**

   * 自动 GPU/CPU 检测。
   * 图片损坏/字段缺失自动跳过。
   * 详细的日志和进度显示。

---

## 📂 输出文件说明

训练流程目录结构：

```text
urban2vec_complaint/
├── data/                          # 原始数据与中间数据
│   ├── raw/                       # 原始输入
│   │   ├── images/                # 街景图片
│   │   ├── complaints.csv         # 投诉数据
│   │   └── communities.csv        # 社区基础信息
│   └── processed/                 # 处理后数据（Stage 1-2 产出）
│       ├── metadata.csv           # 图片元数据
│       ├── context_knn.pickle     # K近邻上下文关系
│       ├── train_pair_knn.pickle  # 训练对
│       └── val_pair_knn.pickle    # 验证对
│
├── models/                        # 模型文件（Stage 2-3 产出）
│   ├── checkpoints/               # 训练检查点
│   │   ├── street_view_epoch5.tar
│   │   ├── street_view_epoch10.tar
│   │   └── street_view_best.tar   # 最佳模型（用于后续预测）
│   └── predictors/                # Stage 4 预测模型
│       ├── ridge_total.pkl
│       ├── xgboost_house.pkl
│       └── ...                    # 各类别预测模型
│
├── embeddings/                    # 嵌入向量（核心产出）
│   ├── train/                     # 训练集社区嵌入
│   │   └── community_embeddings.csv
│   └── predict/                   # 新社区预测时生成
│       ├── new_community_metadata.csv
│       └── new_community_embeddings.csv
│
├── results/                       # 结果与评估（Stage 4 产出）
│   ├── evaluation/                # 模型评估
│   │   ├── model_evaluation_results.csv
│   │   └── metrics_summary.json
│   ├── visualizations/            # 可视化图表
│   │   ├── correlation_heatmap.png
│   │   ├── model_comparison.png
│   │   └── *_detailed.png         # 各类别详细图
│   └── predictions/               # 预测结果
│       ├── community_predictions.csv    # 社区级预测
│       └── detailed_predictions.csv     # 详细预测
│
└── logs/                          # 训练日志
    ├── train_stage1.log
    ├── train_stage2.log
    └── predict.log
```

---

## ⚠️ 注意事项

* **图片数量**：每个社区建议至少 10 张以上图片，以保证嵌入质量。
* **内存需求**：Stage1 训练需要较大显存，如遇 OOM 可调整 `batch_size`。
* **计算时间**：完整训练流程可能需要数小时（取决于数据量和 GPU）。
* **数据质量**：确保图片文件名格式正确，经纬度信息准确。
* **社区名称一致性**：图片文件夹名、文件名中的社区名、投诉数据中的社区名必须完全一致。

---

## 🔧 故障排除

### 常见问题

* **"没有找到有效图片"**：检查图片文件名格式和目录结构。
* **"CUDA out of memory"**：减小 `batch_size` 或使用 CPU。
* **"社区名称不匹配"**：检查投诉数据中的社区名称与图片是否对应。
* **"模型加载失败"**：检查模型文件路径和格式。

### 日志查看

所有脚本都输出详细日志，可通过修改 `config.yaml` 中的 `log_level` 调整日志级别：

```yaml
general:
  log_level: "DEBUG"  # 可选：DEBUG/INFO/WARNING/ERROR
```

---

## 🛠️ 扩展开发

* **添加新模型**：在 `4_train_predictor.py` 的 `MODELS` 字典中添加新模型即可。
* **修改嵌入维度**：在 `config.yaml` 中修改 `embedding_dim` 参数，支持 50 或 200 维。
* **自定义目标列**：在 `config.yaml` 的 `target_columns` 中添加或修改目标列名称。



