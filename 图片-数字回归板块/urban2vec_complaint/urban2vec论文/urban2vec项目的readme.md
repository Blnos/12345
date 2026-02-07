[原论文的项目Urban2Vec](https://github.com/wangzhecheng/urban2vec_)

# Urban2Vec 项目概述

> **多模态城市社区嵌入系统**
>
> 结合街景图像与 POI（兴趣点）数据，学习城市邻里的多模态表示。
>
> 📚 **论文来源**：本项目实现了 Wang, Li, Rajagopal 发表于 AAAI 2020 的论文：
> *《Urban2Vec: Incorporating Street View Imagery and POIs for Multi-Modal Urban Neighborhood Embedding》*

---

## 📂 目录结构

以下是项目的核心文件结构概览：

```text
Project_Root/
├── urban2vec_step1/           # 第一步：街景图像嵌入
│   ├── mymodels/
│   │   └── skip_gram.py       # PlaceImageSkipGram (Inception3) 模型
│   ├── utils/
│   │   └── image_dataset.py   # 图像对数据集类
│   ├── train_place_embedding.py # Step 1 主训练脚本
│   └── embedding.py           # 嵌入生成工具
│
├── urban2vec_step2/           # 第二步：融入 POI 信息
│   ├── mymodels/
│   │   └── skip_gram.py       # PlaceSkipGram (Sigmoid) 模型
│   ├── utils/
│   │   └── image_dataset.py   # PlacePairDataset 类
│   └── train_place_embedding.py # Step 2 主训练脚本
│
├── pic_preprocess/            # 街景图像预处理
│   ├── pic_form_context.py    # 基于地理邻近性创建空间上下文
│   └── pic_form_pair.py       # 生成对比学习训练/验证对
│
├── poi_preprocess/            # POI 数据处理
│   ├── poi_crawl_data.py      # Yelp API 爬虫 (商业数据/评论)
│   ├── review_form_pair.py    # 处理评论并生成词对
│   └── poi_form_pair.py       # 生成 POI 对和类别
│
├── pic_analysis/              # 街景分析 (回归分析)
├── poi_analysis/              # POI 分析 (聚类、人口统计回归)
└── dimension_correlation/     # 维度分析 (PCA)
```

---

## 🏗️ 核心组件详情

### 1. 核心训练组件 (Training Components)

#### **Step 1: 街景图像嵌入 (Street View Embedding)**

* **路径**: `urban2vec_step1/`
* **目标**: 从街景图像中学习视觉特征。
* **关键文件**:

  * `mymodels/skip_gram.py`: 包含 `PlaceImageSkipGram` 和 `PlaceImageEmb` 模型，使用修改版的 **Inception3** 架构。
  * `train_place_embedding.py`: 第一阶段的主训练脚本。
  * `utils/image_dataset.py`: 处理图像对的数据加载。

#### **Step 2: 融入 POI 信息 (Incorporating POIs)**

* **路径**: `urban2vec_step2/`
* **目标**: 将文本信息（POI）融入到视觉嵌入中。
* **关键文件**:

  * `mymodels/skip_gram.py`: 包含 `PlaceSkipGram` 和 `PlaceSkipGrammargin` 模型。
  * `train_place_embedding.py`: 第二阶段的主训练脚本，加载第一阶段的预训练模型。

### 2. 数据预处理 (Data Preprocessing)

#### **街景预处理**

* **路径**: `pic_preprocess/`

-----.py`**: 为对比学习（Contrastive Learning）生成正负样本对。

#### **POI 预处理**

* **路径**: `poi_preprocess/`

----- 清洗评论数据并生成词对。

* **`poi_form_pair.py`**: 生成 POI 类别对。

### 3. 分析组件 (Analysis)

* **`pic_analysis/`**: 街景分析（如：`pic_regression.py` 使用嵌入向量进行回归预测）。
* **`poi_analysis/`**: POI 分析（包含聚类分析、人口统计学特征回归）。
* **`dimension_correlation/`**: 维度相关性分析（如：`pca1.py` 进行 PCA 主成分分析）。

---

## 🔑 关键技术与模型架构

### 模型定义

1. **PlaceImageSkipGram (`urban2vec_step1`)**

   * **架构**: 基于 CNN 的图像嵌入模型。
   * **骨干网**: 修改版的 **Inception3**。
   * **用途**: 提取纯视觉特征。

2. **PlaceSkipGram (`urban2vec_step2`)**

   * **架构**: 用于联合学习地点（Place）和词（Word）嵌入的模型。
   * **机制**: 采用 Sigmoid 评分函数。

### 训练流程

1. **第一步训练 (Visual Only)**

   * **脚本**: `urban2vec_step1/train_place_embedding.py`
   * **方法**: 使用带边界排名的 **三元组损失 (Triplet Loss with Margin Ranking)**。
   * **数据**: 使用 `PlaceImagePairDataset` 生成 "锚点-正例" 图像对。

2. **第二步训练 (Visual + Textual)**

   * **脚本**: `urban2vec_step2/train_place_embedding.py`
   * **方法**: 加载第一步训练好的嵌入作为初始化，融入 POI 文本信息进行微调。

---

## 🔄 项目工作流程 (Workflow)

1. **数据采集与预处理**:

   * 使用 `poi_crawl_data.py` 获取 Yelp 数据。
   * 使用 `pic_form_context.py` 基于地理位置建立图像关联。
   * 生成图像对和 POI 词对。
2. **Step 1 训练**:

   * 运行 `urban2vec_step1` 中的训练脚本，仅从街景图像学习视觉特征。
3. **Step 2 训练**:

   * 加载 Step 1 的模型，运行 `urban2vec_step2` 脚本，融合 POI 文本信息。
     4向量的质量。

---

## 📦 依赖项 (Dependencies)

* **PyTorch**: 核心深度学习框架。
* **TorchVision**: 提供 Inception3 预训练模型和图像转换工具。
* **scikit-learn**: 用于 PCA 降维、回归模型及评估指标。
* **NumPy / Pandas**: 科学计算与数据处理。
* **geopy**: 地理坐标计算与距离处理。

---

## 📝 总结

该项目展示了一种复杂的**两阶段城市社区表示学习方法**：

1. 技术处理 POI 数据。
2. 通过多模态融合，生成包含丰富语义信息的城市社区嵌入向量。
