#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/models.py - 街景嵌入模型定义
基于Inception-v3的嵌入模型，用于学习街景图像的向量表示。

包含：
- PlaceImageSkipGram: 用于对比学习的嵌入模型
- Inception3_modified: 修改版的Inception-v3，输出2048维特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception3
from typing import Optional, List, Tuple


class Inception3_modified(Inception3):
    """
    修改版的Inception-v3模型，移除最后的分类层，输出2048维特征向量。
    基于原Urban2Vec代码中的实现。
    """

    def __init__(self, num_classes: int = 1000, aux_logits: bool = False,
                 transform_input: bool = False, init_weights: bool = True):
        super().__init__(num_classes=num_classes, aux_logits=aux_logits,
                        transform_input=transform_input, init_weights=init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回2048维特征向量

        Args:
            x: 输入图像张量，形状为 [batch_size, 3, 299, 299]

        Returns:
            2048维特征向量，形状为 [batch_size, 2048]
        """
        # 如果需要，转换输入（ImageNet标准化）
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # Inception-v3前向传播（直到全局平均池化层）
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8

        # 自适应平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048

        return x


class PlaceImageSkipGram(nn.Module):
    """
    街景图像嵌入模型，用于对比学习。
    基于Inception-v3提取特征，然后通过线性层映射到嵌入空间。
    """

    def __init__(self, embedding_dim: int = 50, pretrained: bool = True):
        """
        初始化模型

        Args:
            embedding_dim: 输出嵌入维度（默认：50）
            pretrained: 是否使用ImageNet预训练的Inception-v3权重
        """
        super(PlaceImageSkipGram, self).__init__()

        # 主干网络：修改版的Inception-v3
        self.inception3 = Inception3_modified(aux_logits=False, transform_input=False)

        # 加载预训练权重（ImageNet）
        if pretrained:
            try:
                # 加载torchvision预训练的Inception-v3
                from torchvision.models import inception_v3
                pretrained_model = inception_v3(pretrained=True, transform_input=False)

                # 复制权重（除了最后的fc层）
                state_dict = pretrained_model.state_dict()
                # 删除fc层的权重
                state_dict.pop('fc.weight', None)
                state_dict.pop('fc.bias', None)

                # 加载权重到我们的模型
                self.inception3.load_state_dict(state_dict, strict=False)
                print("成功加载ImageNet预训练权重")
            except Exception as e:
                print(f"加载预训练权重失败: {e}，使用随机初始化")

        # 冻结Inception层（可选，训练时可解冻）
        # for param in self.inception3.parameters():
        #     param.requires_grad = False

        # 线性层：2048维 -> embedding_dim维
        self.linear1 = nn.Linear(2048, embedding_dim)

        # 初始化线性层
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            images: 输入图像张量，形状为 [batch_size, 3, 299, 299]

        Returns:
            嵌入向量，形状为 [batch_size, embedding_dim]
        """
        # 通过Inception-v3提取特征
        features = self.inception3(images)  # [batch_size, 2048]

        # 通过线性层映射到嵌入空间
        embeddings = self.linear1(features)  # [batch_size, embedding_dim]

        return embeddings

    def load_pretrained_cnn(self, cnn_model_path: str, device: torch.device = torch.device('cpu')):
        """
        加载预训练的CNN权重（兼容原代码）

        Args:
            cnn_model_path: CNN模型权重文件路径
            device: 设备
        """
        old_params = torch.load(cnn_model_path, map_location=device)
        if cnn_model_path.endswith('.tar'):  # 检查点字典
            old_params = old_params.get('model_state_dict', old_params)

        # 删除fc层的权重（如果存在）
        old_params.pop('fc.weight', None)
        old_params.pop('fc.bias', None)

        # 加载权重
        self.inception3.load_state_dict(old_params, strict=False)
        print(f'加载预训练CNN参数从: {cnn_model_path}')

    def freeze_cnn(self, freeze: bool = True):
        """
        冻结或解冻CNN部分的参数

        Args:
            freeze: 是否冻结CNN参数
        """
        for param in self.inception3.parameters():
            param.requires_grad = not freeze
        print(f"CNN参数已{'冻结' if freeze else '解冻'}")


class PlaceImageEmb(nn.Module):
    """
    简化的嵌入模型，用于推理阶段提取图像嵌入。
    与PlaceImageSkipGram结构相同，但提供更简洁的接口。
    """

    def __init__(self, embedding_dim: int = 50, pretrained: bool = True):
        super(PlaceImageEmb, self).__init__()
        self.model = PlaceImageSkipGram(embedding_dim, pretrained)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict)

    def state_dict(self):
        return self.model.state_dict()