#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/datasets.py - 数据加载器定义
包含用于训练和推理的数据集类。
"""

import os
from pathlib import Path  # 新增：用于跨平台路径处理
from typing import List, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset
from PIL import Image


class PlaceImagePairDataset(Dataset):
    """
    用于对比学习的图像对数据集。
    每个样本包含锚点图像、正样本图像和社区名称。
    """

    def __init__(self, root_dir: str, pair_list: List[Tuple[str, str, str]], transform=None):
        """
        初始化数据集

        Args:
            root_dir: 图像根目录
            pair_list: 图像对列表，每个元素为 (anchor_path, positive_path, community_name)
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.pair_list = pair_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pair_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        """
        获取一个样本

        Returns:
            anchor_image: 锚点图像张量
            positive_image: 正样本图像张量
            community_name: 社区名称（字符串）
            mode: 固定为1（表示正样本对），用于兼容原代码
        """
        anchor_path, positive_path, community_name = self.pair_list[idx]

        # 构建完整路径（跨平台适配：自动处理 Windows \ 和 Linux /）
        anchor_full_path = Path(os.path.join(self.root_dir, anchor_path)).as_posix()
        positive_full_path = Path(os.path.join(self.root_dir, positive_path)).as_posix()

        # 加载图像
        anchor_image = Image.open(anchor_full_path)
        positive_image = Image.open(positive_full_path)

        # 转换为RGB（处理灰度图）
        if not anchor_image.mode == 'RGB':
            anchor_image = anchor_image.convert('RGB')
        if not positive_image.mode == 'RGB':
            positive_image = positive_image.convert('RGB')

        # 应用变换
        if self.transform is not None:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)

        # 社区名称转换为整数ID（简单哈希）
        # 这里可以改为使用社区名称到ID的映射
        community_id = hash(community_name) % 10000

        # 返回结果，mode固定为1（正样本对）
        return anchor_image, positive_image, community_id, 1


class ImageDataset(Dataset):
    """
    用于推理的单个图像数据集。
    用于提取图像嵌入向量。
    """

    def __init__(self, root_dir: str, image_paths: List[str], transform=None):
        """
        初始化数据集

        Args:
            root_dir: 图像根目录
            image_paths: 图像路径列表（相对路径）
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        获取一个样本

        Returns:
            image: 图像张量
            image_path: 图像相对路径
        """
        image_path = self.image_paths[idx]
        # 跨平台路径适配
        full_path = Path(os.path.join(self.root_dir, image_path)).as_posix()

        # 加载图像
        image = Image.open(full_path)

        # 转换为RGB（处理灰度图）
        if not image.mode == 'RGB':
            image = image.convert('RGB')

        # 应用变换
        if self.transform is not None:
            image = self.transform(image)

        return image, image_path


class CommunityImageDataset(Dataset):
    """
    社区级别的图像数据集，用于提取社区平均嵌入。
    按社区分组，返回同一社区的所有图像。
    """

    def __init__(self, root_dir: str, community_images: dict, transform=None):
        """
        初始化数据集

        Args:
            root_dir: 图像根目录
            community_images: 字典，{社区名称: [图像路径列表]}
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.community_images = community_images
        self.transform = transform

        # 展平为列表以便迭代
        self.community_list = list(community_images.keys())
        self.image_indices = []
        for comm_idx, community in enumerate(self.community_list):
            for img_idx in range(len(community_images[community])):
                self.image_indices.append((comm_idx, img_idx))

    def __len__(self) -> int:
        return len(self.image_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        """
        获取一个样本

        Returns:
            image: 图像张量
            community_name: 社区名称
            image_idx: 图像在社区中的索引
        """
        comm_idx, img_idx = self.image_indices[idx]
        community = self.community_list[comm_idx]
        image_path = self.community_images[community][img_idx]

        # 跨平台路径适配
        full_path = Path(os.path.join(self.root_dir, image_path)).as_posix()

        # 加载图像
        image = Image.open(full_path)

        # 转换为RGB（处理灰度图）
        if not image.mode == 'RGB':
            image = image.convert('RGB')

        # 应用变换
        if self.transform is not None:
            image = self.transform(image)

        return image, community, img_idx

    def get_community_images(self, community: str) -> List[str]:
        """获取指定社区的所有图像路径"""
        return self.community_images.get(community, [])

    def get_all_communities(self) -> List[str]:
        """获取所有社区名称"""
        return self.community_list