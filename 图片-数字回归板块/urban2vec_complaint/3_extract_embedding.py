#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3_extract_embedding.py - 社区特征提取器
提取社区级别的街景图像嵌入向量。

输入：训练好的模型 + 图片文件夹
输出：community_embeddings.csv（列：community_name, dim_0...dim_n）

关键逻辑：
- 遍历社区文件夹内所有图片，提取embedding
- Mean Pooling：同一社区所有图片向量取平均，得到1×embedding_dim社区向量
- 方位角处理：4个方向的图片全部参与平均
"""

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 导入自定义模块
from src.models import PlaceImageEmb
from src.datasets import CommunityImageDataset
from src.utils import setup_logging, check_gpu_availability, parse_filename_metadata

# 配置日志
logger = setup_logging()


def load_model(model_path: str, embedding_dim: int, device: torch.device) -> PlaceImageEmb:
    """
    加载训练好的模型

    Args:
        model_path: 模型文件路径
        embedding_dim: 嵌入维度
        device: 设备

    Returns:
        加载的模型
    """
    logger.info(f"加载模型: {model_path}")

    # 创建模型
    model = PlaceImageEmb(embedding_dim=embedding_dim, pretrained=False)
    model = model.to(device)

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # 设置为评估模式
    model.eval()

    logger.info(f"模型加载完成，嵌入维度: {embedding_dim}")
    return model


def scan_community_images(image_root: str) -> Dict[str, List[str]]:
    """
    扫描图片目录，按社区分组

    Args:
        image_root: 图片根目录

    Returns:
        字典：{社区名称: [图像路径列表]}
    """
    image_root_path = Path(image_root)
    if not image_root_path.exists():
        raise ValueError(f"图片根目录不存在: {image_root}")

    community_images = defaultdict(list)
    total_images = 0
    skipped_images = 0

    # 支持多种图像格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    # 遍历所有社区文件夹
    for item in image_root_path.iterdir():
        if item.is_dir() and item.name.endswith('_images'):
            community_name = item.name.replace('_images', '')
            logger.info(f"扫描社区: {community_name}")

            # 遍历社区文件夹内的图片
            for image_file in item.glob('*'):
                if image_file.suffix.lower() in image_extensions:
                    # 解析文件名元数据
                    metadata = parse_filename_metadata(image_file.name)
                    if metadata:
                        # 添加相对路径
                        rel_path = str(image_file.relative_to(image_root_path))
                        community_images[community_name].append(rel_path)
                        total_images += 1
                    else:
                        skipped_images += 1

            logger.info(f"  找到 {len(community_images[community_name])} 张有效图片")

    logger.info(f"总共扫描 {len(community_images)} 个社区，{total_images} 张图片，跳过 {skipped_images} 张无效图片")
    return dict(community_images)


def extract_community_embeddings(model: PlaceImageEmb, community_images: Dict[str, List[str]],
                                image_root: str, batch_size: int = 32,
                                device: torch.device = torch.device('cpu')) -> Dict[str, np.ndarray]:
    """
    提取社区嵌入向量

    Args:
        model: 模型
        community_images: 社区图像字典
        image_root: 图像根目录
        batch_size: 批大小
        device: 设备

    Returns:
        字典：{社区名称: 嵌入向量}
    """
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 创建数据集
    dataset = CommunityImageDataset(image_root, community_images, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    # 初始化存储结构
    community_embeddings = defaultdict(list)
    embedding_dim = None

    logger.info("开始提取嵌入向量...")
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="提取嵌入", leave=False)
        for batch_images, batch_communities, _ in progress_bar:
            # 移动到设备
            batch_images = batch_images.to(device)

            # 提取嵌入
            batch_embeddings = model(batch_images).cpu().numpy()

            if embedding_dim is None:
                embedding_dim = batch_embeddings.shape[1]

            # 按社区分组
            for i, community in enumerate(batch_communities):
                community_embeddings[community].append(batch_embeddings[i])

    # 对每个社区的嵌入进行平均池化
    logger.info("进行平均池化...")
    final_embeddings = {}
    for community, embeddings_list in community_embeddings.items():
        if embeddings_list:
            # 堆叠所有嵌入
            embeddings_array = np.vstack(embeddings_list)
            # 计算平均值
            mean_embedding = np.mean(embeddings_array, axis=0)
            final_embeddings[community] = mean_embedding
            logger.debug(f"社区 {community}: {len(embeddings_list)} 张图片，嵌入维度 {mean_embedding.shape}")
        else:
            logger.warning(f"社区 {community} 没有有效嵌入")

    logger.info(f"完成 {len(final_embeddings)} 个社区的嵌入提取")
    return final_embeddings


def save_embeddings_csv(embeddings: Dict[str, np.ndarray], output_path: str) -> None:
    """
    将嵌入向量保存为CSV文件

    Args:
        embeddings: 嵌入向量字典
        output_path: 输出CSV文件路径
    """
    if not embeddings:
        logger.error("没有嵌入向量可保存")
        return

    # 获取嵌入维度
    sample_embedding = next(iter(embeddings.values()))
    embedding_dim = len(sample_embedding)

    # 准备CSV列名
    fieldnames = ['community_name'] + [f'dim_{i}' for i in range(embedding_dim)]

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for community, embedding in embeddings.items():
            row = {'community_name': community}
            for i in range(embedding_dim):
                row[f'dim_{i}'] = embedding[i]
            writer.writerow(row)

    logger.info(f"嵌入向量已保存到: {output_path}")
    logger.info(f"总共 {len(embeddings)} 个社区，嵌入维度 {embedding_dim}")


def main():
    parser = argparse.ArgumentParser(description='提取社区级街景嵌入向量')
    parser.add_argument('--image_root', type=str, required=True,
                       help='图片根目录，包含社区子文件夹')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型文件路径（.tar）')
    parser.add_argument('--embedding_dim', type=int, default=50,
                       help='嵌入维度（默认：50）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小（默认：32）')
    parser.add_argument('--output', type=str, default='community_embeddings.csv',
                       help='输出CSV文件路径（默认：community_embeddings.csv）')

    args = parser.parse_args()

    # 检查GPU可用性
    use_gpu, device = check_gpu_availability()
    logger.info(f"使用设备: {device}")

    try:
        # 1. 扫描社区图像
        logger.info(f"扫描图片目录: {args.image_root}")
        community_images = scan_community_images(args.image_root)

        if not community_images:
            logger.error("没有找到社区图像，请检查目录结构")
            sys.exit(1)

        # 2. 加载模型
        model = load_model(args.model_path, args.embedding_dim, device)

        # 3. 提取社区嵌入
        embeddings = extract_community_embeddings(
            model=model,
            community_images=community_images,
            image_root=args.image_root,
            batch_size=args.batch_size,
            device=device
        )

        if not embeddings:
            logger.error("没有成功提取任何嵌入向量")
            sys.exit(1)

        # 4. 保存嵌入向量
        save_embeddings_csv(embeddings, args.output)

        # 5. 打印统计信息
        logger.info("=" * 50)
        logger.info("提取完成统计信息:")
        logger.info(f"  处理社区数: {len(community_images)}")
        logger.info(f"  成功提取嵌入的社区数: {len(embeddings)}")
        logger.info(f"  嵌入维度: {args.embedding_dim}")
        logger.info(f"  输出文件: {args.output}")
        logger.info("=" * 50)

        # 6. 可选：保存为npy格式以便后续使用
        npy_output = args.output.replace('.csv', '.npy')
        np.save(npy_output, embeddings)
        logger.info(f"嵌入字典已保存为npy格式: {npy_output}")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)

    logger.info("社区嵌入提取完成")


if __name__ == '__main__':
    main()