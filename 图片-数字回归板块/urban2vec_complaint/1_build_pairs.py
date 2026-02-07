#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1_build_pairs.py - 训练对构建器
基于地理距离构建K近邻训练对，替代原代码的 pic_form_context.py + pic_form_pair.py。

输入：metadata.csv
输出：context_knn.pickle, train_pair_knn.pickle, val_pair_knn.pickle

关键逻辑：
- 基于Haversine球面距离计算K近邻（K=6）
- 关键去重：同一point_id的不同方位角图片不互为最近邻
- 生成锚点-正样本对，8:2划分训练/验证
"""

import argparse
import csv
import logging
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Set

import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 地球半径（公里）
EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点之间的Haversine距离（公里）

    Args:
        lat1, lon1: 第一个点的纬度和经度（度）
        lat2, lon2: 第二个点的纬度和经度（度）

    Returns:
        两点之间的球面距离（公里）
    """
    # 将角度转换为弧度
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine公式
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def load_metadata(metadata_path: str) -> List[Dict]:
    """
    加载元数据CSV文件

    Args:
        metadata_path: metadata.csv文件路径

    Returns:
        元数据列表
    """
    metadata = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数据类型
            row['longitude'] = float(row['longitude'])
            row['latitude'] = float(row['latitude'])
            row['heading'] = int(row['heading'])
            row['point_id'] = str(row['point_id'])
            metadata.append(row)

    logger.info(f"加载了 {len(metadata)} 条元数据记录")
    return metadata


def build_point_groups(metadata: List[Dict]) -> Dict[str, List[int]]:
    """
    按point_id分组，用于去重处理

    Args:
        metadata: 元数据列表

    Returns:
        字典：point_id -> 在metadata列表中的索引列表
    """
    point_groups = defaultdict(list)
    for idx, item in enumerate(metadata):
        point_groups[item['point_id']].append(idx)

    logger.info(f"识别出 {len(point_groups)} 个唯一的采样点")
    return point_groups


def find_k_neighbors(metadata: List[Dict], k: int = 6) -> Dict[str, List[str]]:
    """
    为每个图像找到K个最近邻（排除同一点的不同方位角）

    Args:
        metadata: 元数据列表
        k: 近邻数量

    Returns:
        字典：图像路径 -> 最近邻图像路径列表
    """
    n = len(metadata)
    logger.info(f"开始计算 {n} 个图像之间的距离矩阵...")

    # 提取经纬度
    lats = np.array([item['latitude'] for item in metadata])
    lons = np.array([item['longitude'] for item in metadata])

    # 预先计算point_id映射
    point_ids = [item['point_id'] for item in metadata]
    filenames = [item['filename'] for item in metadata]

    # 构建上下文关系字典
    context = {}

    # 对于大数据集，这里可以优化（如使用KD树）
    # 但考虑到街景图像通常在同一城市，数量不会太大，使用简单方法
    for i in range(n):
        if i % 100 == 0:
            logger.info(f"  处理第 {i}/{n} 个图像...")

        # 计算当前点到所有点的距离
        distances = []
        for j in range(n):
            if i == j:
                distances.append(float('inf'))  # 排除自身
            elif point_ids[i] == point_ids[j]:
                distances.append(float('inf'))  # 排除同一点的不同方位角
            else:
                dist = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                distances.append(dist)

        # 找到最近的k个邻居
        distances = np.array(distances)
        neighbor_indices = np.argsort(distances)[:k]

        # 保存邻居的文件路径
        context[filenames[i]] = [filenames[idx] for idx in neighbor_indices if distances[idx] < float('inf')]

    logger.info(f"完成K近邻计算")
    return context


def build_training_pairs(context: Dict[str, List[str]], metadata: List[Dict],
                        train_ratio: float = 0.8) -> Tuple[List, List]:
    """
    构建训练和验证对

    Args:
        context: 上下文关系字典
        metadata: 元数据列表
        train_ratio: 训练集比例

    Returns:
        (训练对列表, 验证对列表)
    """
    # 构建文件名到元数据的映射
    filename_to_metadata = {item['filename']: item for item in metadata}

    # 收集所有可能的锚点-正样本对
    all_pairs = []
    for anchor, neighbors in context.items():
        for neighbor in neighbors:
            # 获取社区信息
            anchor_community = filename_to_metadata[anchor]['community_name']
            neighbor_community = filename_to_metadata[neighbor]['community_name']

            # 只保留同一社区内的对（可选，根据需求调整）
            # if anchor_community == neighbor_community:
            all_pairs.append((anchor, neighbor, anchor_community))

    logger.info(f"总共生成 {len(all_pairs)} 个锚点-正样本对")

    # 随机打乱
    random.shuffle(all_pairs)

    # 划分训练/验证集
    split_idx = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    logger.info(f"训练集: {len(train_pairs)} 对, 验证集: {len(val_pairs)} 对")

    return train_pairs, val_pairs


def save_pickle(data, filepath: str) -> None:
    """保存数据到pickle文件"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"数据已保存到: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='构建K近邻训练对')
    parser.add_argument('--metadata', type=str, default='metadata.csv',
                       help='元数据CSV文件路径（默认：metadata.csv）')
    parser.add_argument('--k', type=int, default=6,
                       help='近邻数量K（默认：6）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例（默认：0.8）')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='输出目录（默认：当前目录）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认：42）')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始构建训练对...")
    logger.info(f"随机种子: {args.seed}, K={args.k}, 训练比例={args.train_ratio}")

    try:
        # 1. 加载元数据
        metadata = load_metadata(args.metadata)

        if not metadata:
            logger.error("元数据为空，请检查输入文件")
            sys.exit(1)

        # 2. 构建point_id分组（用于去重）
        point_groups = build_point_groups(metadata)

        # 3. 计算K近邻
        context = find_k_neighbors(metadata, k=args.k)

        # 4. 构建训练对
        train_pairs, val_pairs = build_training_pairs(context, metadata, args.train_ratio)

        # 5. 保存结果
        context_path = output_dir / 'context_knn.pickle'
        train_path = output_dir / 'train_pair_knn.pickle'
        val_path = output_dir / 'val_pair_knn.pickle'

        save_pickle(context, str(context_path))
        save_pickle(train_pairs, str(train_path))
        save_pickle(val_pairs, str(val_path))

        # 打印统计信息
        logger.info("=" * 50)
        logger.info("构建完成统计信息:")
        logger.info(f"  总图像数: {len(metadata)}")
        logger.info(f"  唯一采样点: {len(point_groups)}")
        logger.info(f"  上下文关系数: {len(context)}")
        logger.info(f"  训练对数量: {len(train_pairs)}")
        logger.info(f"  验证对数量: {len(val_pairs)}")
        logger.info(f"  输出文件:")
        logger.info(f"    - {context_path}")
        logger.info(f"    - {train_path}")
        logger.info(f"    - {val_path}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)

    logger.info("构建训练对完成")


if __name__ == '__main__':
    main()