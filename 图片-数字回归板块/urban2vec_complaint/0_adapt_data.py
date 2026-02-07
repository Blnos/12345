#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_adapt_data.py - 文件名解析器
从图片路径提取元数据，生成Urban2Vec所需CSV。

输入：--image_root ./街景图片/
输出：metadata.csv（列：filename, latitude, longitude, tract_fips, community_name, point_id, heading）

文件名格式：{路网点序号}_{社区名}_{社区ID}_{经度}_{纬度}_{方位角}_{俯仰角}.png
示例：0_左家庄街道_110105005000_116.4_39.9_0_0.png
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    解析图片文件名，提取元数据

    Args:
        filename: 图片文件名（如"0_左家庄街道_110105005000_116.4_39.9_0_0.png"）

    Returns:
        包含解析后元数据的字典，如果解析失败则返回None
    """
    # 移除扩展名
    basename = os.path.splitext(filename)[0]

    # 按'_'分割
    parts = basename.split('_')

    # 预期有7个部分：point_id, community_name, tract_fips, longitude, latitude, heading, pitch
    if len(parts) != 7:
        logger.warning(f"文件名格式不正确，期望7个部分但得到{len(parts)}个: {filename}")
        return None

    try:
        point_id = parts[0]
        community_name = parts[1]
        tract_fips = parts[2]

        # 验证经纬度是否为数字
        longitude = float(parts[3])
        latitude = float(parts[4])

        # 方位角应该是整数
        heading = int(parts[5])

        # 俯仰角（可选，不用于后续处理）
        pitch = int(parts[6])

        return {
            'filename': filename,
            'point_id': point_id,
            'community_name': community_name,
            'tract_fips': tract_fips,
            'longitude': longitude,
            'latitude': latitude,
            'heading': heading,
            'pitch': pitch
        }

    except ValueError as e:
        logger.warning(f"解析数值失败 {filename}: {e}")
        return None
    except Exception as e:
        logger.warning(f"解析文件名失败 {filename}: {e}")
        return None


def scan_image_directory(image_root: str) -> List[Dict[str, str]]:
    """
    扫描图片目录，解析所有图片文件名

    Args:
        image_root: 图片根目录

    Returns:
        包含所有有效图片元数据的列表
    """
    image_root_path = Path(image_root)
    if not image_root_path.exists():
        raise ValueError(f"图片根目录不存在: {image_root}")

    metadata_list = []
    skipped_count = 0

    # 支持多种图像格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    # 遍历所有社区文件夹
    for item in image_root_path.iterdir():
        if item.is_dir() and item.name.endswith('_images'):
            community_name = item.name.replace('_images', '')
            logger.info(f"处理社区: {community_name}")

            community_image_count = 0
            # 遍历社区文件夹内的图片
            for image_file in item.glob('*'):
                if image_file.suffix.lower() in image_extensions:
                    metadata = parse_filename(image_file.name)
                    if metadata:
                        # 更新社区名（从文件名解析的社区名可能更准确）
                        metadata['community_name'] = community_name
                        # 添加相对路径
                        rel_path = str(image_file.relative_to(image_root_path))
                        metadata['filename'] = rel_path
                        metadata_list.append(metadata)
                        community_image_count += 1
                    else:
                        skipped_count += 1

            logger.info(f"  找到 {community_image_count} 张有效图片")

    logger.info(f"总共解析 {len(metadata_list)} 张图片，跳过 {skipped_count} 张无效图片")
    return metadata_list


def save_metadata_csv(metadata_list: List[Dict[str, str]], output_path: str) -> None:
    """
    将元数据保存为CSV文件

    Args:
        metadata_list: 元数据列表
        output_path: 输出CSV路径
    """
    if not metadata_list:
        logger.error("没有有效的元数据可保存")
        return

    # 定义CSV列顺序
    fieldnames = ['filename', 'point_id', 'community_name', 'tract_fips',
                  'longitude', 'latitude', 'heading', 'pitch']

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)

    logger.info(f"元数据已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='从图片文件名提取元数据')
    parser.add_argument('--image_root', type=str, required=True,
                       help='图片根目录，包含社区子文件夹（如images/）')
    parser.add_argument('--output', type=str, default='metadata.csv',
                       help='输出CSV文件路径（默认：metadata.csv）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认：42）')

    args = parser.parse_args()

    # 设置随机种子
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"开始处理图片目录: {args.image_root}")
    logger.info(f"随机种子: {args.seed}")

    try:
        # 扫描目录并解析文件名
        metadata_list = scan_image_directory(args.image_root)

        if not metadata_list:
            logger.error("没有找到有效图片，请检查目录结构和文件名格式")
            sys.exit(1)

        # 保存CSV
        save_metadata_csv(metadata_list, args.output)

        # 打印统计信息
        communities = set(m['community_name'] for m in metadata_list)
        points = set(m['point_id'] for m in metadata_list)
        logger.info(f"统计信息: {len(communities)} 个社区, {len(points)} 个采样点, {len(metadata_list)} 张图片")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        sys.exit(1)

    logger.info("处理完成")


if __name__ == '__main__':
    main()