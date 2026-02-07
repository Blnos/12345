#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/utils.py - 工具函数集合
包含距离计算、评估指标、文件处理等通用函数。
"""

import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    配置日志记录器

    Args:
        level: 日志级别

    Returns:
        配置好的日志记录器
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def set_random_seed(seed: int = 42) -> None:
    """
    设置所有随机种子以确保可复现性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点之间的Haversine距离（公里）

    Args:
        lat1, lon1: 第一个点的纬度和经度（度）
        lat2, lon2: 第二个点的纬度和经度（度）

    Returns:
        两点之间的球面距离（公里）
    """
    # 地球半径（公里）
    R = 6371.0

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

    return R * c


def save_pickle(data: Any, filepath: str) -> None:
    """
    保存数据到pickle文件

    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    从pickle文件加载数据

    Args:
        filepath: 文件路径

    Returns:
        加载的数据
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, filepath: str, indent: int = 2) -> None:
    """
    保存数据到JSON文件

    Args:
        data: 要保存的数据（字典）
        filepath: 文件路径
        indent: JSON缩进
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(filepath: str) -> Dict:
    """
    从JSON文件加载数据

    Args:
        filepath: 文件路径

    Returns:
        加载的数据（字典）
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算回归任务的评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        包含各种评估指标的字典
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mse': mean_squared_error(y_true, y_pred)
    }

    return metrics


def create_directory(dir_path: str) -> Path:
    """
    创建目录（如果不存在）

    Args:
        dir_path: 目录路径

    Returns:
        Path对象
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_gpu_availability() -> Tuple[bool, torch.device]:
    """
    检查GPU可用性并返回合适的设备

    Returns:
        (是否使用GPU, 设备对象)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        return True, device
    else:
        device = torch.device("cpu")
        print("使用CPU")
        return False, device


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    标准化特征（零均值，单位方差）

    Args:
        features: 输入特征数组

    Returns:
        标准化后的特征数组
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def parse_filename_metadata(filename: str) -> Optional[Dict[str, Any]]:
    """
    解析图片文件名中的元数据

    Args:
        filename: 图片文件名（如"0_左家庄街道_110105005000_116.4_39.9_0_0.png"）

    Returns:
        包含解析后元数据的字典，如果解析失败则返回None
    """
    import os

    # 移除扩展名
    basename = os.path.splitext(filename)[0]

    # 按'_'分割
    parts = basename.split('_')

    # 预期有7个部分：point_id, community_name, tract_fips, longitude, latitude, heading, pitch
    if len(parts) != 7:
        return None

    try:
        point_id = parts[0]
        community_name = parts[1]
        tract_fips = parts[2]
        longitude = float(parts[3])
        latitude = float(parts[4])
        heading = int(parts[5])
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

    except ValueError:
        return None