#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2_train_stage1.py - 街景嵌入训练脚本
基于对比学习训练街景图像嵌入模型（Urban2Vec Stage1）。

输入：train_pair_knn.pickle, val_pair_knn.pickle
输出：best_model.tar（含200维或50维嵌入模型）

关键参数：
- 模型：Inception-v3（ImageNet预训练）→ 2048维 → Linear → embedding_dim（默认50）
- 损失：MarginRankingLoss(margin=5)，L2欧氏距离
- 训练逻辑：anchor与positive距离应比anchor与negative（batch内随机打乱）距离小至少5个单位
"""

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 导入自定义模块
from src.models import PlaceImageSkipGram
from src.datasets import PlaceImagePairDataset
from src.utils import setup_logging, set_random_seed, check_gpu_availability

# 配置日志
logger = setup_logging()


def load_training_pairs(train_path: str, val_path: str) -> Tuple[List, List]:
    """
    加载训练和验证对

    Args:
        train_path: 训练对pickle文件路径
        val_path: 验证对pickle文件路径

    Returns:
        (训练对列表, 验证对列表)
    """
    with open(train_path, 'rb') as f:
        train_pairs = pickle.load(f)

    with open(val_path, 'rb') as f:
        val_pairs = pickle.load(f)

    logger.info(f"加载训练对: {len(train_pairs)} 个")
    logger.info(f"加载验证对: {len(val_pairs)} 个")

    return train_pairs, val_pairs


def create_data_loaders(train_pairs: List, val_pairs: List, image_root: str,
                       batch_size: int = 32, num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    创建训练和验证数据加载器

    Args:
        train_pairs: 训练对列表
        val_pairs: 验证对列表
        image_root: 图像根目录
        batch_size: 批大小
        num_workers: 数据加载工作线程数

    Returns:
        包含'train'和'val'数据加载器的字典
    """
    # 图像变换
    transform_train = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 创建数据集
    train_dataset = PlaceImagePairDataset(image_root, train_pairs, transform_train)
    val_dataset = PlaceImagePairDataset(image_root, val_pairs, transform_val)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"训练集大小: {len(train_dataset)}，批大小: {batch_size}")
    logger.info(f"验证集大小: {len(val_dataset)}，批大小: {batch_size}")

    return {'train': train_loader, 'val': val_loader}


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
               optimizer: optim.Optimizer, device: torch.device, margin: float = 5.0) -> Dict[str, float]:
    """
    训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        margin: MarginRankingLoss的边界值

    Returns:
        包含训练指标的字典
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    # 用于计算距离
    distance_fn = nn.PairwiseDistance(p=2)

    progress_bar = tqdm(dataloader, desc="训练", leave=False)
    for batch_idx, (anchor_imgs, pos_imgs, _, _) in enumerate(progress_bar):
        # 移动到设备
        anchor_imgs = anchor_imgs.to(device)
        pos_imgs = pos_imgs.to(device)

        batch_size = anchor_imgs.size(0)

        # 前向传播
        anchor_emb = model(anchor_imgs)
        pos_emb = model(pos_imgs)

        # 计算正样本距离
        pos_dist = distance_fn(anchor_emb, pos_emb)

        # 创建负样本（batch内随机打乱）
        neg_indices = torch.randperm(batch_size).to(device)
        neg_emb = pos_emb[neg_indices]
        neg_dist = distance_fn(anchor_emb, neg_emb)

        # 计算损失
        target = torch.ones(batch_size).to(device)  # 我们希望pos_dist < neg_dist
        loss = criterion(neg_dist, pos_dist, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * batch_size
        running_correct += torch.sum(pos_dist < neg_dist).item()
        total_samples += batch_size

        # 更新进度条
        progress_bar.set_postfix({
            'loss': running_loss / total_samples,
            'acc': running_correct / total_samples
        })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return {'loss': epoch_loss, 'accuracy': epoch_acc}


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                  device: torch.device, margin: float = 5.0) -> Dict[str, float]:
    """
    验证一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        margin: MarginRankingLoss的边界值

    Returns:
        包含验证指标的字典
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    # 用于计算距离
    distance_fn = nn.PairwiseDistance(p=2)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="验证", leave=False)
        for batch_idx, (anchor_imgs, pos_imgs, _, _) in enumerate(progress_bar):
            # 移动到设备
            anchor_imgs = anchor_imgs.to(device)
            pos_imgs = pos_imgs.to(device)

            batch_size = anchor_imgs.size(0)

            # 前向传播
            anchor_emb = model(anchor_imgs)
            pos_emb = model(pos_imgs)

            # 计算正样本距离
            pos_dist = distance_fn(anchor_emb, pos_emb)

            # 创建负样本（batch内随机打乱）
            neg_indices = torch.randperm(batch_size).to(device)
            neg_emb = pos_emb[neg_indices]
            neg_dist = distance_fn(anchor_emb, neg_emb)

            # 计算损失
            target = torch.ones(batch_size).to(device)
            loss = criterion(neg_dist, pos_dist, target)

            # 统计
            running_loss += loss.item() * batch_size
            running_correct += torch.sum(pos_dist < neg_dist).item()
            total_samples += batch_size

            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / total_samples,
                'acc': running_correct / total_samples
            })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return {'loss': epoch_loss, 'accuracy': epoch_acc}


def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader], criterion: nn.Module,
               optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
               device: torch.device, num_epochs: int, margin: float,
               output_dir: str, model_name: str = 'street_view') -> nn.Module:
    """
    训练模型主函数

    Args:
        model: 模型
        dataloaders: 数据加载器字典
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        num_epochs: 训练轮数
        margin: MarginRankingLoss边界值
        output_dir: 输出目录
        model_name: 模型名称

    Returns:
        训练好的模型
    """
    since = time.time()

    # 保存最佳模型
    best_model_wts = model.state_dict().copy()
    best_val_acc = 0.0
    best_epoch = 0

    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 40)

        # 训练阶段
        train_metrics = train_epoch(model, dataloaders['train'], criterion, optimizer, device, margin)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])

        # 验证阶段
        val_metrics = validate_epoch(model, dataloaders['val'], criterion, device, margin)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 更新学习率
        scheduler.step()

        # 打印结果
        logger.info(f"训练 - 损失: {train_metrics['loss']:.4f}, 准确率: {train_metrics['accuracy']:.4f}")
        logger.info(f"验证 - 损失: {val_metrics['loss']:.4f}, 准确率: {val_metrics['accuracy']:.4f}")
        logger.info(f"学习率: {history['lr'][-1]:.6f}")

        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            best_model_wts = model.state_dict().copy()
            logger.info(f"新的最佳验证准确率: {best_val_acc:.4f}")

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(output_dir, f"{model_name}_epoch{epoch + 1}.tar")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'best_val_acc': best_val_acc,
                'history': history
            }, checkpoint_path)
            logger.info(f"检查点保存到: {checkpoint_path}")

        logger.info("")

    time_elapsed = time.time() - since
    logger.info(f"训练完成，用时 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒")
    logger.info(f"最佳验证准确率: {best_val_acc:.4f} (epoch {best_epoch})")

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 保存最终模型
    final_model_path = os.path.join(output_dir, f"{model_name}_best.tar")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'history': history,
        'config': {
            'embedding_dim': model.linear1.out_features,
            'margin': margin,
            'num_epochs': num_epochs
        }
    }, final_model_path)
    logger.info(f"最佳模型保存到: {final_model_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='训练街景嵌入模型（Stage1）')
    parser.add_argument('--image_root', type=str, required=True,
                       help='图像根目录')
    parser.add_argument('--train_pairs', type=str, default='train_pair_knn.pickle',
                       help='训练对pickle文件路径（默认：train_pair_knn.pickle）')
    parser.add_argument('--val_pairs', type=str, default='val_pair_knn.pickle',
                       help='验证对pickle文件路径（默认：val_pair_knn.pickle）')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='输出目录（默认：./checkpoints）')
    parser.add_argument('--embedding_dim', type=int, default=50,
                       help='嵌入维度（默认：50）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小（默认：32）')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='训练轮数（默认：20）')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='学习率（默认：0.0005）')
    parser.add_argument('--margin', type=float, default=5.0,
                       help='MarginRankingLoss边界值（默认：5.0）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作线程数（默认：4）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认：42）')

    args = parser.parse_args()

    # 设置随机种子
    set_random_seed(args.seed)

    # 检查GPU可用性
    use_gpu, device = check_gpu_availability()
    logger.info(f"使用设备: {device}")
    logger.info(f"随机种子: {args.seed}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 加载训练对
        logger.info("加载训练对...")
        train_pairs, val_pairs = load_training_pairs(args.train_pairs, args.val_pairs)

        if not train_pairs or not val_pairs:
            logger.error("训练对或验证对为空")
            sys.exit(1)

        # 2. 创建数据加载器
        logger.info("创建数据加载器...")
        dataloaders = create_data_loaders(
            train_pairs, val_pairs, args.image_root,
            batch_size=args.batch_size, num_workers=args.num_workers
        )

        # 3. 创建模型
        logger.info(f"创建模型，嵌入维度: {args.embedding_dim}")
        model = PlaceImageSkipGram(embedding_dim=args.embedding_dim, pretrained=True)
        model = model.to(device)

        # 4. 定义损失函数和优化器
        criterion = nn.MarginRankingLoss(margin=args.margin)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数总数: {total_params:,}")
        logger.info(f"可训练参数数: {trainable_params:,}")

        # 5. 训练模型
        logger.info("开始训练...")
        model = train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.num_epochs,
            margin=args.margin,
            output_dir=str(output_dir),
            model_name='street_view'
        )

        logger.info("训练完成！")

    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()