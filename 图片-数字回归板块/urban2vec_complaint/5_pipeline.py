#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5_pipeline.py - 一键流程脚本
支持训练全流程和新社区预测两种模式。

使用方式：
# 训练全流程
python pipeline.py --mode train --image_root ./pics --label_path ./complaints.csv --output_dir ./output

# 预测新社区
python pipeline.py --mode predict --new_community_dir ./新社区/ --model_path ./output/best_model.tar
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class Urban2VecPipeline:
    """Urban2Vec投诉预测流程管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化流程管理器

        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        if config_path and Path(config_path).exists():
            self.config = self.load_config(config_path)
        else:
            # 使用默认配置
            self.config = self.get_default_config()

        # 设置日志级别
        log_level = getattr(logging, self.config['general']['log_level'])
        logger.setLevel(log_level)

        # 设置随机种子
        import random
        import numpy as np
        import torch
        seed = self.config['general']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        logger.info(f"Urban2Vec管道初始化完成，随机种子: {seed}")

    @staticmethod
    def load_config(config_path: str) -> Dict:
        """加载YAML配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {config_path}")
        return config

    @staticmethod
    def get_default_config() -> Dict:
        """获取默认配置"""
        return {
            'paths': {
                'image_root': './街景图片/',
                'complaints_csv': './complaints.csv',
                'metadata_csv': './metadata.csv',
                'context_pickle': './context_knn.pickle',
                'train_pairs_pickle': './train_pair_knn.pickle',
                'val_pairs_pickle': './val_pair_knn.pickle',
                'embeddings_csv': './community_embeddings.csv',
                'model_checkpoint': './checkpoints/street_view_best.tar',
                'prediction_results': './prediction_results'
            },
            'stage1_training': {
                'embedding_dim': 50,
                'batch_size': 32,
                'num_epochs': 20,
                'learning_rate': 0.0005,
                'margin': 5.0,
                'num_workers': 4
            },
            'pair_building': {
                'k_neighbors': 6,
                'train_ratio': 0.8
            },
            'prediction': {
                'cv_folds': 5,
                'models': ['Ridge', 'SVR', 'RandomForest', 'XGBoost'],
                'target_columns': [
                    'total_complaints',
                    'house_management',
                    'traffic_municipal',
                    'public_service',
                    'social_affairs',
                    'market_economy',
                    'urban_rural',
                    'OTHER'
                ]
            },
            'general': {
                'seed': 42,
                'device': 'auto',
                'log_level': 'INFO'
            }
        }

    def run_command(self, cmd: List[str], step_name: str) -> bool:
        """
        运行命令行命令

        Args:
            cmd: 命令列表
            step_name: 步骤名称

        Returns:
            是否成功
        """
        logger.info(f"开始执行步骤: {step_name}")
        logger.debug(f"命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"步骤完成: {step_name}")
            if result.stdout:
                logger.debug(f"输出: {result.stdout[:500]}...")  # 只显示前500字符
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"步骤失败: {step_name}")
            logger.error(f"错误输出: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"执行步骤时发生异常: {step_name} - {e}")
            return False

    def run_training_pipeline(self, image_root: str, label_path: str, output_dir: str) -> bool:
        """
        运行训练全流程

        Args:
            image_root: 图片根目录
            label_path: 标签文件路径
            output_dir: 输出目录

        Returns:
            是否成功
        """
        logger.info("=" * 60)
        logger.info("开始训练全流程")
        logger.info(f"图片根目录: {image_root}")
        logger.info(f"标签文件: {label_path}")
        logger.info(f"输出目录: {output_dir}")
        logger.info("=" * 60)

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 更新配置中的路径
        self.config['paths']['image_root'] = image_root
        self.config['paths']['complaints_csv'] = label_path
        self.config['paths']['metadata_csv'] = str(output_path / 'metadata.csv')
        self.config['paths']['context_pickle'] = str(output_path / 'context_knn.pickle')
        self.config['paths']['train_pairs_pickle'] = str(output_path / 'train_pair_knn.pickle')
        self.config['paths']['val_pairs_pickle'] = str(output_path / 'val_pair_knn.pickle')
        self.config['paths']['embeddings_csv'] = str(output_path / 'community_embeddings.csv')
        self.config['paths']['model_checkpoint'] = str(output_path / 'checkpoints' / 'street_view_best.tar')
        self.config['paths']['prediction_results'] = str(output_path / 'prediction_results')

        # 步骤0: 数据适配
        cmd = [
            sys.executable, '0_adapt_data.py',
            '--image_root', image_root,
            '--output', self.config['paths']['metadata_csv'],
            '--seed', str(self.config['general']['seed'])
        ]
        if not self.run_command(cmd, "数据适配"):
            return False

        # 步骤1: 构建训练对
        cmd = [
            sys.executable, '1_build_pairs.py',
            '--metadata', self.config['paths']['metadata_csv'],
            '--k', str(self.config['pair_building']['k_neighbors']),
            '--train_ratio', str(self.config['pair_building']['train_ratio']),
            '--output_dir', str(output_path),
            '--seed', str(self.config['general']['seed'])
        ]
        if not self.run_command(cmd, "构建训练对"):
            return False

        # 步骤2: 训练Stage1模型
        checkpoint_dir = output_path / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, '2_train_stage1.py',
            '--image_root', image_root,
            '--train_pairs', self.config['paths']['train_pairs_pickle'],
            '--val_pairs', self.config['paths']['val_pairs_pickle'],
            '--output_dir', str(checkpoint_dir),
            '--embedding_dim', str(self.config['stage1_training']['embedding_dim']),
            '--batch_size', str(self.config['stage1_training']['batch_size']),
            '--num_epochs', str(self.config['stage1_training']['num_epochs']),
            '--learning_rate', str(self.config['stage1_training']['learning_rate']),
            '--margin', str(self.config['stage1_training']['margin']),
            '--num_workers', str(self.config['stage1_training']['num_workers']),
            '--seed', str(self.config['general']['seed'])
        ]
        if not self.run_command(cmd, "训练Stage1模型"):
            return False

        # 步骤3: 提取社区嵌入
        cmd = [
            sys.executable, '3_extract_embedding.py',
            '--image_root', image_root,
            '--model_path', self.config['paths']['model_checkpoint'],
            '--embedding_dim', str(self.config['stage1_training']['embedding_dim']),
            '--batch_size', str(self.config['stage1_training']['batch_size']),
            '--output', self.config['paths']['embeddings_csv']
        ]
        if not self.run_command(cmd, "提取社区嵌入"):
            return False

        # 步骤4: 训练投诉预测器
        prediction_results_dir = output_path / 'prediction_results'
        prediction_results_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, '4_train_predictor.py',
            '--embeddings', self.config['paths']['embeddings_csv'],
            '--complaints', label_path,
            '--output_dir', str(prediction_results_dir),
            '--cv_folds', str(self.config['prediction']['cv_folds'])
        ]
        if not self.run_command(cmd, "训练投诉预测器"):
            return False

        logger.info("=" * 60)
        logger.info("训练全流程完成!")
        logger.info(f"所有输出文件保存在: {output_path}")
        logger.info(f"模型文件: {self.config['paths']['model_checkpoint']}")
        logger.info(f"预测结果: {prediction_results_dir}")
        logger.info("=" * 60)

        return True

    def run_prediction_pipeline(self, new_community_dir: str, model_path: str,
                               output_dir: Optional[str] = None) -> bool:
        """
        运行新社区预测流程

        Args:
            new_community_dir: 新社区图片目录
            model_path: 训练好的模型文件路径
            output_dir: 输出目录，如果为None则使用当前目录

        Returns:
            是否成功
        """
        logger.info("=" * 60)
        logger.info("开始新社区预测流程")
        logger.info(f"新社区目录: {new_community_dir}")
        logger.info(f"模型文件: {model_path}")
        logger.info("=" * 60)

        # 设置输出目录
        if output_dir is None:
            output_dir = Path.cwd() / 'predictions'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 步骤1: 为新社区提取嵌入
        # 首先需要为新社区生成元数据
        metadata_path = output_dir / 'new_community_metadata.csv'

        cmd = [
            sys.executable, '0_adapt_data.py',
            '--image_root', new_community_dir,
            '--output', str(metadata_path),
            '--seed', str(self.config['general']['seed'])
        ]
        if not self.run_command(cmd, "为新社区生成元数据"):
            return False

        # 步骤2: 提取社区嵌入
        embeddings_path = output_dir / 'new_community_embeddings.csv'

        # 需要从模型文件中推断嵌入维度
        # 这里简化处理，使用配置文件中的维度
        embedding_dim = self.config['stage1_training']['embedding_dim']

        cmd = [
            sys.executable, '3_extract_embedding.py',
            '--image_root', new_community_dir,
            '--model_path', model_path,
            '--embedding_dim', str(embedding_dim),
            '--batch_size', str(self.config['stage1_training']['batch_size']),
            '--output', str(embeddings_path)
        ]
        if not self.run_command(cmd, "提取新社区嵌入"):
            return False

        # 步骤3: 加载预测模型并进行预测
        # 这里需要加载步骤4训练好的预测模型
        # 假设预测模型保存在prediction_results/final_models目录下
        prediction_models_dir = Path(model_path).parent.parent / 'prediction_results' / 'final_models'
        if not prediction_models_dir.exists():
            logger.error(f"找不到预测模型目录: {prediction_models_dir}")
            logger.error("请确保已运行训练流程并生成了预测模型")
            return False

        # 加载嵌入向量
        import pandas as pd
        import numpy as np
        import joblib

        embeddings_df = pd.read_csv(embeddings_path)
        feature_cols = [col for col in embeddings_df.columns if col.startswith('dim_')]
        X_new = embeddings_df[feature_cols].values

        # 加载标准化器
        scaler_path = prediction_models_dir / 'scaler.joblib'
        if not scaler_path.exists():
            logger.error(f"找不到标准化器: {scaler_path}")
            return False

        scaler = joblib.load(scaler_path)
        X_new_scaled = scaler.transform(X_new)

        # 为每个目标进行预测
        predictions = {}
        target_columns = self.config['prediction']['target_columns']

        for target in target_columns:
            model_path_target = prediction_models_dir / f'best_model_{target}.joblib'
            if not model_path_target.exists():
                logger.warning(f"找不到目标 {target} 的预测模型，跳过")
                continue

            model = joblib.load(model_path_target)
            y_pred = model.predict(X_new_scaled)

            # 对于单个社区，取平均值
            if len(y_pred) > 0:
                predictions[target] = float(np.mean(y_pred))
            else:
                predictions[target] = 0.0

            logger.info(f"目标 {target} 预测值: {predictions[target]:.4f}")

        # 保存预测结果
        predictions_df = pd.DataFrame([predictions])
        predictions_csv = output_dir / 'community_predictions.csv'
        predictions_df.to_csv(predictions_csv, index=False)

        # 保存详细结果
        detailed_results = []
        for i, community in enumerate(embeddings_df['community_name']):
            community_result = {'community_name': community}
            for target in predictions.keys():
                if len(X_new_scaled) > i:
                    # 为每个社区的每个目标进行预测
                    model_path_target = prediction_models_dir / f'best_model_{target}.joblib'
                    if model_path_target.exists():
                        model = joblib.load(model_path_target)
                        y_pred_single = model.predict(X_new_scaled[i:i+1])
                        community_result[target] = float(y_pred_single[0])
            detailed_results.append(community_result)

        detailed_df = pd.DataFrame(detailed_results)
        detailed_csv = output_dir / 'detailed_predictions.csv'
        detailed_df.to_csv(detailed_csv, index=False)

        logger.info("=" * 60)
        logger.info("新社区预测完成!")
        logger.info(f"预测结果已保存到: {predictions_csv}")
        logger.info(f"详细预测结果: {detailed_csv}")
        logger.info("预测结果摘要:")
        for target, value in predictions.items():
            logger.info(f"  {target}: {value:.4f}")
        logger.info("=" * 60)

        return True


def main():
    parser = argparse.ArgumentParser(description='Urban2Vec投诉预测一键流程')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                       help='运行模式：train（训练）或 predict（预测）')
    parser.add_argument('--image_root', type=str,
                       help='图片根目录（训练模式必需）')
    parser.add_argument('--label_path', type=str,
                       help='投诉数据CSV文件路径（训练模式必需）')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录（默认：./output）')
    parser.add_argument('--new_community_dir', type=str,
                       help='新社区图片目录（预测模式必需）')
    parser.add_argument('--model_path', type=str,
                       help='训练好的模型文件路径（预测模式必需）')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径（默认：config.yaml）')

    args = parser.parse_args()

    # 检查参数
    if args.mode == 'train':
        if not args.image_root or not args.label_path:
            parser.error("训练模式需要 --image_root 和 --label_path 参数")
    elif args.mode == 'predict':
        if not args.new_community_dir or not args.model_path:
            parser.error("预测模式需要 --new_community_dir 和 --model_path 参数")

    # 创建管道
    pipeline = Urban2VecPipeline(args.config)

    start_time = time.time()

    if args.mode == 'train':
        success = pipeline.run_training_pipeline(
            image_root=args.image_root,
            label_path=args.label_path,
            output_dir=args.output_dir
        )
    else:  # predict
        success = pipeline.run_prediction_pipeline(
            new_community_dir=args.new_community_dir,
            model_path=args.model_path,
            output_dir=args.output_dir
        )

    elapsed_time = time.time() - start_time
    logger.info(f"总运行时间: {elapsed_time:.2f} 秒")

    if not success:
        logger.error("流程执行失败")
        sys.exit(1)
    else:
        logger.info("流程执行成功")


if __name__ == '__main__':
    main()