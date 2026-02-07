#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4_train_predictor.py - 多模型投诉预测器
基于社区街景嵌入预测12345投诉数量。

输入：community_embeddings.csv + complaints.csv（按community_name关联）
输出：8个目标×4种模型的对比结果 + 最优模型文件 + 可视化图表

目标列：total_complaints, house_management, traffic_municipal, public_service,
        social_affairs, market_economy, urban_rural, OTHER
模型：Ridge（基线）、SVR（RBF）、RandomForest、XGBoost
评估：5-fold CV，记录R²、MAE、RMSE
可视化：每个目标画4张图（预测vs真实散点图、RF特征重要性Top20、PCA散点、残差图）
特征分析：输出与投诉最相关的视觉维度Top10
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 目标列定义
TARGET_COLUMNS = [
    'total_complaints',
    'house_management',
    'traffic_municipal',
    'public_service',
    'social_affairs',
    'market_economy',
    'urban_rural',
    'OTHER'
]

# 模型定义
MODELS = {
    'Ridge': Ridge(alpha=1.0),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}


def load_and_merge_data(embeddings_path: str, complaints_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载嵌入向量和投诉数据，并按社区名称合并

    Args:
        embeddings_path: 嵌入向量CSV文件路径
        complaints_path: 投诉数据CSV文件路径

    Returns:
        (特征DataFrame, 目标DataFrame)
    """
    logger.info(f"加载嵌入向量: {embeddings_path}")
    embeddings_df = pd.read_csv(embeddings_path)

    logger.info(f"加载投诉数据: {complaints_path}")
    # complaints_df = pd.read_csv(complaints_path)
    # 尝试常见的中文编码
    try:
        complaints_df = pd.read_csv(complaints_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            complaints_df = pd.read_csv(complaints_path, encoding='gbk')
        except UnicodeDecodeError:
            complaints_df = pd.read_csv(complaints_path, encoding='gb2312')

    # 检查必需的列
    if 'community_name' not in embeddings_df.columns:
        raise ValueError("嵌入向量文件必须包含'community_name'列")

    if 'community_name' not in complaints_df.columns:
        raise ValueError("投诉数据文件必须包含'community_name'列")

    # 检查目标列
    missing_targets = [col for col in TARGET_COLUMNS if col not in complaints_df.columns]
    if missing_targets:
        logger.warning(f"投诉数据缺少以下目标列: {missing_targets}")
        # 只保留存在的目标列
        available_targets = [col for col in TARGET_COLUMNS if col in complaints_df.columns]
        if not available_targets:
            raise ValueError("投诉数据中没有任何目标列")
        return X, y_df, available_targets

    # 按社区名称合并
    merged_df = pd.merge(embeddings_df, complaints_df, on='community_name', how='inner')

    if merged_df.empty:
        raise ValueError("没有找到匹配的社区数据，请检查社区名称是否一致")

    logger.info(f"合并后数据: {merged_df.shape[0]} 个样本，{merged_df.shape[1]} 个特征")

    # 分离特征和目标
    # 特征列：以'dim_'开头的列
    feature_cols = [col for col in merged_df.columns if col.startswith('dim_')]
    X = merged_df[feature_cols].values

    # 目标列
    y_dict = {}
    for target in TARGET_COLUMNS:
        if target in merged_df.columns:
            y_dict[target] = merged_df[target].values
        else:
            logger.warning(f"目标列 {target} 不存在，跳过")

    # 转换为DataFrame以便处理
    y_df = pd.DataFrame(y_dict)

    logger.info(f"特征维度: {X.shape}, 目标维度: {y_df.shape}")
    logger.info(f"可用目标: {list(y_df.columns)}")

    return X, y_df


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray, model_name: str,
                  target_name: str, cv_folds: int = 5) -> Dict[str, float]:
    """
    使用交叉验证评估模型

    Args:
        model: 模型对象
        X: 特征矩阵
        y: 目标向量
        model_name: 模型名称
        target_name: 目标名称
        cv_folds: 交叉验证折数

    Returns:
        包含评估指标的字典
    """
    logger.info(f"评估模型 {model_name} 对目标 {target_name}")

    # 初始化KFold
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # 存储每折的结果
    fold_metrics = {'r2': [], 'mae': [], 'rmse': []}

    # 用于存储所有预测值
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 训练模型
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_val_scaled)

        # 计算指标
        fold_metrics['r2'].append(r2_score(y_val, y_pred))
        fold_metrics['mae'].append(mean_absolute_error(y_val, y_pred))
        fold_metrics['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))

        # 收集所有预测值
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

    # 计算平均指标
    metrics = {
        'model': model_name,
        'target': target_name,
        'r2_mean': np.mean(fold_metrics['r2']),
        'r2_std': np.std(fold_metrics['r2']),
        'mae_mean': np.mean(fold_metrics['mae']),
        'mae_std': np.std(fold_metrics['mae']),
        'rmse_mean': np.mean(fold_metrics['rmse']),
        'rmse_std': np.std(fold_metrics['rmse']),
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred)
    }

    logger.info(f"  {target_name} - R²: {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}, "
                f"MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")

    return metrics


def create_visualizations(metrics_dict: Dict[str, Dict], X: np.ndarray,
                         feature_names: List[str], output_dir: str) -> None:
    """
    创建可视化图表

    Args:
        metrics_dict: 指标字典 {model_target: metrics}
        X: 特征矩阵
        feature_names: 特征名称列表
        output_dir: 输出目录
    """
    logger.info("创建可视化图表...")

    # 设置图形风格
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 创建输出目录
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. 模型性能比较图（所有目标）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    performance_data = []
    for key, metrics in metrics_dict.items():
        model_name, target_name = key.split('_', 1)
        performance_data.append({
            'Model': model_name,
            'Target': target_name,
            'R²': metrics['r2_mean'],
            'MAE': metrics['mae_mean'],
            'RMSE': metrics['rmse_mean']
        })

    perf_df = pd.DataFrame(performance_data)

    # R²比较
    ax = axes[0]
    pivot_r2 = perf_df.pivot(index='Target', columns='Model', values='R²')
    pivot_r2.plot(kind='bar', ax=ax)
    ax.set_title('模型R²分数比较（各目标）')
    ax.set_xlabel('目标变量')
    ax.set_ylabel('R²分数')
    ax.legend(title='模型')
    ax.tick_params(axis='x', rotation=45)

    # MAE比较
    ax = axes[1]
    pivot_mae = perf_df.pivot(index='Target', columns='Model', values='MAE')
    pivot_mae.plot(kind='bar', ax=ax)
    ax.set_title('模型MAE比较（各目标）')
    ax.set_xlabel('目标变量')
    ax.set_ylabel('MAE')
    ax.legend(title='模型')
    ax.tick_params(axis='x', rotation=45)

    # RMSE比较
    ax = axes[2]
    pivot_rmse = perf_df.pivot(index='Target', columns='Model', values='RMSE')
    pivot_rmse.plot(kind='bar', ax=ax)
    ax.set_title('模型RMSE比较（各目标）')
    ax.set_xlabel('目标变量')
    ax.set_ylabel('RMSE')
    ax.legend(title='模型')
    ax.tick_params(axis='x', rotation=45)

    # 热力图：各模型在各目标上的R²分数
    ax = axes[3]
    heatmap_data = pivot_r2.T
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_title('模型性能热力图（R²分数）')
    ax.set_xlabel('目标变量')
    ax.set_ylabel('模型')

    plt.tight_layout()
    plt.savefig(vis_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 为每个目标创建详细图表
    for target in TARGET_COLUMNS:
        target_metrics = {k: v for k, v in metrics_dict.items() if k.endswith(f'_{target}')}
        if not target_metrics:
            continue

        fig = plt.figure(figsize=(20, 16))

        # 子图1：预测vs真实散点图（所有模型）
        ax1 = plt.subplot(2, 2, 1)
        colors = plt.cm.tab10.colors
        for idx, (key, metrics) in enumerate(target_metrics.items()):
            model_name = key.split('_')[0]
            ax1.scatter(metrics['y_true'], metrics['y_pred'],
                       alpha=0.6, label=model_name, color=colors[idx % len(colors)])

        # 添加对角线
        min_val = min(metrics['y_true'].min(), metrics['y_pred'].min())
        max_val = max(metrics['y_true'].max(), metrics['y_pred'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')
        ax1.set_title(f'{target} - 预测vs真实散点图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2：RandomForest特征重要性（如果可用）
        ax2 = plt.subplot(2, 2, 2)
        rf_key = f'RandomForest_{target}'
        if rf_key in target_metrics:
            # 需要重新训练RandomForest以获取特征重要性
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            rf_model.fit(X_scaled, metrics['y_true'])  # 使用所有数据训练

            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20

            ax2.barh(range(len(indices)), importances[indices], align='center')
            ax2.set_yticks(range(len(indices)))
            ax2.set_yticklabels([feature_names[i] for i in indices])
            ax2.set_xlabel('特征重要性')
            ax2.set_title(f'{target} - RandomForest特征重要性Top20')
        else:
            ax2.text(0.5, 0.5, 'RandomForest模型不可用', ha='center', va='center')
            ax2.set_title(f'{target} - 特征重要性')

        # 子图3：PCA降维散点图（按目标值着色）
        ax3 = plt.subplot(2, 2, 3)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1],
                             c=metrics['y_true'], cmap='viridis', alpha=0.7)
        ax3.set_xlabel('PC1 (解释方差: {:.1f}%)'.format(pca.explained_variance_ratio_[0] * 100))
        ax3.set_ylabel('PC2 (解释方差: {:.1f}%)'.format(pca.explained_variance_ratio_[1] * 100))
        ax3.set_title(f'{target} - PCA降维散点图')
        plt.colorbar(scatter, ax=ax3, label='目标值')

        # 子图4：残差图（最佳模型）
        ax4 = plt.subplot(2, 2, 4)
        # 找到最佳模型（R²最高）
        best_model_key = max(target_metrics.items(), key=lambda x: x[1]['r2_mean'])[0]
        best_metrics = target_metrics[best_model_key]

        residuals = best_metrics['y_true'] - best_metrics['y_pred']
        ax4.scatter(best_metrics['y_pred'], residuals, alpha=0.6)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.set_xlabel('预测值')
        ax4.set_ylabel('残差')
        ax4.set_title(f'{target} - 残差图（最佳模型: {best_model_key.split("_")[0]}）')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(vis_dir / f'{target}_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"可视化图表已保存到: {vis_dir}")


def analyze_feature_correlation(X: np.ndarray, y_df: pd.DataFrame,
                               feature_names: List[str], output_dir: str) -> None:
    """
    分析特征与目标的相关性

    Args:
        X: 特征矩阵
        y_df: 目标DataFrame
        feature_names: 特征名称列表
        output_dir: 输出目录
    """
    logger.info("分析特征相关性...")

    # 计算Pearson相关系数
    correlation_results = {}

    for target_col in y_df.columns:
        y = y_df[target_col].values
        correlations = []

        for i, feature_name in enumerate(feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append((feature_name, corr))

        # 按绝对值排序，取Top10
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        top10 = correlations[:10]

        correlation_results[target_col] = top10

    # 保存结果
    output_path = Path(output_dir) / 'feature_correlation_top10.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("特征与投诉目标相关性分析（Top10）\n")
        f.write("=" * 60 + "\n\n")

        for target, correlations in correlation_results.items():
            f.write(f"目标: {target}\n")
            f.write("-" * 40 + "\n")
            for feature_name, corr in correlations:
                f.write(f"  {feature_name}: {corr:.4f}\n")
            f.write("\n")

    logger.info(f"相关性分析结果已保存到: {output_path}")

    # 创建热力图
    plt.figure(figsize=(12, 8))
    corr_matrix = []
    for target_col in y_df.columns:
        y = y_df[target_col].values
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(corr)
        corr_matrix.append(correlations)

    corr_matrix = np.array(corr_matrix)

    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                xticklabels=feature_names[::5],  # 每5个特征显示一个标签
                yticklabels=y_df.columns)
    plt.title('特征与目标变量相关性热力图')
    plt.xlabel('特征维度')
    plt.ylabel('目标变量')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'visualizations' / 'correlation_heatmap.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='训练投诉预测模型')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='社区嵌入向量CSV文件路径')
    parser.add_argument('--complaints', type=str, required=True,
                       help='投诉数据CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='./prediction_results',
                       help='输出目录（默认：./prediction_results）')
    parser.add_argument('--cv_folds', type=int, default=2,
                       help='交叉验证折数（默认：5）')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始投诉预测模型训练...")
    logger.info(f"嵌入向量: {args.embeddings}")
    logger.info(f"投诉数据: {args.complaints}")
    logger.info(f"输出目录: {output_dir}")

    try:
        # 1. 加载和合并数据
        X, y_df = load_and_merge_data(args.embeddings, args.complaints)

        # 获取特征名称
        feature_names = [f'dim_{i}' for i in range(X.shape[1])]

        # 2. 训练和评估模型
        metrics_dict = {}
        best_models = {}

        for model_name, model in MODELS.items():
            logger.info(f"\n训练模型: {model_name}")

            for target in y_df.columns:
                y = y_df[target].values

                # 评估模型
                metrics = evaluate_model(
                    model=model,
                    X=X,
                    y=y,
                    model_name=model_name,
                    target_name=target,
                    cv_folds=args.cv_folds
                )

                # 保存指标
                key = f"{model_name}_{target}"
                metrics_dict[key] = metrics

        # 3. 保存评估结果
        results_df = pd.DataFrame([
            {
                'Model': m['model'],
                'Target': m['target'],
                'R2_mean': m['r2_mean'],
                'R2_std': m['r2_std'],
                'MAE_mean': m['mae_mean'],
                'MAE_std': m['mae_std'],
                'RMSE_mean': m['rmse_mean'],
                'RMSE_std': m['rmse_std']
            }
            for m in metrics_dict.values()
        ])

        results_path = output_dir / 'model_evaluation_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"评估结果已保存到: {results_path}")

        # 4. 创建可视化图表
        create_visualizations(metrics_dict, X, feature_names, str(output_dir))

        # 5. 分析特征相关性
        analyze_feature_correlation(X, y_df, feature_names, str(output_dir))

        # 6. 训练最终模型（在所有数据上）并保存
        logger.info("\n训练最终模型...")
        final_models_dir = output_dir / 'final_models'
        final_models_dir.mkdir(parents=True, exist_ok=True)

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for target in y_df.columns:
            y = y_df[target].values

            # 找到最佳模型（R²最高）
            target_metrics = {k: v for k, v in metrics_dict.items() if k.endswith(f'_{target}')}
            if not target_metrics:
                continue

            best_model_key = max(target_metrics.items(), key=lambda x: x[1]['r2_mean'])[0]
            best_model_name = best_model_key.split('_')[0]

            logger.info(f"目标 {target} 的最佳模型: {best_model_name}")

            # 使用最佳模型类型重新训练
            if best_model_name == 'Ridge':
                final_model = Ridge(alpha=1.0)
            elif best_model_name == 'SVR':
                final_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            elif best_model_name == 'RandomForest':
                final_model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif best_model_name == 'XGBoost':
                final_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            else:
                logger.warning(f"未知模型类型: {best_model_name}，使用RandomForest")
                final_model = RandomForestRegressor(n_estimators=100, random_state=42)

            # 在所有数据上训练
            final_model.fit(X_scaled, y)

            # 保存模型
            import joblib
            model_path = final_models_dir / f'best_model_{target}.joblib'
            joblib.dump(final_model, model_path)
            logger.info(f"  模型已保存到: {model_path}")

        # 保存标准化器
        scaler_path = final_models_dir / 'scaler.joblib'
        joblib.dump(scaler, scaler_path)

        logger.info(f"\n最终模型已保存到: {final_models_dir}")

        # 7. 打印总结
        logger.info("=" * 60)
        logger.info("训练完成总结:")
        logger.info(f"  数据样本数: {X.shape[0]}")
        logger.info(f"  特征维度: {X.shape[1]}")
        logger.info(f"  预测目标数: {y_df.shape[1]}")
        logger.info(f"  评估结果: {results_path}")
        logger.info(f"  可视化图表: {output_dir / 'visualizations'}")
        logger.info(f"  最终模型: {final_models_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)

    logger.info("投诉预测模型训练完成")


if __name__ == '__main__':
    main()