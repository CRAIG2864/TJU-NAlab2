"""
Plotting and visualization module
Uses matplotlib for creating figures
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def plot_interpolation(x_sample, y_sample, x_dense, y_true, y_interp, 
                       title, save_path, method_name="Interpolation"):
    """
    绘制插值结果对比图
    
    Args:
        x_sample: list - 采样点x坐标
        y_sample: list - 采样点y坐标
        x_dense: list - 密集点x坐标（用于绘制曲线）
        y_true: list - 真实函数值
        y_interp: list - 插值函数值
        title: str - 图标题
        save_path: str - 保存路径
        method_name: str - 方法名称
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制真实函数
    plt.plot(x_dense, y_true, 'b-', linewidth=2, label='True Function', alpha=0.7)
    
    # 绘制插值函数
    plt.plot(x_dense, y_interp, 'r--', linewidth=2, label=f'{method_name}', alpha=0.7)
    
    # 绘制采样点
    plt.scatter(x_sample, y_sample, color='green', s=100, zorder=5, 
                label='Sample Points', marker='o', edgecolors='black')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {save_path}")


def plot_interpolation_comparison(results_dict, save_path):
    """
    绘制多种插值方法的对比图
    
    Args:
        results_dict: dict - 包含不同方法的结果
            {
                'method_name': {
                    'x_sample': [...],
                    'y_sample': [...],
                    'x_dense': [...],
                    'y_true': [...],
                    'y_interp': [...]
                },
                ...
            }
        save_path: str - 保存路径
    """
    n_methods = len(results_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, data) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        # 绘制真实函数
        ax.plot(data['x_dense'], data['y_true'], 'b-', linewidth=2, 
                label='True Function', alpha=0.7)
        
        # 绘制插值函数
        ax.plot(data['x_dense'], data['y_interp'], 'r--', linewidth=2, 
                label='Interpolation', alpha=0.7)
        
        # 绘制采样点
        ax.scatter(data['x_sample'], data['y_sample'], color='green', s=80, 
                  zorder=5, label='Sample Points', marker='o', edgecolors='black')
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {save_path}")


def plot_fitting(x_data, y_data, x_dense, y_true, y_fit, title, save_path,
                 show_residuals=False):
    """
    绘制曲线拟合结果
    
    Args:
        x_data: list - 拟合数据点x坐标
        y_data: list - 拟合数据点y坐标
        x_dense: list - 密集点x坐标
        y_true: list - 真实函数值
        y_fit: list - 拟合函数值
        title: str - 图标题
        save_path: str - 保存路径
        show_residuals: bool - 是否显示残差
    """
    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # 主图：拟合结果
    ax1.plot(x_dense, y_true, 'b-', linewidth=2, label='True Function', alpha=0.7)
    ax1.plot(x_dense, y_fit, 'r--', linewidth=2, label='Fitted Curve', alpha=0.7)
    ax1.scatter(x_data, y_data, color='green', s=50, zorder=5, 
                label='Data Points', marker='o', alpha=0.6)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 残差图
    if show_residuals and len(x_data) == len(x_dense):
        residuals = [y_data[i] - y_fit[i] for i in range(len(x_data))]
        ax2.scatter(x_data, residuals, color='purple', s=30, alpha=0.6)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {save_path}")


def plot_piecewise_fitting(x_data, y_data, x_dense, y_true, y_fit_segments, 
                          intervals, title, save_path):
    """
    绘制分段拟合结果
    
    Args:
        x_data: list - 数据点x坐标
        y_data: list - 数据点y坐标
        x_dense: list - 密集点x坐标
        y_true: list - 真实函数值
        y_fit_segments: list - 分段拟合结果列表
        intervals: list - 区间列表 [(a1,b1), (a2,b2), ...]
        title: str - 图标题
        save_path: str - 保存路径
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制真实函数
    plt.plot(x_dense, y_true, 'b-', linewidth=2.5, label='True Function', alpha=0.7)
    
    # 绘制分段拟合结果（不同颜色）
    colors = plt.cm.Set1.colors
    for i, (y_segment, interval) in enumerate(zip(y_fit_segments, intervals)):
        # 找到对应区间的密集点索引
        mask = [(x >= interval[0] and x <= interval[1]) for x in x_dense]
        x_segment = [x_dense[j] for j in range(len(x_dense)) if mask[j]]
        
        label = f'Segment {i+1} [{interval[0]:.2f}, {interval[1]:.2f}]'
        plt.plot(x_segment, y_segment, '--', linewidth=2, 
                color=colors[i % len(colors)], label=label, alpha=0.8)
    
    # 绘制数据点
    plt.scatter(x_data, y_data, color='green', s=40, zorder=5, 
                label='Data Points', marker='o', alpha=0.6, edgecolors='black')
    
    # 绘制区间分界线
    for interval in intervals[1:]:
        plt.axvline(x=interval[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {save_path}")


def plot_ransac_comparison(x_data, y_data, x_dense, y_true, y_fit_standard, 
                          y_fit_ransac, inliers, title, save_path):
    """
    绘制RANSAC拟合对比图
    
    Args:
        x_data: list - 数据点x坐标
        y_data: list - 数据点y坐标（含噪声）
        x_dense: list - 密集点x坐标
        y_true: list - 真实函数值
        y_fit_standard: list - 标准最小二乘拟合
        y_fit_ransac: list - RANSAC拟合
        inliers: list - 内点索引列表
        title: str - 图标题
        save_path: str - 保存路径
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制真实函数
    plt.plot(x_dense, y_true, 'b-', linewidth=2.5, label='True Function', alpha=0.7)
    
    # 绘制拟合曲线
    plt.plot(x_dense, y_fit_standard, 'orange', linestyle='--', linewidth=2, 
            label='Standard Least Squares', alpha=0.7)
    plt.plot(x_dense, y_fit_ransac, 'r-', linewidth=2.5, 
            label='RANSAC Least Squares', alpha=0.8)
    
    # 绘制数据点（区分内点和外点）
    outliers = [i for i in range(len(x_data)) if i not in inliers]
    
    if inliers:
        x_inliers = [x_data[i] for i in inliers]
        y_inliers = [y_data[i] for i in inliers]
        plt.scatter(x_inliers, y_inliers, color='green', s=60, zorder=5, 
                   label=f'Inliers ({len(inliers)})', marker='o', alpha=0.7)
    
    if outliers:
        x_outliers = [x_data[i] for i in outliers]
        y_outliers = [y_data[i] for i in outliers]
        plt.scatter(x_outliers, y_outliers, color='red', s=80, zorder=5, 
                   label=f'Outliers ({len(outliers)})', marker='x', linewidths=2)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {save_path}")


def plot_error_matrix(error_matrix, row_labels, col_labels, title, save_path):
    """
    绘制误差矩阵热力图
    
    Args:
        error_matrix: list of lists - 误差矩阵
        row_labels: list - 行标签
        col_labels: list - 列标签
        title: str - 图标题
        save_path: str - 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(error_matrix, cmap='YlOrRd', aspect='auto')
    
    # 设置刻度
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # 添加数值标签
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f'{error_matrix[i][j]:.4f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Polynomial Degree', fontsize=12)
    ax.set_ylabel('Number of Segments', fontsize=12)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RMSE', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error matrix: {save_path}")


if __name__ == "__main__":
    print("Plotting module ready!")
    print("This module is designed to be imported and used by main programs.")




