"""
专门为任务二设计的可视化模块
重点：清晰的对比、信息丰富、易于理解
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')


def plot_subtask1_comprehensive(x_sample, y_samples_dict, x_dense, y_true, 
                                 y_fits_dict, params, save_path):
    """
    子任务1综合图：2x2布局，每个扰动水平一个独立子图
    清晰展示每个扰动水平的拟合效果
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    c, d, e, f = params['c'], params['d'], params['e'], params['f']
    
    # 为每个扰动水平创建子图
    labels_list = list(y_fits_dict.keys())
    colors = ['green', 'orange', 'red', 'purple']
    
    for idx, label in enumerate(labels_list):
        ax = axes[idx]
        
        y_sample = y_samples_dict[label]
        y_fit = y_fits_dict[label]
        color = colors[idx]
        
        # 绘制真实函数
        ax.plot(x_dense, y_true, 'b-', linewidth=2.5, label='True Function', 
                alpha=0.7, zorder=1)
        
        # 绘制拟合曲线
        ax.plot(x_dense, y_fit, color=color, linestyle='--', linewidth=2.5, 
                label='Fitted Curve', alpha=0.85, zorder=2)
        
        # 绘制采样点
        ax.scatter(x_sample, y_sample, color=color, s=30, alpha=0.6, 
                  label='Sample Points', zorder=3, edgecolors='black', linewidths=0.5)
        
        # 计算并显示误差
        # 在采样点处计算误差
        y_fit_sample = []
        for x in x_sample:
            # 找到最近的x_dense索引
            idx_closest = min(range(len(x_dense)), key=lambda i: abs(x_dense[i] - x))
            y_fit_sample.append(y_fit[idx_closest])
        
        residuals = [abs(y_sample[i] - y_fit_sample[i]) for i in range(len(y_sample))]
        max_residual = max(residuals)
        avg_residual = sum(residuals) / len(residuals)
        
        # 设置标题和标签
        ax.set_xlabel('x', fontsize=11, fontweight='bold')
        ax.set_ylabel('y', fontsize=11, fontweight='bold')
        ax.set_title(f'{label}\nMax Residual: {max_residual:.4f}, Avg: {avg_residual:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # 总标题
    fig.suptitle(f'Subtask 1: Effect of Perturbation on Least Squares Fitting\n'
                 f'Target Function: g(x) = {c}·sin({d}x) + {e}·cos({f}x)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_subtask2_by_segments(x_sample, y_sample, x_dense, y_true, 
                               results_by_degree, n_segments, params, save_path):
    """
    子任务2：按分段数生成对比图
    一张图包含4个子图，对应4种多项式次数
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    degrees = sorted(results_by_degree.keys())
    
    for idx, degree in enumerate(degrees):
        ax = axes[idx]
        
        y_fit_dense = results_by_degree[degree]['y_fit_dense']
        rmse_val = results_by_degree[degree]['rmse']
        
        # 绘制真实函数
        ax.plot(x_dense, y_true, 'b-', linewidth=2.5, label='True Function', alpha=0.7)
        
        # 绘制拟合结果
        ax.plot(x_dense, y_fit_dense, 'r--', linewidth=2, label=f'Fitted (Degree {degree})', alpha=0.8)
        
        # 绘制数据点
        ax.scatter(x_sample, y_sample, color='green', s=25, alpha=0.5, 
                  label='Sample Points', zorder=5)
        
        # 绘制分段边界
        if 'intervals' in results_by_degree[degree]:
            intervals = results_by_degree[degree]['intervals']
            for i, interval in enumerate(intervals[1:], 1):  # 跳过第一个
                ax.axvline(x=interval[0], color='gray', linestyle=':', 
                          linewidth=1.5, alpha=0.6, label='Segment Boundary' if i == 1 else '')
        
        ax.set_xlabel('x', fontsize=11, fontweight='bold')
        ax.set_ylabel('y', fontsize=11, fontweight='bold')
        ax.set_title(f'Polynomial Degree {degree}\nRMSE = {rmse_val:.6f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    c, d, e, f = params['c'], params['d'], params['e'], params['f']
    fig.suptitle(f'Subtask 2: Piecewise Polynomial Fitting ({n_segments} Segments)\n'
                 f'Target Function: g(x) = {c}·sin({d}x) + {e}·cos({f}x)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_heatmap_enhanced(error_matrix, row_labels, col_labels, 
                                best_config, params, save_path):
    """
    增强版误差热力图
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 转换为对数尺度以更好地显示差异
    import math
    log_matrix = [[math.log10(val) if val > 0 else -10 for val in row] 
                  for row in error_matrix]
    
    im = ax.imshow(log_matrix, cmap='RdYlGn_r', aspect='auto')
    
    # 设置刻度
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=12, fontweight='bold')
    ax.set_yticklabels(row_labels, fontsize=12, fontweight='bold')
    
    # 添加数值标签
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = error_matrix[i][j]
            # 使用不同颜色使文字可见
            text_color = 'white' if log_matrix[i][j] < -0.5 else 'black'
            
            # 标记最佳配置
            if best_config and i == best_config[0] and j == best_config[1]:
                text = ax.text(j, i, f'{val:.4f}\n★ BEST', 
                              ha="center", va="center", color='blue', 
                              fontsize=11, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            else:
                text = ax.text(j, i, f'{val:.4f}',
                              ha="center", va="center", color=text_color, 
                              fontsize=10, fontweight='bold')
    
    ax.set_title('Piecewise Fitting Error Matrix (RMSE)\nLower is Better', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Polynomial Degree', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Segments', fontsize=13, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(RMSE)', fontsize=12, fontweight='bold')
    
    c, d, e, f = params['c'], params['d'], params['e'], params['f']
    fig.text(0.5, 0.02, f'Target Function: g(x) = {c}·sin({d}x) + {e}·cos({f}x)', 
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_subtask3_comparison(x_data, y_data_clean, y_data_noisy, x_dense, y_true,
                             y_fit_standard, y_fit_ransac, inliers, 
                             noise_info, params, save_path):
    """
    子任务3：RANSAC vs 标准最小二乘对比
    布局：2x2网格
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. 标准最小二乘结果
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(x_dense, y_true, 'b-', linewidth=2.5, label='True Function', alpha=0.7)
    ax1.plot(x_dense, y_fit_standard, 'orange', linestyle='--', linewidth=2.5, 
            label='Standard LS Fit', alpha=0.8)
    ax1.scatter(x_data, y_data_noisy, color='gray', s=30, alpha=0.5, label='Noisy Data')
    ax1.set_xlabel('x', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y', fontsize=11, fontweight='bold')
    ax1.set_title('Standard Least Squares', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. RANSAC结果
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(x_dense, y_true, 'b-', linewidth=2.5, label='True Function', alpha=0.7)
    ax2.plot(x_dense, y_fit_ransac, 'red', linestyle='-', linewidth=2.5, 
            label='RANSAC Fit', alpha=0.8)
    
    # 区分内点和外点
    outliers = [i for i in range(len(x_data)) if i not in inliers]
    if inliers:
        x_in = [x_data[i] for i in inliers]
        y_in = [y_data_noisy[i] for i in inliers]
        ax2.scatter(x_in, y_in, color='green', s=40, alpha=0.6, 
                   label=f'Inliers ({len(inliers)})', marker='o')
    if outliers:
        x_out = [x_data[i] for i in outliers]
        y_out = [y_data_noisy[i] for i in outliers]
        ax2.scatter(x_out, y_out, color='red', s=60, alpha=0.8, 
                   label=f'Outliers ({len(outliers)})', marker='x', linewidths=2)
    
    ax2.set_xlabel('x', fontsize=11, fontweight='bold')
    ax2.set_ylabel('y', fontsize=11, fontweight='bold')
    ax2.set_title('RANSAC Least Squares', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. 对比图（两条拟合曲线）
    ax3 = plt.subplot(gs[1, :])
    ax3.plot(x_dense, y_true, 'b-', linewidth=3, label='True Function', alpha=0.8, zorder=3)
    ax3.plot(x_dense, y_fit_standard, 'orange', linestyle='--', linewidth=2.5, 
            label='Standard LS', alpha=0.7, zorder=2)
    ax3.plot(x_dense, y_fit_ransac, 'red', linestyle='-', linewidth=2.5, 
            label='RANSAC LS', alpha=0.8, zorder=2)
    ax3.scatter(x_data, y_data_noisy, color='gray', s=20, alpha=0.3, 
               label='Noisy Data', zorder=1)
    
    ax3.set_xlabel('x', fontsize=12, fontweight='bold')
    ax3.set_ylabel('y', fontsize=12, fontweight='bold')
    ax3.set_title('Comparison: Standard LS vs RANSAC', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11, loc='best', ncol=4)
    ax3.grid(True, alpha=0.3)
    
    # 总标题
    c, d, e, f = params['c'], params['d'], params['e'], params['f']
    fig.suptitle(f'Subtask 3: RANSAC Robust Fitting - {noise_info}\n'
                 f'Target Function: g(x) = {c}·sin({d}x) + {e}·cos({f}x)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("Task 2 plotting module ready!")
    print("Designed for clarity and easy comparison.")

