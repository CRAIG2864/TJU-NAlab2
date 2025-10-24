"""
Enhanced plotting functions for better visualization
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_fitting_comparison(x_data, y_data_original, y_data_perturbed, 
                            x_dense, y_true, y_fit_original, y_fit_perturbed,
                            perturb_level, title, save_path):
    """
    绘制扰动前后的拟合对比图
    
    Args:
        x_data: list - 数据点x坐标
        y_data_original: list - 原始数据点
        y_data_perturbed: list - 扰动后数据点
        x_dense: list - 密集点x坐标
        y_true: list - 真实函数值
        y_fit_original: list - 无扰动拟合
        y_fit_perturbed: list - 有扰动拟合
        perturb_level: float - 扰动水平
        title: str - 图标题
        save_path: str - 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上图：无扰动
    ax1.plot(x_dense, y_true, 'b-', linewidth=2.5, label='True Function', alpha=0.7)
    ax1.plot(x_dense, y_fit_original, 'g--', linewidth=2, label='Fitted Curve (No Perturbation)', alpha=0.8)
    ax1.scatter(x_data, y_data_original, color='green', s=30, alpha=0.5, label='Sample Points')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('No Perturbation', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 下图：有扰动
    ax2.plot(x_dense, y_true, 'b-', linewidth=2.5, label='True Function', alpha=0.7)
    ax2.plot(x_dense, y_fit_original, 'g--', linewidth=1.5, label='Fitted (No Perturbation)', alpha=0.5)
    ax2.plot(x_dense, y_fit_perturbed, 'r--', linewidth=2, label=f'Fitted (Perturbation {perturb_level*100}%)', alpha=0.8)
    ax2.scatter(x_data, y_data_perturbed, color='orange', s=30, alpha=0.5, label='Perturbed Sample Points')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title(f'With Perturbation ({perturb_level*100}%)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {save_path}")


def plot_fitting_overlay(x_data_list, y_data_list, x_dense, y_true, 
                         y_fit_list, labels, title, save_path):
    """
    在同一张图上叠加多条拟合曲线
    
    Args:
        x_data_list: list of lists - 多组数据点x坐标
        y_data_list: list of lists - 多组数据点y坐标
        x_dense: list - 密集点x坐标
        y_true: list - 真实函数值
        y_fit_list: list of lists - 多组拟合结果
        labels: list - 每组的标签
        title: str - 图标题
        save_path: str - 保存路径
    """
    plt.figure(figsize=(14, 8))
    
    # 绘制真实函数
    plt.plot(x_dense, y_true, 'b-', linewidth=3, label='True Function', alpha=0.8, zorder=1)
    
    # 颜色方案
    colors = ['green', 'orange', 'red', 'purple']
    alphas = [0.7, 0.7, 0.8, 0.8]
    linewidths = [2, 2, 2.5, 2.5]
    
    # 绘制多条拟合曲线
    for i, (x_data, y_data, y_fit, label) in enumerate(zip(x_data_list, y_data_list, y_fit_list, labels)):
        color = colors[i % len(colors)]
        alpha = alphas[i % len(alphas)]
        lw = linewidths[i % len(linewidths)]
        
        plt.plot(x_dense, y_fit, '--', linewidth=lw, color=color, 
                label=f'Fitted: {label}', alpha=alpha, zorder=2+i)
        plt.scatter(x_data, y_data, color=color, s=25, alpha=0.4, zorder=10+i)
    
    plt.xlabel('x', fontsize=13)
    plt.ylabel('y', fontsize=13)
    plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved overlay plot: {save_path}")


if __name__ == "__main__":
    print("Enhanced plotting module ready!")



