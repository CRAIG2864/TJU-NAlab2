"""
生成额外的可视化图表：误差分布图、箱线图等
用于增强实验报告的数据展示
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
from core.interpolation import newton_divided_difference
from utils.sampling import uniform_sampling, random_sampling, chebyshev_nodes, legendre_nodes
from utils.metrics import max_error, rmse, mae
from core.fitting import least_squares_fit
import math
import random

def runge_function(x):
    """龙格函数"""
    return 1.0 / (1.0 + 25.0 * x * x)

def target_function_task2(x, c=2.0, d=1.5, e=1.5, f=2.5):
    """任务2的目标函数"""
    return c * math.sin(d * x) + e * math.cos(f * x)

def newton_interp_eval(x, x_sample, coeffs):
    """使用牛顿插值多项式求值"""
    y_val = coeffs[0]
    prod = 1.0
    for i in range(1, len(coeffs)):
        prod *= (x - x_sample[i-1])
        y_val += coeffs[i] * prod
    return y_val

def generate_task1_error_distribution():
    """生成任务1的误差分布图"""
    print("生成任务1误差分布图...")
    
    n_values = [5, 9, 15, 21]
    
    # 评估点
    x_eval = uniform_sampling(-1, 1, 1000)
    y_true = [runge_function(x) for x in x_eval]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, n in enumerate(n_values):
        ax = axes[idx]
        
        # 三种采样方法
        random.seed(42)
        x_sample_random = random_sampling(-1, 1, n)
        x_sample_cheb = chebyshev_nodes(n, -1, 1)
        x_sample_leg = legendre_nodes(n, -1, 1)
        
        samples = [
            ('Random', x_sample_random),
            ('Chebyshev', x_sample_cheb),
            ('Legendre', x_sample_leg)
        ]
        
        for method_name, x_sample in samples:
            y_sample = [runge_function(x) for x in x_sample]
            coeffs = newton_divided_difference(x_sample, y_sample)
            
            # 计算插值误差
            errors = []
            for x in x_eval:
                y_interp = newton_interp_eval(x, x_sample, coeffs)
                errors.append(abs(runge_function(x) - y_interp))
            
            # 绘制误差分布
            ax.hist(errors, bins=50, alpha=0.5, label=method_name, density=True)
        
        ax.set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'Error Distribution (n={n})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/task1/additional', exist_ok=True)
    plt.savefig('results/task1/additional/error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  保存: results/task1/additional/error_distribution.png")

def generate_task1_error_boxplot():
    """生成任务1的误差箱线图"""
    print("生成任务1误差箱线图...")
    
    n_values = [5, 9, 15, 21]
    x_eval = uniform_sampling(-1, 1, 1000)
    y_true = [runge_function(x) for x in x_eval]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = [
        ('Random', lambda n: (random.seed(42), random_sampling(-1, 1, n))[1]),
        ('Chebyshev', lambda n: chebyshev_nodes(n, -1, 1)),
        ('Legendre', lambda n: legendre_nodes(n, -1, 1))
    ]
    
    for method_idx, (method_name, sample_func) in enumerate(methods):
        ax = axes[method_idx]
        error_data = []
        
        for n in n_values:
            x_sample = sample_func(n)
            y_sample = [runge_function(x) for x in x_sample]
            coeffs = newton_divided_difference(x_sample, y_sample)
            
            errors = []
            for x in x_eval:
                y_interp = newton_interp_eval(x, x_sample, coeffs)
                errors.append(abs(runge_function(x) - y_interp))
            
            error_data.append(errors)
        
        bp = ax.boxplot(error_data, labels=[f'n={n}' for n in n_values],
                        patch_artist=True, notch=True)
        
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
        ax.set_title(f'{method_name} Sampling', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/task1/additional/error_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  保存: results/task1/additional/error_boxplot.png")

def generate_task1_convergence_plot():
    """生成任务1的收敛性分析图"""
    print("生成任务1收敛性分析图...")
    
    n_values = list(range(5, 26, 2))
    x_eval = uniform_sampling(-1, 1, 1000)
    y_true = [runge_function(x) for x in x_eval]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    methods = {
        'Random': (lambda n: (random.seed(42), random_sampling(-1, 1, n))[1], 'red', 'o'),
        'Chebyshev': (lambda n: chebyshev_nodes(n, -1, 1), 'green', 's'),
        'Legendre': (lambda n: legendre_nodes(n, -1, 1), 'blue', '^')
    }
    
    error_metrics = {name: {'max': [], 'rmse': []} for name in methods.keys()}
    
    for n in n_values:
        for method_name, (sample_func, _, _) in methods.items():
            x_sample = sample_func(n)
            y_sample = [runge_function(x) for x in x_sample]
            coeffs = newton_divided_difference(x_sample, y_sample)
            
            y_interp = [newton_interp_eval(x, x_sample, coeffs) for x in x_eval]
            
            max_err = max_error(y_true, y_interp)
            rmse_err = rmse(y_true, y_interp)
            
            error_metrics[method_name]['max'].append(max_err)
            error_metrics[method_name]['rmse'].append(rmse_err)
    
    # 绘制Max Error
    for method_name, (_, color, marker) in methods.items():
        ax1.plot(n_values, error_metrics[method_name]['max'], 
                marker=marker, color=color, label=method_name, 
                linewidth=2, markersize=6)
    
    ax1.set_xlabel('Number of Nodes (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max Error', fontsize=12, fontweight='bold')
    ax1.set_title('Convergence Analysis: Max Error', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 绘制RMSE
    for method_name, (_, color, marker) in methods.items():
        ax2.plot(n_values, error_metrics[method_name]['rmse'],
                marker=marker, color=color, label=method_name, 
                linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Nodes (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence Analysis: RMSE', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/task1/additional/convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  保存: results/task1/additional/convergence_analysis.png")

def generate_task2_residual_plots():
    """生成任务2的残差分析图"""
    print("生成任务2残差分析图...")
    
    c, d, e, f = 2.0, 1.5, 1.5, 2.5
    n = 80
    
    x_sample = uniform_sampling(0, 2 * math.pi, n + 1)
    y_sample = [target_function_task2(x, c, d, e, f) for x in x_sample]
    
    perturbation_levels = [0.0, 0.10, 0.20, 0.30]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, alpha in enumerate(perturbation_levels):
        ax = axes[idx]
        
        # 添加扰动
        random.seed(42)
        y_perturbed = []
        for y in y_sample:
            if alpha > 0:
                noise = random.uniform(-alpha * abs(y), alpha * abs(y))
                y_perturbed.append(y + noise)
            else:
                y_perturbed.append(y)
        
        # 拟合
        basis_funcs = [
            lambda x, d=d: math.sin(d * x),
            lambda x, f=f: math.cos(f * x)
        ]
        coeffs = least_squares_fit(x_sample, y_perturbed, basis_funcs)
        
        # 计算残差
        residuals = []
        for i, x in enumerate(x_sample):
            y_fit = sum(coeffs[j] * basis_funcs[j](x) for j in range(len(coeffs)))
            residuals.append(y_perturbed[i] - y_fit)
        
        # 绘制残差图
        ax.scatter(x_sample, residuals, alpha=0.6, s=30, c='blue', edgecolors='black', linewidths=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Line')
        
        # 统计信息
        mean_residual = sum(residuals) / len(residuals)
        std_residual = (sum((r - mean_residual)**2 for r in residuals) / len(residuals)) ** 0.5
        
        ax.axhline(y=mean_residual, color='green', linestyle=':', linewidth=1.5, 
                  label=f'Mean: {mean_residual:.2e}')
        ax.axhline(y=mean_residual + 2*std_residual, color='orange', linestyle=':', 
                  linewidth=1.5, alpha=0.7)
        ax.axhline(y=mean_residual - 2*std_residual, color='orange', linestyle=':', 
                  linewidth=1.5, alpha=0.7, label=f'±2σ: {std_residual:.2e}')
        
        ax.set_xlabel('x', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        
        if alpha == 0:
            title = 'No Perturbation'
        else:
            title = f'Perturbation {int(alpha*100)}%'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Residual Analysis for Different Perturbation Levels', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    os.makedirs('results/task2/additional', exist_ok=True)
    plt.savefig('results/task2/additional/residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  保存: results/task2/additional/residual_analysis.png")

def generate_task2_coefficient_comparison():
    """生成任务2的系数对比图"""
    print("生成任务2系数对比图...")
    
    c_true, d, e_true, f = 2.0, 1.5, 1.5, 2.5
    n = 80
    
    x_sample = uniform_sampling(0, 2 * math.pi, n + 1)
    y_sample = [target_function_task2(x, c_true, d, e_true, f) for x in x_sample]
    
    perturbation_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    c_fitted = []
    e_fitted = []
    c_errors = []
    e_errors = []
    
    for alpha in perturbation_levels:
        random.seed(42)
        y_perturbed = []
        for y in y_sample:
            if alpha > 0:
                noise = random.uniform(-alpha * abs(y), alpha * abs(y))
                y_perturbed.append(y + noise)
            else:
                y_perturbed.append(y)
        
        basis_funcs = [
            lambda x, d=d: math.sin(d * x),
            lambda x, f=f: math.cos(f * x)
        ]
        coeffs = least_squares_fit(x_sample, y_perturbed, basis_funcs)
        
        c_fitted.append(coeffs[0])
        e_fitted.append(coeffs[1])
        c_errors.append(abs(coeffs[0] - c_true))
        e_errors.append(abs(coeffs[1] - e_true))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    perturbation_percent = [alpha * 100 for alpha in perturbation_levels]
    
    ax1.plot(perturbation_percent, c_fitted, marker='o', linewidth=2, 
            markersize=8, label=f'Fitted c (True: {c_true})', color='blue')
    ax1.plot(perturbation_percent, e_fitted, marker='s', linewidth=2, 
            markersize=8, label=f'Fitted e (True: {e_true})', color='red')
    ax1.axhline(y=c_true, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.axhline(y=e_true, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax1.set_xlabel('Perturbation Level (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax1.set_title('Fitted Coefficients vs Perturbation Level', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(perturbation_percent, c_errors, marker='o', linewidth=2, 
            markersize=8, label='Error in c', color='blue')
    ax2.plot(perturbation_percent, e_errors, marker='s', linewidth=2, 
            markersize=8, label='Error in e', color='red')
    
    ax2.set_xlabel('Perturbation Level (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax2.set_title('Coefficient Errors vs Perturbation Level', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/task2/additional/coefficient_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  保存: results/task2/additional/coefficient_analysis.png")

def main():
    """主函数"""
    print("="*60)
    print("生成额外的可视化图表")
    print("="*60)
    
    # 任务1
    print("\n任务一：")
    generate_task1_error_distribution()
    generate_task1_error_boxplot()
    generate_task1_convergence_plot()
    
    # 任务2
    print("\n任务二：")
    generate_task2_residual_plots()
    generate_task2_coefficient_comparison()
    
    print("\n" + "="*60)
    print("所有图表生成完成！")
    print("="*60)

if __name__ == "__main__":
    main()
