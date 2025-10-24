"""
任务一：龙格函数插值与龙格现象对比分析

对龙格函数 f(x) = 1/(1+25x²), x∈[-1,1]
使用三种采样方案：
1. 随机采样
2. 切比雪夫零点
3. 勒让德零点

采用牛顿插值法进行插值，分析龙格现象差异
"""

import math
import sys
import os

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.interpolation import newton_interpolation_batch
from utils.sampling import random_sampling, chebyshev_nodes, legendre_nodes, generate_dense_points
from utils.metrics import max_error, rmse, mae
from utils.plotting import plot_interpolation, plot_interpolation_comparison


def runge_function(x):
    """龙格函数 f(x) = 1/(1+25x²)"""
    return 1.0 / (1.0 + 25.0 * x * x)


def evaluate_function(x_list, func):
    """批量评估函数"""
    return [func(x) for x in x_list]


def run_interpolation_experiment(sampling_method, n_samples, a=-1, b=1, 
                                 n_dense=1000, seed=None):
    """
    运行单次插值实验
    
    Args:
        sampling_method: str - 'random', 'chebyshev', 'legendre'
        n_samples: int - 采样点个数
        a, b: float - 区间端点
        n_dense: int - 密集评估点个数
        seed: int - 随机种子（仅用于random方法）
    
    Returns:
        dict - 实验结果
    """
    # 生成采样点
    if sampling_method == 'random':
        x_sample = random_sampling(a, b, n_samples, seed=seed)
    elif sampling_method == 'chebyshev':
        x_sample = chebyshev_nodes(n_samples, a, b)
    elif sampling_method == 'legendre':
        x_sample = legendre_nodes(n_samples, a, b)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # 计算采样点的函数值
    y_sample = evaluate_function(x_sample, runge_function)
    
    # 生成密集点用于评估和绘图
    x_dense = generate_dense_points(a, b, n_dense)
    y_true = evaluate_function(x_dense, runge_function)
    
    # 执行牛顿插值
    y_interp = newton_interpolation_batch(x_sample, y_sample, x_dense)
    
    # 计算误差
    max_err = max_error(y_true, y_interp)
    rmse_err = rmse(y_true, y_interp)
    mae_err = mae(y_true, y_interp)
    
    results = {
        'x_sample': x_sample,
        'y_sample': y_sample,
        'x_dense': x_dense,
        'y_true': y_true,
        'y_interp': y_interp,
        'max_error': max_err,
        'rmse': rmse_err,
        'mae': mae_err
    }
    
    return results


def print_experiment_summary(method_name, n_samples, results):
    """打印实验摘要"""
    print(f"\n{'='*70}")
    print(f"Method: {method_name}")
    print(f"Number of samples: {n_samples}")
    print(f"{'='*70}")
    print(f"Maximum Error:       {results['max_error']:.6e}")
    print(f"Root Mean Square Error (RMSE): {results['rmse']:.6e}")
    print(f"Mean Absolute Error (MAE):     {results['mae']:.6e}")
    print(f"Sample points (first 5): {[f'{x:.4f}' for x in results['x_sample'][:5]]}")


def analyze_runge_phenomenon(results_dict):
    """
    分析龙格现象
    比较边界区域和中心区域的误差
    """
    print(f"\n{'='*70}")
    print("龙格现象分析（Runge Phenomenon Analysis）")
    print(f"{'='*70}")
    
    for method_name, data in results_dict.items():
        x_dense = data['x_dense']
        y_true = data['y_true']
        y_interp = data['y_interp']
        
        # 计算边界区域误差（|x| > 0.8）
        boundary_errors = []
        center_errors = []
        
        for i in range(len(x_dense)):
            error = abs(y_true[i] - y_interp[i])
            if abs(x_dense[i]) > 0.8:
                boundary_errors.append(error)
            else:
                center_errors.append(error)
        
        avg_boundary_error = sum(boundary_errors) / len(boundary_errors) if boundary_errors else 0
        avg_center_error = sum(center_errors) / len(center_errors) if center_errors else 0
        max_boundary_error = max(boundary_errors) if boundary_errors else 0
        
        print(f"\n{method_name}:")
        print(f"  Boundary region (|x| > 0.8) - Avg Error: {avg_boundary_error:.6e}, Max Error: {max_boundary_error:.6e}")
        print(f"  Center region (|x| ≤ 0.8)   - Avg Error: {avg_center_error:.6e}")
        print(f"  Ratio (Boundary/Center): {avg_boundary_error/avg_center_error if avg_center_error > 0 else float('inf'):.2f}")


def main():
    """主函数"""
    print("="*70)
    print("任务一：龙格函数插值与龙格现象分析")
    print("Task 1: Runge Function Interpolation and Runge Phenomenon Analysis")
    print("="*70)
    
    # 实验参数
    interval = [-1, 1]
    sample_sizes = [5, 9, 15, 21]  # 不同的采样点数
    methods = {
        'Random Sampling': 'random',
        'Chebyshev Nodes': 'chebyshev',
        'Legendre Nodes': 'legendre'
    }
    n_dense = 1000
    random_seed = 42
    
    # 创建结果目录
    results_dir = 'results/task1'
    os.makedirs(results_dir, exist_ok=True)
    
    # 存储所有结果用于对比分析
    all_results = {}
    
    # 对每种方法运行实验
    for method_name, method_code in methods.items():
        print(f"\n{'#'*70}")
        print(f"# {method_name}")
        print(f"{'#'*70}")
        
        all_results[method_name] = {}
        
        for n_samples in sample_sizes:
            print(f"\n--- Running with {n_samples} sample points ---")
            
            try:
                # 运行实验
                results = run_interpolation_experiment(
                    sampling_method=method_code,
                    n_samples=n_samples,
                    a=interval[0],
                    b=interval[1],
                    n_dense=n_dense,
                    seed=random_seed if method_code == 'random' else None
                )
                
                # 打印摘要
                print_experiment_summary(method_name, n_samples, results)
                
                # 保存结果
                all_results[method_name][n_samples] = results
                
                # 绘制单独的图
                title = f"{method_name} - {n_samples} Sample Points\nRunge Function Interpolation"
                save_path = f"{results_dir}/{method_code}_n{n_samples}.png"
                
                plot_interpolation(
                    x_sample=results['x_sample'],
                    y_sample=results['y_sample'],
                    x_dense=results['x_dense'],
                    y_true=results['y_true'],
                    y_interp=results['y_interp'],
                    title=title,
                    save_path=save_path,
                    method_name=method_name
                )
                
            except Exception as e:
                print(f"Error in {method_name} with {n_samples} samples: {e}")
                continue
    
    # 生成对比分析
    print(f"\n{'#'*70}")
    print("# 综合对比分析 (Comprehensive Comparison)")
    print(f"{'#'*70}")
    
    # 对每个采样点数，对比三种方法
    for n_samples in sample_sizes:
        print(f"\n\n{'='*70}")
        print(f"Comparison for {n_samples} sample points:")
        print(f"{'='*70}")
        
        comparison_results = {}
        
        for method_name in methods.keys():
            if n_samples in all_results.get(method_name, {}):
                data = all_results[method_name][n_samples]
                comparison_results[method_name] = data
                
                print(f"\n{method_name}:")
                print(f"  Max Error: {data['max_error']:.6e}")
                print(f"  RMSE:      {data['rmse']:.6e}")
                print(f"  MAE:       {data['mae']:.6e}")
        
        # 绘制对比图
        if comparison_results:
            save_path = f"{results_dir}/comparison_n{n_samples}.png"
            plot_interpolation_comparison(comparison_results, save_path)
            
            # 分析龙格现象
            analyze_runge_phenomenon(comparison_results)
    
    # 生成误差汇总表
    print(f"\n\n{'='*70}")
    print("误差汇总表 (Error Summary Table)")
    print(f"{'='*70}")
    print(f"\n{'Method':<20} {'N':<5} {'Max Error':<15} {'RMSE':<15} {'MAE':<15}")
    print("-" * 70)
    
    for method_name in methods.keys():
        for n_samples in sample_sizes:
            if n_samples in all_results.get(method_name, {}):
                data = all_results[method_name][n_samples]
                print(f"{method_name:<20} {n_samples:<5} "
                      f"{data['max_error']:<15.6e} "
                      f"{data['rmse']:<15.6e} "
                      f"{data['mae']:<15.6e}")
    
    print(f"\n{'='*70}")
    print("任务一完成！所有结果已保存到 results/task1/ 目录")
    print("Task 1 completed! All results saved to results/task1/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()



