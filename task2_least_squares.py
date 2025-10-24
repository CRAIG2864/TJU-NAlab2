"""
任务二：最小二乘法曲线拟合（改进版 - 清晰的可视化）

对标准函数 g(x) = c*sin(dx) + e*cos(fx) 进行拟合：
1. 子任务1：基本最小二乘拟合，观察扰动影响
2. 子任务2：分段多项式拟合，对比不同分段数和多项式次数
3. 子任务3：对比标准最小二乘和RANSAC方法在含噪声情况下的效果
"""

import math
import sys
import os

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.fitting import (least_squares_fit, evaluate_fit, polynomial_basis, 
                          trigonometric_basis, piecewise_fit, 
                          evaluate_piecewise_fit, create_intervals)
from core.ransac import ransac_fit
from utils.sampling import uniform_sampling
from utils.metrics import rmse, mae, add_perturbation, add_noise
from utils.plotting_task2 import (plot_subtask1_comprehensive, 
                                   plot_subtask2_by_segments,
                                   plot_error_heatmap_enhanced,
                                   plot_subtask3_comparison)


def target_function(x, c, d, e, f):
    """目标函数 g(x) = c*sin(dx) + e*cos(fx)"""
    return c * math.sin(d * x) + e * math.cos(f * x)


def evaluate_target_function(x_list, c, d, e, f):
    """批量评估目标函数"""
    return [target_function(x, c, d, e, f) for x in x_list]


def subtask1_basic_fitting(params):
    """
    子任务1：基本最小二乘拟合
    - 使用三角基函数拟合
    - 对Y值添加不同程度扰动，观察拟合曲线变化
    - 生成1张综合对比图
    """
    print("\n" + "="*80)
    print("子任务1：基本最小二乘拟合与扰动分析")
    print("Subtask 1: Basic Least Squares Fitting with Perturbation Analysis")
    print("="*80)
    
    # 提取参数
    a, b = params['a'], params['b']
    c, d, e, f = params['c'], params['d'], params['e'], params['f']
    n_samples = params['n'] + 1
    n_test = params['m']
    
    results_dir = 'results/task2/subtask1'
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成采样点
    x_sample = uniform_sampling(a, b, n_samples)
    y_sample_clean = evaluate_target_function(x_sample, c, d, e, f)
    
    # 生成测试点和密集点
    x_test = uniform_sampling(a, b, n_test)
    y_test_true = evaluate_target_function(x_test, c, d, e, f)
    x_dense = uniform_sampling(a, b, 500)
    y_dense_true = evaluate_target_function(x_dense, c, d, e, f)
    
    # 三角基函数
    basis_functions = trigonometric_basis({'d': d, 'f': f})
    
    print(f"\n实验配置:")
    print(f"  采样点数: {n_samples}, 测试点数: {n_test}")
    print(f"  目标函数: g(x) = {c}·sin({d}x) + {e}·cos({f}x)")
    
    # 扰动水平
    perturbation_levels = [0.0, 0.1, 0.2, 0.3]  # 0%, 10%, 20%, 30%
    
    # 存储所有结果
    y_samples_dict = {}
    y_fits_dict = {}
    results_summary = []
    
    for perturb_level in perturbation_levels:
        label = f"No Perturbation" if perturb_level == 0 else f"Perturbation {perturb_level*100:.0f}%"
        
        print(f"\n处理: {label}")
        
        # 添加扰动
        if perturb_level == 0:
            y_sample = y_sample_clean
        else:
            y_sample = add_perturbation(y_sample_clean, perturb_level)
        
        # 拟合
        coef = least_squares_fit(x_sample, y_sample, basis_functions)
        print(f"  拟合系数: [{coef[0]:.6f}, {coef[1]:.6f}]")
        print(f"  期望系数: [{c:.6f}, {e:.6f}]")
        print(f"  系数误差: [{abs(coef[0]-c):.6f}, {abs(coef[1]-e):.6f}]")
        
        # 评估
        y_fit_test = evaluate_fit(x_test, coef, basis_functions)
        y_fit_dense = evaluate_fit(x_dense, coef, basis_functions)
        
        rmse_val = rmse(y_test_true, y_fit_test)
        mae_val = mae(y_test_true, y_fit_test)
        
        print(f"  RMSE: {rmse_val:.6e}")
        print(f"  MAE:  {mae_val:.6e}")
        
        # 存储结果
        y_samples_dict[label] = y_sample
        y_fits_dict[label] = y_fit_dense
        results_summary.append({
            'label': label,
            'perturb_level': perturb_level,
            'coefficients': coef,
            'rmse': rmse_val,
            'mae': mae_val
        })
    
    # 生成综合对比图
    print(f"\n生成综合对比图...")
    plot_subtask1_comprehensive(
        x_sample=x_sample,
        y_samples_dict=y_samples_dict,
        x_dense=x_dense,
        y_true=y_dense_true,
        y_fits_dict=y_fits_dict,
        params=params,
        save_path=f"{results_dir}/comprehensive_comparison.png"
    )
    
    # 打印总结
    print(f"\n{'='*80}")
    print("扰动影响总结:")
    print(f"{'='*80}")
    print(f"{'扰动水平':<15} {'RMSE':<15} {'MAE':<15} {'系数误差'}")
    print("-" * 80)
    for result in results_summary:
        coef_err = abs(result['coefficients'][0] - c) + abs(result['coefficients'][1] - e)
        print(f"{result['label']:<15} {result['rmse']:<15.6e} {result['mae']:<15.6e} {coef_err:.6e}")
    
    print(f"\n子任务1完成！结果保存到 {results_dir}/")
    print(f"  - comprehensive_comparison.png (综合对比图)")


def subtask2_piecewise_fitting(params):
    """
    子任务2：分段多项式拟合
    - 对区间进行2、4、8等分
    - 在每个子区间使用1次、2次、3次、4次多项式拟合
    - 生成3张对比图（按分段数）+ 1张误差热力图
    """
    print("\n" + "="*80)
    print("子任务2：分段多项式拟合")
    print("Subtask 2: Piecewise Polynomial Fitting")
    print("="*80)
    
    # 提取参数
    a, b = params['a'], params['b']
    c, d, e, f = params['c'], params['d'], params['e'], params['f']
    n_samples = params['n'] + 1
    n_test = params['m']
    
    results_dir = 'results/task2/subtask2'
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成数据
    x_sample = uniform_sampling(a, b, n_samples)
    y_sample = evaluate_target_function(x_sample, c, d, e, f)
    x_test = uniform_sampling(a, b, n_test)
    y_test_true = evaluate_target_function(x_test, c, d, e, f)
    x_dense = uniform_sampling(a, b, 500)
    y_dense_true = evaluate_target_function(x_dense, c, d, e, f)
    
    # 实验配置
    n_segments_list = [2, 4, 8]
    degrees = [1, 2, 3, 4]
    
    print(f"\n实验配置:")
    print(f"  采样点数: {n_samples}, 测试点数: {n_test}")
    print(f"  分段数: {n_segments_list}")
    print(f"  多项式次数: {degrees}")
    
    # 误差矩阵
    error_matrix = []
    all_results = {}
    
    # 主实验循环
    for n_segments in n_segments_list:
        print(f"\n{'─'*80}")
        print(f"分段数: {n_segments}")
        print(f"{'─'*80}")
        
        intervals = create_intervals(a, b, n_segments)
        error_row = []
        results_by_degree = {}
        
        for degree in degrees:
            print(f"  处理: {degree}次多项式")
            
            # 分段拟合
            piecewise_coef = piecewise_fit(x_sample, y_sample, intervals, degree)
            
            # 评估
            y_fit_test = evaluate_piecewise_fit(x_test, piecewise_coef)
            y_fit_dense = evaluate_piecewise_fit(x_dense, piecewise_coef)
            
            rmse_val = rmse(y_test_true, y_fit_test)
            mae_val = mae(y_test_true, y_fit_test)
            
            print(f"    RMSE: {rmse_val:.6e}, MAE: {mae_val:.6e}")
            
            error_row.append(rmse_val)
            results_by_degree[degree] = {
                'y_fit_dense': y_fit_dense,
                'rmse': rmse_val,
                'mae': mae_val,
                'intervals': intervals
            }
        
        error_matrix.append(error_row)
        all_results[n_segments] = results_by_degree
        
        # 为每个分段数生成对比图
        print(f"  生成对比图: {n_segments}分段...")
        plot_subtask2_by_segments(
            x_sample=x_sample,
            y_sample=y_sample,
            x_dense=x_dense,
            y_true=y_dense_true,
            results_by_degree=results_by_degree,
            n_segments=n_segments,
            params=params,
            save_path=f"{results_dir}/comparison_{n_segments}_segments.png"
        )
    
    # 找到最佳配置
    min_error = float('inf')
    best_config = None
    for i, row in enumerate(error_matrix):
        for j, val in enumerate(row):
            if val < min_error:
                min_error = val
                best_config = (i, j)
    
    best_segments = n_segments_list[best_config[0]]
    best_degree = degrees[best_config[1]]
    
    # 生成误差热力图
    print(f"\n生成误差热力图...")
    plot_error_heatmap_enhanced(
        error_matrix=error_matrix,
        row_labels=[f"{n} Segments" for n in n_segments_list],
        col_labels=[f"Degree {d}" for d in degrees],
        best_config=best_config,
        params=params,
        save_path=f"{results_dir}/error_heatmap.png"
    )
    
    # 打印误差矩阵
    print(f"\n{'='*80}")
    print("误差矩阵 (RMSE):")
    print(f"{'='*80}")
    print(f"{'分段数':<12} " + " ".join([f"Deg{d:<10}" for d in degrees]))
    print("-" * 80)
    for i, n_seg in enumerate(n_segments_list):
        values = " ".join([f"{error_matrix[i][j]:<12.6f}" for j in range(len(degrees))])
        print(f"{n_seg:<12} {values}")
    
    print(f"\n最佳配置: {best_segments} 分段, {best_degree} 次多项式")
    print(f"最小RMSE: {min_error:.6e}")
    
    print(f"\n子任务2完成！结果保存到 {results_dir}/")
    print(f"  - comparison_X_segments.png (3张对比图)")
    print(f"  - error_heatmap.png (误差热力图)")


def subtask3_ransac_comparison(params):
    """
    子任务3：RANSAC vs 标准最小二乘
    - 添加较大噪声和离群点
    - 对比两种方法的拟合效果
    - 生成3张对比图（3种噪声情况）
    """
    print("\n" + "="*80)
    print("子任务3：RANSAC鲁棒拟合对比")
    print("Subtask 3: RANSAC vs Standard Least Squares")
    print("="*80)
    
    # 提取参数
    a, b = params['a'], params['b']
    c, d, e, f = params['c'], params['d'], params['e'], params['f']
    n_samples = params['n'] + 1
    n_test = params['m']
    
    results_dir = 'results/task2/subtask3'
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成干净数据
    x_sample = uniform_sampling(a, b, n_samples)
    y_sample_clean = evaluate_target_function(x_sample, c, d, e, f)
    x_test = uniform_sampling(a, b, n_test)
    y_test_true = evaluate_target_function(x_test, c, d, e, f)
    x_dense = uniform_sampling(a, b, 500)
    y_dense_true = evaluate_target_function(x_dense, c, d, e, f)
    
    # 基函数
    basis_functions = trigonometric_basis({'d': d, 'f': f})
    
    # 三种噪声实验
    experiments = [
        {
            'name': 'Gaussian Noise',
            'noise_type': 'gaussian',
            'noise_level': 0.2,
            'outlier_ratio': 0.0,
            'ransac_threshold': 0.3
        },
        {
            'name': 'Outliers',
            'noise_type': 'outliers',
            'noise_level': 0.1,
            'outlier_ratio': 0.25,
            'ransac_threshold': 0.5
        },
        {
            'name': 'Mixed Noise',
            'noise_type': 'gaussian',
            'noise_level': 0.15,
            'outlier_ratio': 0.2,
            'ransac_threshold': 0.4
        }
    ]
    
    for exp in experiments:
        print(f"\n{'─'*80}")
        print(f"实验: {exp['name']}")
        print(f"{'─'*80}")
        
        # 添加噪声
        y_sample_noisy = add_noise(y_sample_clean, 
                                    noise_level=exp['noise_level'],
                                    noise_type=exp['noise_type'],
                                    outlier_ratio=exp['outlier_ratio'])
        
        # 如果是混合噪声，还要添加离群点
        if exp['name'] == 'Mixed Noise':
            y_sample_noisy = add_noise(y_sample_noisy,
                                       noise_type='outliers',
                                       outlier_ratio=exp['outlier_ratio'])
        
        # 标准最小二乘
        print(f"  标准最小二乘:")
        coef_standard = least_squares_fit(x_sample, y_sample_noisy, basis_functions)
        y_fit_standard_test = evaluate_fit(x_test, coef_standard, basis_functions)
        y_fit_standard_dense = evaluate_fit(x_dense, coef_standard, basis_functions)
        rmse_standard = rmse(y_test_true, y_fit_standard_test)
        
        print(f"    系数: [{coef_standard[0]:.6f}, {coef_standard[1]:.6f}]")
        print(f"    RMSE: {rmse_standard:.6e}")
        
        # RANSAC
        print(f"  RANSAC:")
        ransac_result = ransac_fit(
            x_sample, y_sample_noisy, basis_functions,
            max_iterations=1500,
            threshold=exp['ransac_threshold'],
            min_samples=5,
            seed=42
        )
        
        coef_ransac = ransac_result['coefficients']
        y_fit_ransac_test = evaluate_fit(x_test, coef_ransac, basis_functions)
        y_fit_ransac_dense = evaluate_fit(x_dense, coef_ransac, basis_functions)
        rmse_ransac = rmse(y_test_true, y_fit_ransac_test)
        
        print(f"    系数: [{coef_ransac[0]:.6f}, {coef_ransac[1]:.6f}]")
        print(f"    RMSE: {rmse_ransac:.6e}")
        print(f"    内点数: {ransac_result['n_inliers']} / {n_samples}")
        print(f"    内点比例: {ransac_result['inlier_ratio']:.2%}")
        
        improvement = (1 - rmse_ransac/rmse_standard) * 100
        print(f"  RMSE改善: {improvement:.2f}%")
        
        # 生成对比图
        print(f"  生成对比图...")
        plot_subtask3_comparison(
            x_data=x_sample,
            y_data_clean=y_sample_clean,
            y_data_noisy=y_sample_noisy,
            x_dense=x_dense,
            y_true=y_dense_true,
            y_fit_standard=y_fit_standard_dense,
            y_fit_ransac=y_fit_ransac_dense,
            inliers=ransac_result['inliers'],
            noise_info=exp['name'],
            params=params,
            save_path=f"{results_dir}/comparison_{exp['name'].replace(' ', '_').lower()}.png"
        )
    
    print(f"\n子任务3完成！结果保存到 {results_dir}/")
    print(f"  - comparison_*.png (3张对比图)")


def main():
    """主函数"""
    print("="*80)
    print("任务二：最小二乘法曲线拟合（改进版）")
    print("Task 2: Least Squares Curve Fitting (Enhanced Version)")
    print("="*80)
    
    # 实验参数
    params = {
        'a': 0,
        'b': 2 * math.pi,
        'c': 2.0,   # 改变系数，使sin分量幅度更大
        'd': 1.5,   # 降低sin频率
        'e': 1.5,   # 改变cos系数
        'f': 2.5,   # 改变cos频率
        'n': 80,    # 增加到81个采样点，确保每段有足够的点
        'm': 100
    }
    
    print(f"\n实验参数:")
    print(f"  区间: [{params['a']:.2f}, {params['b']:.2f}]")
    print(f"  目标函数: g(x) = {params['c']}·sin({params['d']}x) + {params['e']}·cos({params['f']}x)")
    print(f"  采样点数: {params['n'] + 1}")
    print(f"  测试点数: {params['m']}")
    
    # 运行三个子任务
    try:
        subtask1_basic_fitting(params)
    except Exception as e:
        print(f"\n子任务1出错: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        subtask2_piecewise_fitting(params)
    except Exception as e:
        print(f"\n子任务2出错: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        subtask3_ransac_comparison(params)
    except Exception as e:
        print(f"\n子任务3出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("任务二完成！")
    print("Task 2 Completed!")
    print("="*80)
    print("\n生成的图像:")
    print("  子任务1: 1张综合对比图")
    print("  子任务2: 3张分段对比图 + 1张误差热力图")
    print("  子任务3: 3张RANSAC对比图")
    print("  总计: 8张高质量图像")
    print("="*80)


if __name__ == "__main__":
    main()

