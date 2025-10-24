"""
RANSAC (Random Sample Consensus) algorithm for robust fitting
Hand-implemented without using external libraries
"""

import random
import math
from core.fitting import least_squares_fit, evaluate_fit


def compute_residuals(x_data, y_data, coefficients, basis_functions):
    """
    计算残差（用于判断内点）
    
    Args:
        x_data: list - x坐标
        y_data: list - y坐标
        coefficients: list - 拟合系数
        basis_functions: list - 基函数列表
    
    Returns:
        list - 残差列表
    """
    residuals = []
    
    for i in range(len(x_data)):
        y_pred = evaluate_fit(x_data[i], coefficients, basis_functions)
        residual = abs(y_data[i] - y_pred)
        residuals.append(residual)
    
    return residuals


def ransac_fit(x_data, y_data, basis_functions, max_iterations=1000, 
               threshold=0.1, min_samples=None, min_inliers_ratio=0.5, 
               seed=None):
    """
    RANSAC鲁棒拟合算法
    
    Args:
        x_data: list - 数据点x坐标
        y_data: list - 数据点y坐标（可能含噪声和离群点）
        basis_functions: list - 基函数列表
        max_iterations: int - 最大迭代次数
        threshold: float - 内点阈值（残差小于此值认为是内点）
        min_samples: int - 最小采样数（默认为基函数数量）
        min_inliers_ratio: float - 最小内点比例
        seed: int - 随机种子
    
    Returns:
        dict - {
            'coefficients': 拟合系数,
            'inliers': 内点索引列表,
            'n_inliers': 内点数量,
            'best_error': 最佳模型的平均误差
        }
    """
    if seed is not None:
        random.seed(seed)
    
    n_data = len(x_data)
    n_basis = len(basis_functions)
    
    if n_data != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    # 设置最小采样数
    if min_samples is None:
        min_samples = n_basis
    
    if min_samples < n_basis:
        raise ValueError(f"min_samples must be >= number of basis functions ({n_basis})")
    
    if n_data < min_samples:
        raise ValueError(f"Not enough data points. Need at least {min_samples}")
    
    best_coefficients = None
    best_inliers = []
    best_n_inliers = 0
    best_error = float('inf')
    
    # RANSAC主循环
    for iteration in range(max_iterations):
        # 1. 随机选择最小样本集
        sample_indices = random.sample(range(n_data), min_samples)
        x_sample = [x_data[i] for i in sample_indices]
        y_sample = [y_data[i] for i in sample_indices]
        
        try:
            # 2. 拟合模型
            coefficients = least_squares_fit(x_sample, y_sample, basis_functions)
            
            # 3. 计算所有点的残差
            residuals = compute_residuals(x_data, y_data, coefficients, basis_functions)
            
            # 4. 统计内点
            inliers = [i for i in range(n_data) if residuals[i] < threshold]
            n_inliers = len(inliers)
            
            # 5. 如果内点数量更多，更新最佳模型
            if n_inliers > best_n_inliers:
                # 计算内点的平均误差
                inlier_errors = [residuals[i] for i in inliers]
                avg_error = sum(inlier_errors) / n_inliers if n_inliers > 0 else float('inf')
                
                best_coefficients = coefficients
                best_inliers = inliers
                best_n_inliers = n_inliers
                best_error = avg_error
            
            # 早停条件：如果内点比例足够高
            if n_inliers >= n_data * min_inliers_ratio:
                break
        
        except Exception as e:
            # 如果拟合失败（例如矩阵奇异），跳过这次迭代
            continue
    
    # 使用所有内点重新拟合（可选，但通常能提高精度）
    if best_inliers and len(best_inliers) >= n_basis:
        x_inliers = [x_data[i] for i in best_inliers]
        y_inliers = [y_data[i] for i in best_inliers]
        
        try:
            best_coefficients = least_squares_fit(x_inliers, y_inliers, basis_functions)
            # 重新计算误差
            residuals = compute_residuals(x_data, y_data, best_coefficients, basis_functions)
            inlier_errors = [residuals[i] for i in best_inliers]
            best_error = sum(inlier_errors) / len(best_inliers)
        except:
            pass  # 如果重新拟合失败，使用之前的结果
    
    return {
        'coefficients': best_coefficients,
        'inliers': best_inliers,
        'n_inliers': best_n_inliers,
        'best_error': best_error,
        'inlier_ratio': best_n_inliers / n_data if n_data > 0 else 0
    }


def adaptive_threshold(y_data, percentile=75):
    """
    自适应阈值计算
    基于数据的分布自动确定RANSAC阈值
    
    Args:
        y_data: list - y坐标数据
        percentile: float - 百分位数（0-100）
    
    Returns:
        float - 建议的阈值
    """
    # 计算数据的范围和标准差（手工实现）
    n = len(y_data)
    
    # 均值
    mean_y = sum(y_data) / n
    
    # 标准差
    variance = sum((y - mean_y) ** 2 for y in y_data) / n
    std_y = math.sqrt(variance)
    
    # 排序并计算百分位数
    sorted_y = sorted(y_data)
    idx = int(n * percentile / 100)
    
    # 使用标准差的倍数作为阈值
    threshold = 2 * std_y  # 可调整的倍数
    
    return threshold


if __name__ == "__main__":
    # 测试RANSAC算法
    print("Testing RANSAC module...")
    
    import math
    
    # 生成测试数据：y = 2x + 1，加入噪声和离群点
    print("\n1. Testing RANSAC on linear data with outliers:")
    
    n_points = 50
    x_data = [i * 0.1 for i in range(n_points)]
    y_data = [2*x + 1 + random.gauss(0, 0.1) for x in x_data]  # 加入小噪声
    
    # 添加离群点
    outlier_indices = random.sample(range(n_points), 10)
    for idx in outlier_indices:
        y_data[idx] += random.uniform(5, 10) * random.choice([-1, 1])
    
    # 线性基函数
    basis_functions = [
        lambda x: 1,
        lambda x: x
    ]
    
    # 标准最小二乘拟合
    print("\nStandard Least Squares:")
    coef_standard = least_squares_fit(x_data, y_data, basis_functions)
    print(f"Coefficients: {[f'{c:.4f}' for c in coef_standard]}")
    residuals_standard = compute_residuals(x_data, y_data, coef_standard, basis_functions)
    print(f"Average residual: {sum(residuals_standard)/len(residuals_standard):.4f}")
    
    # RANSAC拟合
    print("\nRANSAC Least Squares:")
    ransac_result = ransac_fit(
        x_data, y_data, basis_functions,
        max_iterations=500,
        threshold=0.5,
        min_samples=2,
        seed=42
    )
    
    print(f"Coefficients: {[f'{c:.4f}' for c in ransac_result['coefficients']]}")
    print(f"Expected: [1.0000, 2.0000]")
    print(f"Number of inliers: {ransac_result['n_inliers']} / {n_points}")
    print(f"Inlier ratio: {ransac_result['inlier_ratio']:.2%}")
    print(f"Average inlier error: {ransac_result['best_error']:.4f}")
    print(f"Outliers detected: {n_points - ransac_result['n_inliers']}")
    print(f"True outliers: {len(outlier_indices)}")
    
    # 测试三角函数拟合
    print("\n2. Testing RANSAC on trigonometric data:")
    
    x_data = [i * 0.1 for i in range(100)]
    y_data = [math.sin(2*x) + 0.5*math.cos(x) + random.gauss(0, 0.05) 
              for x in x_data]
    
    # 添加离群点
    outlier_indices = random.sample(range(len(x_data)), 15)
    for idx in outlier_indices:
        y_data[idx] += random.uniform(2, 4) * random.choice([-1, 1])
    
    basis_functions = [
        lambda x: math.sin(2*x),
        lambda x: math.cos(x)
    ]
    
    ransac_result = ransac_fit(
        x_data, y_data, basis_functions,
        max_iterations=800,
        threshold=0.3,
        seed=42
    )
    
    print(f"Coefficients: {[f'{c:.4f}' for c in ransac_result['coefficients']]}")
    print(f"Expected: [1.0000, 0.5000]")
    print(f"Inlier ratio: {ransac_result['inlier_ratio']:.2%}")
    print(f"Outliers detected: {len(x_data) - ransac_result['n_inliers']}")
    print(f"True outliers: {len(outlier_indices)}")
    
    # 测试自适应阈值
    print("\n3. Testing adaptive threshold:")
    test_data = [1, 2, 3, 4, 5, 10, 15]  # 包含离群值
    threshold = adaptive_threshold(test_data)
    print(f"Data: {test_data}")
    print(f"Adaptive threshold: {threshold:.4f}")
    
    print("\nRANSAC module ready!")

