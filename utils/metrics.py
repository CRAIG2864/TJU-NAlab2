"""
Metrics and error calculation module
Hand-implemented without numpy
"""

import math
import random


def max_error(y_true, y_pred):
    """
    计算最大绝对误差
    Args:
        y_true: list - 真实值
        y_pred: list - 预测值
    Returns:
        float - 最大绝对误差
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")
    
    max_err = 0.0
    for i in range(len(y_true)):
        err = abs(y_true[i] - y_pred[i])
        if err > max_err:
            max_err = err
    
    return max_err


def mse(y_true, y_pred):
    """
    计算均方误差 (Mean Squared Error)
    Args:
        y_true: list - 真实值
        y_pred: list - 预测值
    Returns:
        float - MSE
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")
    
    n = len(y_true)
    sum_squared_error = 0.0
    
    for i in range(n):
        error = y_true[i] - y_pred[i]
        sum_squared_error += error * error
    
    return sum_squared_error / n


def rmse(y_true, y_pred):
    """
    计算均方根误差 (Root Mean Squared Error)
    Args:
        y_true: list - 真实值
        y_pred: list - 预测值
    Returns:
        float - RMSE
    """
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """
    计算平均绝对误差 (Mean Absolute Error)
    Args:
        y_true: list - 真实值
        y_pred: list - 预测值
    Returns:
        float - MAE
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")
    
    n = len(y_true)
    sum_abs_error = 0.0
    
    for i in range(n):
        sum_abs_error += abs(y_true[i] - y_pred[i])
    
    return sum_abs_error / n


def r_squared(y_true, y_pred):
    """
    计算决定系数 R^2
    Args:
        y_true: list - 真实值
        y_pred: list - 预测值
    Returns:
        float - R^2 值
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")
    
    # 计算均值
    n = len(y_true)
    y_mean = sum(y_true) / n
    
    # 总平方和
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    
    # 残差平方和
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1.0 - (ss_res / ss_tot)


def add_noise(y_data, noise_level=0.1, noise_type='gaussian', outlier_ratio=0.0):
    """
    向数据添加噪声
    Args:
        y_data: list - 原始数据
        noise_level: float - 噪声水平（相对于数据范围）
        noise_type: str - 噪声类型 'gaussian', 'uniform', 'outliers'
        outlier_ratio: float - 离群点比例（0-1）
    Returns:
        list - 添加噪声后的数据
    """
    y_noisy = y_data[:]
    
    # 计算数据范围
    y_min = min(y_data)
    y_max = max(y_data)
    data_range = y_max - y_min
    
    if noise_type == 'gaussian':
        # 高斯噪声
        sigma = noise_level * data_range
        for i in range(len(y_noisy)):
            y_noisy[i] += random.gauss(0, sigma)
    
    elif noise_type == 'uniform':
        # 均匀噪声
        amplitude = noise_level * data_range
        for i in range(len(y_noisy)):
            y_noisy[i] += random.uniform(-amplitude, amplitude)
    
    elif noise_type == 'outliers':
        # 添加离群点
        n_outliers = int(len(y_data) * outlier_ratio)
        outlier_indices = random.sample(range(len(y_data)), n_outliers)
        
        for idx in outlier_indices:
            # 离群点偏离较大
            outlier_value = y_data[idx] + random.uniform(-5, 5) * data_range
            y_noisy[idx] = outlier_value
    
    return y_noisy


def add_perturbation(y_data, perturbation_level=0.05):
    """
    添加小扰动（用于实验2的扰动分析）
    Args:
        y_data: list - 原始数据
        perturbation_level: float - 扰动水平（百分比）
    Returns:
        list - 扰动后的数据
    """
    y_perturbed = []
    for y in y_data:
        # 相对扰动
        perturbation = y * random.uniform(-perturbation_level, perturbation_level)
        y_perturbed.append(y + perturbation)
    
    return y_perturbed


def compute_residuals(y_true, y_pred):
    """
    计算残差
    Args:
        y_true: list - 真实值
        y_pred: list - 预测值
    Returns:
        list - 残差
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")
    
    return [y_true[i] - y_pred[i] for i in range(len(y_true))]


if __name__ == "__main__":
    # 简单测试
    print("Testing metrics module...")
    
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]
    
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"Max Error: {max_error(y_true, y_pred):.4f}")
    print(f"MSE: {mse(y_true, y_pred):.4f}")
    print(f"RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"MAE: {mae(y_true, y_pred):.4f}")
    print(f"R²: {r_squared(y_true, y_pred):.4f}")
    
    # 测试添加噪声
    y_noisy = add_noise(y_true, noise_level=0.1, noise_type='gaussian')
    print(f"\nOriginal: {y_true}")
    print(f"With Gaussian noise: {[f'{y:.2f}' for y in y_noisy]}")
    
    print("\nMetrics module ready!")





