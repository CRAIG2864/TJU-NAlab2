"""
Least squares curve fitting module
Hand-implemented without using numerical libraries
"""

import math
from core.matrix import Matrix, solve_least_squares


def polynomial_basis(degree):
    """
    生成多项式基函数列表
    
    Args:
        degree: int - 多项式最高次数
    Returns:
        list - 基函数列表 [f0, f1, f2, ...]
               其中 f0(x) = 1, f1(x) = x, f2(x) = x^2, ...
    """
    basis_functions = []
    for i in range(degree + 1):
        basis_functions.append(lambda x, power=i: x ** power)
    
    return basis_functions


def trigonometric_basis(params):
    """
    生成三角基函数（针对 g(x) = c*sin(dx) + e*cos(fx)）
    
    Args:
        params: dict - 包含 'd' 和 'f' 参数
    Returns:
        list - 基函数列表 [sin(dx), cos(fx)]
    """
    d = params.get('d', 1)
    f = params.get('f', 1)
    
    basis_functions = [
        lambda x, _d=d: math.sin(_d * x),
        lambda x, _f=f: math.cos(_f * x)
    ]
    
    return basis_functions


def custom_basis(function_list):
    """
    自定义基函数
    
    Args:
        function_list: list - lambda函数列表
    Returns:
        list - 基函数列表
    """
    return function_list


def least_squares_fit(x_data, y_data, basis_functions):
    """
    最小二乘拟合
    求解 min ||Ax - b||^2，其中A是设计矩阵
    
    Args:
        x_data: list - 数据点x坐标
        y_data: list - 数据点y坐标
        basis_functions: list - 基函数列表
    
    Returns:
        list - 拟合系数
    """
    n_data = len(x_data)
    n_basis = len(basis_functions)
    
    if n_data != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    if n_data < n_basis:
        raise ValueError("Number of data points must be >= number of basis functions")
    
    # 构造设计矩阵 A
    A_data = []
    for i in range(n_data):
        row = []
        for basis_func in basis_functions:
            row.append(basis_func(x_data[i]))
        A_data.append(row)
    
    A = Matrix(A_data)
    
    # 使用法方程求解
    coefficients = solve_least_squares(A, y_data)
    
    return coefficients


def evaluate_fit(x_eval, coefficients, basis_functions):
    """
    评估拟合函数在给定点的值
    
    Args:
        x_eval: float or list - 评估点
        coefficients: list - 拟合系数
        basis_functions: list - 基函数列表
    
    Returns:
        float or list - 拟合值
    """
    # 单点评估
    if isinstance(x_eval, (int, float)):
        result = 0.0
        for i, (coef, basis_func) in enumerate(zip(coefficients, basis_functions)):
            result += coef * basis_func(x_eval)
        return result
    
    # 多点评估
    results = []
    for x in x_eval:
        result = 0.0
        for coef, basis_func in zip(coefficients, basis_functions):
            result += coef * basis_func(x)
        results.append(result)
    
    return results


def piecewise_fit(x_data, y_data, intervals, degree):
    """
    分段多项式拟合
    
    Args:
        x_data: list - 数据点x坐标
        y_data: list - 数据点y坐标
        intervals: list - 区间列表 [(a1,b1), (a2,b2), ...]
        degree: int - 多项式次数
    
    Returns:
        list - 每段的系数列表
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    all_coefficients = []
    
    for idx, interval in enumerate(intervals):
        a, b = interval
        
        # 找到属于这个区间的数据点
        # 为避免边界点重复，除了最后一个区间，其他区间使用 [a, b)
        if idx == len(intervals) - 1:
            # 最后一个区间包含右端点
            indices = [i for i in range(len(x_data)) 
                      if a <= x_data[i] <= b]
        else:
            # 其他区间不包含右端点
            indices = [i for i in range(len(x_data)) 
                      if a <= x_data[i] < b]
        
        # 检查数据点是否足够
        if len(indices) < degree + 1:
            # 数据点不足的警告信息
            if len(indices) > 0:
                print(f"Warning: Only {len(indices)} points in interval [{a:.2f}, {b:.2f}], "
                      f"need at least {degree + 1} for degree {degree} polynomial. "
                      f"Using degree {min(degree, len(indices)-1)}.")
            
            # 自动调整次数
            actual_degree = max(0, min(degree, len(indices) - 1))
        else:
            actual_degree = degree
        
        if len(indices) == 0:
            # 没有数据点，使用零系数（这种情况应该很少见）
            print(f"Warning: No data points in interval [{a:.2f}, {b:.2f}]! Using zero coefficients.")
            coefficients = [0.0] * (actual_degree + 1)
        else:
            # 提取该区间的数据
            x_segment = [x_data[i] for i in indices]
            y_segment = [y_data[i] for i in indices]
            
            # 拟合多项式
            basis_functions = polynomial_basis(actual_degree)
            coefficients = least_squares_fit(x_segment, y_segment, basis_functions)
        
        all_coefficients.append({
            'interval': interval,
            'degree': actual_degree,
            'coefficients': coefficients
        })
    
    return all_coefficients


def evaluate_piecewise_fit(x_eval, piecewise_coefficients):
    """
    评估分段拟合函数
    
    Args:
        x_eval: float or list - 评估点
        piecewise_coefficients: list - 分段系数列表（来自piecewise_fit）
    
    Returns:
        float or list - 拟合值
    """
    # 单点评估
    if isinstance(x_eval, (int, float)):
        # 找到对应的区间
        for segment in piecewise_coefficients:
            a, b = segment['interval']
            if a <= x_eval <= b:
                degree = segment['degree']
                coefficients = segment['coefficients']
                basis_functions = polynomial_basis(degree)
                return evaluate_fit(x_eval, coefficients, basis_functions)
        
        # 如果不在任何区间内，返回0或抛出错误
        return 0.0
    
    # 多点评估
    results = []
    for x in x_eval:
        # 找到对应的区间
        found = False
        for segment in piecewise_coefficients:
            a, b = segment['interval']
            if a <= x <= b:
                degree = segment['degree']
                coefficients = segment['coefficients']
                basis_functions = polynomial_basis(degree)
                result = evaluate_fit(x, coefficients, basis_functions)
                results.append(result)
                found = True
                break
        
        if not found:
            results.append(0.0)
    
    return results


def create_intervals(a, b, n_segments):
    """
    将区间[a,b]等分成n_segments段
    
    Args:
        a, b: float - 区间端点
        n_segments: int - 分段数
    
    Returns:
        list - 区间列表 [(a1,b1), (a2,b2), ...]
    """
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    
    step = (b - a) / n_segments
    intervals = []
    
    for i in range(n_segments):
        start = a + i * step
        end = a + (i + 1) * step
        intervals.append((start, end))
    
    return intervals


if __name__ == "__main__":
    # 测试拟合模块
    print("Testing fitting module...")
    
    # 测试简单多项式拟合
    print("\n1. Testing polynomial fitting (y = 2x² + 3x + 1):")
    
    # 生成测试数据
    x_data = [i * 0.1 for i in range(-10, 11)]
    y_data = [2*x**2 + 3*x + 1 for x in x_data]
    
    # 二次多项式拟合
    basis = polynomial_basis(2)
    coef = least_squares_fit(x_data, y_data, basis)
    print(f"Fitted coefficients: {[f'{c:.4f}' for c in coef]}")
    print(f"Expected: [1.0000, 3.0000, 2.0000]")
    
    # 测试三角函数拟合
    print("\n2. Testing trigonometric fitting (y = sin(x) + 0.5*cos(2x)):")
    
    x_data = [i * 0.2 for i in range(0, 32)]  # 0 to 2π
    y_data = [math.sin(x) + 0.5 * math.cos(2*x) for x in x_data]
    
    # 使用sin(x)和cos(2x)作为基函数
    basis = [
        lambda x: math.sin(x),
        lambda x: math.cos(2*x)
    ]
    coef = least_squares_fit(x_data, y_data, basis)
    print(f"Fitted coefficients: {[f'{c:.4f}' for c in coef]}")
    print(f"Expected: [1.0000, 0.5000]")
    
    # 测试分段拟合
    print("\n3. Testing piecewise fitting:")
    
    x_data = [i * 0.1 for i in range(0, 41)]  # 0 to 4
    y_data = [x**2 if x < 2 else 4 + (x-2) for x in x_data]
    
    intervals = create_intervals(0, 4, 2)
    print(f"Intervals: {intervals}")
    
    piecewise_coef = piecewise_fit(x_data, y_data, intervals, degree=2)
    print(f"Number of segments: {len(piecewise_coef)}")
    
    for i, segment in enumerate(piecewise_coef):
        print(f"  Segment {i+1} {segment['interval']}: "
              f"degree={segment['degree']}, "
              f"coef={[f'{c:.4f}' for c in segment['coefficients']]}")
    
    print("\nFitting module ready!")

