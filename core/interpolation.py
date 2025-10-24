"""
Interpolation algorithms module
Newton interpolation with divided differences
"""


def newton_divided_difference(x_points, y_points):
    """
    计算牛顿差商表
    
    差商递推公式:
    f[x_i] = y_i
    f[x_i, x_{i+1}, ..., x_j] = (f[x_{i+1},...,x_j] - f[x_i,...,x_{j-1}]) / (x_j - x_i)
    
    Args:
        x_points: list - 插值节点x坐标
        y_points: list - 插值节点y坐标
    Returns:
        list - 差商表的第一列（即牛顿插值多项式的系数）
    """
    n = len(x_points)
    
    if n != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    
    # 初始化差商表（二维列表）
    # divided_diff[i][j] 表示 f[x_i, x_{i+1}, ..., x_{i+j}]
    divided_diff = [[0.0] * n for _ in range(n)]
    
    # 第0列：f[x_i] = y_i
    for i in range(n):
        divided_diff[i][0] = y_points[i]
    
    # 递推计算其他列
    for j in range(1, n):  # 列（差商的阶数）
        for i in range(n - j):  # 行
            numerator = divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]
            denominator = x_points[i + j] - x_points[i]
            
            if abs(denominator) < 1e-10:
                raise ValueError(f"Duplicate x values at indices {i} and {i+j}")
            
            divided_diff[i][j] = numerator / denominator
    
    # 返回第一行（包含所有需要的系数）
    coefficients = [divided_diff[0][j] for j in range(n)]
    
    return coefficients


def newton_interpolation_eval(x_points, coefficients, x_eval):
    """
    在给定点评估牛顿插值多项式
    使用嵌套乘法（霍纳法则的变体）提高效率
    
    P(x) = c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + ...
    
    Args:
        x_points: list - 插值节点
        coefficients: list - 差商系数
        x_eval: float - 要评估的点
    Returns:
        float - 插值多项式在x_eval处的值
    """
    n = len(coefficients)
    result = coefficients[n - 1]
    
    # 从后向前计算（嵌套乘法）
    for i in range(n - 2, -1, -1):
        result = result * (x_eval - x_points[i]) + coefficients[i]
    
    return result


def newton_interpolation_batch(x_points, y_points, x_eval_list):
    """
    批量评估牛顿插值多项式
    
    Args:
        x_points: list - 插值节点x坐标
        y_points: list - 插值节点y坐标
        x_eval_list: list - 要评估的点列表
    Returns:
        list - 插值结果
    """
    # 计算差商系数
    coefficients = newton_divided_difference(x_points, y_points)
    
    # 批量评估
    results = []
    for x in x_eval_list:
        y = newton_interpolation_eval(x_points, coefficients, x)
        results.append(y)
    
    return results


def lagrange_interpolation_eval(x_points, y_points, x_eval):
    """
    拉格朗日插值（作为备选方案，仅用于验证）
    
    L(x) = Σ y_i * l_i(x)
    其中 l_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
    
    Args:
        x_points: list - 插值节点x坐标
        y_points: list - 插值节点y坐标
        x_eval: float - 要评估的点
    Returns:
        float - 插值结果
    """
    n = len(x_points)
    result = 0.0
    
    for i in range(n):
        # 计算基函数 l_i(x)
        l_i = 1.0
        for j in range(n):
            if i != j:
                numerator = x_eval - x_points[j]
                denominator = x_points[i] - x_points[j]
                
                if abs(denominator) < 1e-10:
                    raise ValueError(f"Duplicate x values at indices {i} and {j}")
                
                l_i *= numerator / denominator
        
        result += y_points[i] * l_i
    
    return result


if __name__ == "__main__":
    # 测试插值算法
    print("Testing interpolation module...")
    
    # 测试简单函数 y = x^2
    x_points = [-1, 0, 1, 2]
    y_points = [1, 0, 1, 4]
    
    print(f"\nInterpolation nodes:")
    print(f"x: {x_points}")
    print(f"y: {y_points}")
    
    # 计算差商
    coefficients = newton_divided_difference(x_points, y_points)
    print(f"\nDivided difference coefficients: {[f'{c:.4f}' for c in coefficients]}")
    
    # 评估插值
    x_test = 0.5
    y_newton = newton_interpolation_eval(x_points, coefficients, x_test)
    y_lagrange = lagrange_interpolation_eval(x_points, y_points, x_test)
    y_true = x_test ** 2
    
    print(f"\nEvaluation at x = {x_test}:")
    print(f"Newton interpolation: {y_newton:.6f}")
    print(f"Lagrange interpolation: {y_lagrange:.6f}")
    print(f"True value (x²): {y_true:.6f}")
    print(f"Newton error: {abs(y_newton - y_true):.6e}")
    print(f"Lagrange error: {abs(y_lagrange - y_true):.6e}")
    
    # 批量测试
    x_eval_list = [-0.5, 0, 0.5, 1.5]
    y_interp = newton_interpolation_batch(x_points, y_points, x_eval_list)
    print(f"\nBatch evaluation:")
    for x, y in zip(x_eval_list, y_interp):
        print(f"  x = {x:5.2f}, y_interp = {y:7.4f}, y_true = {x**2:7.4f}")
    
    print("\nInterpolation module ready!")





