"""
Sampling points generation module
Supports random, Chebyshev nodes, and Legendre nodes
"""

import math
import random


def uniform_sampling(a, b, n):
    """
    均匀采样点
    Args:
        a, b: float - 区间端点
        n: int - 采样点个数
    Returns:
        list - 采样点
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    if n == 1:
        return [(a + b) / 2]
    
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def random_sampling(a, b, n, seed=None):
    """
    随机采样点（均匀分布）
    Args:
        a, b: float - 区间端点
        n: int - 采样点个数
        seed: int - 随机种子（可选）
    Returns:
        list - 排序后的采样点
    """
    if seed is not None:
        random.seed(seed)
    
    if n <= 0:
        raise ValueError("n must be positive")
    
    points = [random.uniform(a, b) for _ in range(n)]
    points.sort()  # 排序便于插值
    
    return points


def chebyshev_nodes(n, a=-1, b=1):
    """
    切比雪夫多项式零点（第一类）
    在区间[a,b]上生成n个切比雪夫节点
    
    公式: x_k = cos((2k-1)π/(2n)), k=1,2,...,n
    然后映射到[a,b]区间
    
    Args:
        n: int - 节点个数
        a, b: float - 目标区间
    Returns:
        list - 切比雪夫节点
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    nodes = []
    for k in range(1, n + 1):
        # 标准区间[-1,1]上的切比雪夫零点
        x_standard = math.cos((2 * k - 1) * math.pi / (2 * n))
        
        # 映射到[a,b]区间
        x_mapped = 0.5 * (a + b) + 0.5 * (b - a) * x_standard
        nodes.append(x_mapped)
    
    # 排序（从小到大）
    nodes.sort()
    
    return nodes


def legendre_nodes(n, a=-1, b=1):
    """
    勒让德多项式零点（高斯-勒让德求积节点）
    使用numpy.polynomial.legendre获取零点
    
    Args:
        n: int - 节点个数
        a, b: float - 目标区间
    Returns:
        list - 勒让德节点
    """
    try:
        # 这是唯一允许使用numpy的地方
        import numpy.polynomial.legendre as leg
        
        # 获取n阶勒让德多项式的零点和权重
        nodes_standard, _ = leg.leggauss(n)
        
        # 将节点从[-1,1]映射到[a,b]
        nodes = [0.5 * (a + b) + 0.5 * (b - a) * x for x in nodes_standard]
        
        # 转换为Python list并排序
        nodes = sorted([float(x) for x in nodes])
        
        return nodes
    
    except ImportError:
        raise ImportError("numpy is required for Legendre nodes computation")


def generate_dense_points(a, b, n=1000):
    """
    生成密集点用于绘图和误差评估
    Args:
        a, b: float - 区间端点
        n: int - 点的个数
    Returns:
        list - 密集点
    """
    return uniform_sampling(a, b, n)


if __name__ == "__main__":
    # 测试所有采样方法
    print("Testing sampling module...")
    
    a, b = -1, 1
    n = 5
    
    # 均匀采样
    uniform_points = uniform_sampling(a, b, n)
    print(f"\nUniform sampling ({n} points):")
    print([f"{x:.4f}" for x in uniform_points])
    
    # 随机采样
    random_points = random_sampling(a, b, n, seed=42)
    print(f"\nRandom sampling ({n} points):")
    print([f"{x:.4f}" for x in random_points])
    
    # 切比雪夫节点
    cheby_points = chebyshev_nodes(n, a, b)
    print(f"\nChebyshev nodes ({n} points):")
    print([f"{x:.4f}" for x in cheby_points])
    
    # 勒让德节点
    try:
        legendre_points = legendre_nodes(n, a, b)
        print(f"\nLegendre nodes ({n} points):")
        print([f"{x:.4f}" for x in legendre_points])
    except ImportError:
        print("\nLegendre nodes: numpy not available")
    
    print("\nSampling module ready!")





