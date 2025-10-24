"""
Matrix operations module - Hand-implemented matrix class
No numpy for matrix operations, only pure Python
"""

class Matrix:
    """手工实现的矩阵类，用于线性代数运算"""
    
    def __init__(self, data):
        """
        初始化矩阵
        Args:
            data: 嵌套列表 [[row1], [row2], ...]
        """
        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty")
        
        self.rows = len(data)
        self.cols = len(data[0])
        
        # 验证所有行长度相同
        for row in data:
            if len(row) != self.cols:
                raise ValueError("All rows must have the same length")
        
        # 深拷贝数据
        self.data = [[float(val) for val in row] for row in data]
    
    def __getitem__(self, index):
        """获取元素或行"""
        return self.data[index]
    
    def __setitem__(self, index, value):
        """设置元素或行"""
        self.data[index] = value
    
    def __repr__(self):
        """字符串表示"""
        return f"Matrix({self.rows}x{self.cols})"
    
    def shape(self):
        """返回矩阵形状"""
        return (self.rows, self.cols)
    
    def copy(self):
        """深拷贝矩阵"""
        return Matrix([row[:] for row in self.data])
    
    def transpose(self):
        """矩阵转置"""
        transposed = [[self.data[i][j] for i in range(self.rows)] 
                      for j in range(self.cols)]
        return Matrix(transposed)
    
    def scalar_multiply(self, scalar):
        """标量乘法"""
        result = [[self.data[i][j] * scalar for j in range(self.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    def add(self, other):
        """矩阵加法"""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions")
        
        result = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)
    
    def subtract(self, other):
        """矩阵减法"""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions")
        
        result = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)
    
    def multiply(self, other):
        """矩阵乘法"""
        if self.cols != other.rows:
            raise ValueError(f"Cannot multiply {self.rows}x{self.cols} by {other.rows}x{other.cols}")
        
        result = [[sum(self.data[i][k] * other.data[k][j] 
                      for k in range(self.cols))
                  for j in range(other.cols)]
                 for i in range(self.rows)]
        return Matrix(result)
    
    def solve_linear_system(self, b):
        """
        求解线性方程组 Ax = b，使用高斯消元法（列主元）
        Args:
            b: Matrix or list - 右端向量
        Returns:
            list - 解向量
        """
        if isinstance(b, list):
            b = Matrix([[val] for val in b])
        
        if self.rows != b.rows:
            raise ValueError("Incompatible dimensions")
        
        # 创建增广矩阵 [A|b]
        augmented = [self.data[i][:] + b.data[i] for i in range(self.rows)]
        n = self.rows
        
        # 前向消元（列主元）
        for col in range(n):
            # 选择列主元
            max_row = col
            for row in range(col + 1, n):
                if abs(augmented[row][col]) > abs(augmented[max_row][col]):
                    max_row = row
            
            # 交换行
            augmented[col], augmented[max_row] = augmented[max_row], augmented[col]
            
            # 检查奇异性
            if abs(augmented[col][col]) < 1e-10:
                raise ValueError("Matrix is singular or nearly singular")
            
            # 消元
            for row in range(col + 1, n):
                factor = augmented[row][col] / augmented[col][col]
                for j in range(col, n + 1):
                    augmented[row][j] -= factor * augmented[col][j]
        
        # 回代
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = augmented[i][n]
            for j in range(i + 1, n):
                x[i] -= augmented[i][j] * x[j]
            x[i] /= augmented[i][i]
        
        return x
    
    def to_list(self):
        """转换为嵌套列表"""
        return [row[:] for row in self.data]
    
    def to_vector(self):
        """如果是列向量，转换为一维列表"""
        if self.cols != 1:
            raise ValueError("Not a column vector")
        return [self.data[i][0] for i in range(self.rows)]
    
    @staticmethod
    def identity(n):
        """创建n×n单位矩阵"""
        data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return Matrix(data)
    
    @staticmethod
    def from_vector(vec):
        """从一维列表创建列向量"""
        return Matrix([[val] for val in vec])


def solve_least_squares(A, b):
    """
    求解最小二乘问题 min ||Ax - b||^2
    通过法方程 A^T A x = A^T b
    
    Args:
        A: Matrix - 设计矩阵
        b: list or Matrix - 观测向量
    Returns:
        list - 最小二乘解
    """
    if isinstance(b, list):
        b = Matrix.from_vector(b)
    
    # 计算 A^T A
    AT = A.transpose()
    ATA = AT.multiply(A)
    
    # 计算 A^T b
    ATb = AT.multiply(b)
    
    # 求解法方程
    x = ATA.solve_linear_system(ATb)
    
    return x


if __name__ == "__main__":
    # 简单测试
    print("Testing Matrix class...")
    
    # 测试基本运算
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    
    print(f"A = {A.to_list()}")
    print(f"B = {B.to_list()}")
    print(f"A + B = {A.add(B).to_list()}")
    print(f"A * B = {A.multiply(B).to_list()}")
    print(f"A^T = {A.transpose().to_list()}")
    
    # 测试线性方程组求解
    A = Matrix([[2, 1], [1, 3]])
    b = [5, 6]
    x = A.solve_linear_system(b)
    print(f"\nSolve 2x + y = 5, x + 3y = 6")
    print(f"Solution: x = {x}")
    
    print("\nMatrix module ready!")




