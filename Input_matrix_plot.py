import numpy as np
import matplotlib.pyplot as plt

# 输入矩阵的行和列
rows = int(input("请输入矩阵的行数: "))
cols = int(input("请输入矩阵的列数: "))

# 输入矩阵元素
print("请输入矩阵的每一行，用空格分隔：")
matrix = []
for i in range(rows):
    row = list(map(float, input(f"第{i+1}行: ").split()))
    matrix.append(row)

matrix = np.array(matrix)

# 可视化矩阵
plt.imshow(matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("矩阵可视化")
plt.show()