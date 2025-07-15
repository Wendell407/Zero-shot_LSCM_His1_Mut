import numpy as np
import matplotlib.pyplot as plt

# 直接定义矩阵数据
matrix = np.array([
    [0, 1.890, 3.071, 4.790, 2.945],
    [1.890, 0, 2.612, 1.578, 1.753],
    [3.071, 2.612, 0, 3.448, 2.184],
    [4.790, 1.578, 3.448, 0, 2.385],
    [2.945, 1.753, 2.184, 2.385, 0]
])

# 定义行列名
labels = ['AF3-Predict', 'AFDB(query)', 'AF2-Predict', '4oo8', '5f9r']

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(matrix, cmap='Blues', interpolation='nearest')  # 使用单色系

# 设置坐标轴刻度和标签
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=13, rotation=30, ha='left')
ax.set_yticklabels(labels, fontsize=13)

# 设置x轴表头在顶部
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# 在每个格子中标注浮点数
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, f"{matrix[i, j]:.3f}", ha='center', va='center',
                color='black' if matrix[i, j] > matrix.max()/2 else 'white', fontsize=12)

# 添加色条
plt.colorbar(im, ax=ax)
plt.title("RMSD_Score", pad=50, fontsize=15)

plt.tight_layout()
plt.savefig("matrix.png", dpi=300, bbox_inches='tight')
plt.show()