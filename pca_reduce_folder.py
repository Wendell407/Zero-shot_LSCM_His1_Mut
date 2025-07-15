import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

input_dir = '/root/autodl-tmp/Wendell/Data/Esm2_emb/nothing'    # 输入embedding文件夹

# 收集所有npy文件路径
embedding_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
print(f'共找到{len(embedding_files)}个embedding文件')

# 读取所有embedding，每个文件聚合成一个点
all_embeddings = []
for fname in embedding_files:
    arr = np.load(os.path.join(input_dir, fname))
    arr_mean = arr.mean(axis=0)  # 对每个文件的embedding取均值
    all_embeddings.append(arr_mean)
all_embeddings = np.stack(all_embeddings, axis=0)
print(f'拼接后总形状: {all_embeddings.shape}')

# 在读取所有embedding后添加TSNE预处理
print(f'原始数据形状: {all_embeddings.shape}')

# 先用t-SNE降到50维
# 修改TSNE参数部分
tsne = TSNE(n_components=min(50, all_embeddings.shape[0]-1),  # 自动适配最大可用维度
            random_state=42,  # 固定随机种子以确保可重复性
            perplexity=30,  # 保持默认值
            n_iter=1000,  # 增加迭代次数以提高收敛性
            n_iter_without_progress=300,  # 增加无进展迭代次数
            verbose=1,  # 打印进度信息
            metric='euclidean',  # 使用欧氏距离
            angle=0.5,  # 保持默认值
            n_jobs=-1,  # 使用所有可用CPU核心
            early_exaggeration=12.0,  # 保持默认值
            method='exact',
            init='random',  # 改用随机初始化
            learning_rate='auto')
            
embeddings_tsne = tsne.fit_transform(all_embeddings)
print(f'TSNE降维后形状: {embeddings_tsne.shape}')

# 修改后续PCA部分（替换原来的all_embeddings）
pca_2d = PCA(n_components=2)
embeddings_2d = pca_2d.fit_transform(embeddings_tsne)

# 修复二维图输出问题
plt.figure(figsize=(12, 9))
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], s=50, alpha=0.7)
for i, fname in enumerate(embedding_files):
    plt.text(embeddings_2d[i,0]+0.1, 
             embeddings_2d[i,1]+0.1,
             fname, 
             fontsize=8,
             alpha=0.8,
             va='center', 
             ha='left')

# 添加缺失的标题和标签设置
plt.title('PCA降到2维的散点图')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('pca_2d_scatter.png', dpi=200)
plt.close()  # 显式关闭当前figure

# 3D PCA同理修改
pca_3d = PCA(n_components=3)
embeddings_3d = pca_3d.fit_transform(embeddings_tsne)  # 改为使用TSNE降维后的数据

plt.figure(figsize=(8,6))
# 修改2D绘图部分
plt.figure(figsize=(12, 9))  # 增大画布尺寸
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], s=50, alpha=0.7)
for i, fname in enumerate(embedding_files):
    plt.text(embeddings_2d[i,0]+0.1,  # X偏移
             embeddings_2d[i,1]+0.1,  # Y偏移
             fname, 
             fontsize=8,  # 增大字号
             alpha=0.8,
             va='center', 
             ha='left')

# 修改3D绘图部分
fig = plt.figure(figsize=(12, 9))  # 增大3D画布尺寸
ax = fig.add_subplot(111, projection='3d')  # 添加axes初始化
ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], s=50, alpha=0.7)
for i, fname in enumerate(embedding_files):
    ax.text(embeddings_3d[i,0], 
            embeddings_3d[i,1], 
            embeddings_3d[i,2]+0.1,  # Z偏移
            fname, 
            fontsize=6,  # 3D使用稍小字号
            alpha=0.8,
            va='bottom')
ax.set_title('PCA降到3维的散点图')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.tight_layout()
plt.savefig('pca_3d_scatter.png', dpi=200)
plt.show()