import numpy as np

file1 = '/root/autodl-tmp/Wendell/Data/Esm2_emb/nothing/Sp33.npy'
file2 = '/root/autodl-tmp/Wendell/Data/Esm2_emb/nothing/Sp31.npy'

emb1 = np.load(file1)
emb2 = np.load(file2)

# 检查整体是否完全相同
print("整体是否完全相同：", np.all(emb1 == emb2))

# 检查均值向量是否完全相同
mean1 = emb1.mean(axis=0)
mean2 = emb2.mean(axis=0)
print("均值向量是否完全相同：", np.all(mean1 == mean2))

# 如果不完全相同，输出最大/最小差异
if not np.all(emb1 == emb2):
    print("最大绝对差异：", np.max(np.abs(emb1 - emb2)))
    print("最小绝对差异：", np.min(np.abs(emb1 - emb2)))
if not np.all(mean1 == mean2):
    print("均值向量最大绝对差异：", np.max(np.abs(mean1 - mean2)))