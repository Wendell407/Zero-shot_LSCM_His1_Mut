import pandas as pd

df = pd.read_csv('/root/autodl-tmp/Wendell/CAS9_STRP1_Spencer_2017_positive_sub.csv')

# 使用正则表达式拆分mutant列
split_cols = df['mutant'].str.extract(r'([A-Z])(\d+)([A-Z])')
df.insert(1, 'WT', split_cols[0])  # 插入在原mutant列之后
df.insert(2, 'position', split_cols[1].astype(int))
df.insert(3, 'MUT', split_cols[2])

# 保存结果（覆盖原文件）
df.to_csv('/root/autodl-tmp/Wendell/CAS9_STRP1_Spencer_2017_positive_sub.csv', index=False)
