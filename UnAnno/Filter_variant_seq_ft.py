import pandas as pd

# 读取原始文件（自动识别制表符分隔）
df = pd.read_csv('/root/autodl-tmp/Wendell/Files/GB1/random_selected_samples_zero×1000.csv', sep='\t')

# 随机抽取200条数据
sampled_df = df.sample(n=200, random_state=42)

# 保存为新的csv文件
sampled_df.to_csv('/root/autodl-tmp/Wendell/Files/GB1/Random/selected_samples_zero×1000_random200.csv', sep='\t', index=False)