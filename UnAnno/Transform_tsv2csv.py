import pandas as pd

# 读取 tsv 文件（制表符分隔）
tsv_path = "/root/autodl-tmp/Wendell/Data/Files/CSV/GB1/Random/selected_samples[2.920655-8.761966)×834.csv"
csv_path = "/root/autodl-tmp/Wendell/Data/Files/CSV/GB1/Random/[2.920655-8.761966)×834.csv"

df = pd.read_csv(tsv_path, sep='\t')
df.to_csv(csv_path, index=False, encoding='utf-8-sig')