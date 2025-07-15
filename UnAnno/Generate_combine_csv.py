import pandas as pd
import glob

# 指定要合并的csv文件路径（可根据实际情况修改）
file_list = [
    "/root/autodl-tmp/Wendell/Files/GB1/Random/selected_samples_[2.920655-8.761966)random1000_converted.csv",
    "/root/autodl-tmp/Wendell/Files/GB1/Random/selected_samples_(0.000000-2.920655)random1000_converted.csv",
    "/root/autodl-tmp/Wendell/Files/GB1/Random/selected_samples_zero×1000_random200_converted.csv"
]

# 读取并合并
dfs = [pd.read_csv(f) for f in file_list]
merged = pd.concat(dfs, ignore_index=True)

# 保存合并后的文件
merged.to_csv("/root/autodl-tmp/Wendell/Files/GB1/Random/selected_samples_merged.csv", index=False, encoding='utf-8-sig')