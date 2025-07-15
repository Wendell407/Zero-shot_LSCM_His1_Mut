import os
import pandas as pd

# 1. 读取 Last_wet_norm_all.csv，建立 Sample_ID 到 AVE 的映射
last_wet_path = '/root/autodl-tmp/Wendell/Data/Files/CSV/Last_wet_norm_all.csv'
last_wet = pd.read_csv(last_wet_path)
ave_map = dict(zip(last_wet['Sample_ID'], last_wet['AVE']))

# 2. 遍历所有 ranking_results_fold_*.csv 文件
root_dir = '/root/autodl-tmp/lc/results-stru-seq-3site-posneg'
for seed_dir in os.listdir(root_dir):
    if not seed_dir.startswith('SEED='):
        continue
    seed_path = os.path.join(root_dir, seed_dir)
    for fold_dir in os.listdir(seed_path):
        if not fold_dir.startswith('fold_'):
            continue
        fold_path = os.path.join(seed_path, fold_dir)
        for file in os.listdir(fold_path):
            if file.startswith('ranking_results_fold_') and file.endswith('.csv'):
                file_path = os.path.join(fold_path, file)
                df = pd.read_csv(file_path)

                # 3. 添加 AVE 列
                df['AVE'] = df['ID'].map(ave_map)

                # 4. 按 AVE 排序，添加“AVE排名”列（不覆盖原综合排名）
                df['AVE排名'] = df['AVE'].rank(method='min', ascending=False)
                df['AVE排名'] = df['AVE排名'].fillna(0).astype(int)

                # 5. 添加“是否正样本”列（AVE > 0.1 为正样本）
                df['是否正样本'] = (df['AVE'] > 0.1).astype(int)

                # 6. 添加“是否被模型识别”列
                # 规则：正样本的综合排名前没有负样本，且没有负样本与其综合排名并列
                df['是否被模型识别'] = 0
                for i in df[df['是否正样本'] == 1].index:
                    my_rank = df.at[i, '综合排名']
                    # 检查所有综合排名小于等于my_rank的样本
                    subset = df[df['综合排名'] <= my_rank]
                    # 如果这些样本中全是正样本，则被识别
                    if (subset['是否正样本'] == 1).all():
                        # 还要排除有负样本综合排名等于my_rank的情况
                        if not ((df['综合排名'] == my_rank) & (df['是否正样本'] == 0)).any():
                            df.at[i, '是否被模型识别'] = 1

                # 按 AVE排名 排序（AVE最大排第一）
                df = df.sort_values('AVE排名').reset_index(drop=True)

                # 7. 保存结果（覆盖原文件或另存为新文件），编码为GBK
                df.to_csv(file_path.replace('.csv', '_with_AVE.csv'), index=False, encoding='gbk')
                print(f'处理完成: {file_path}')
