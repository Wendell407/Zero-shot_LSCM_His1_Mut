import pandas as pd
import json

# 读取json文件
with open('/root/autodl-tmp/Wendell/Data/Json/New_wet_all_with_combinations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转为DataFrame
df = pd.DataFrame(data)

# 只保留Variants和FITNESS两列（如果Variants缺失则填空）
df = df[['Variants', 'FITNESS', 'ALDOB_HPRT1', 'ALDOB_HSP90AB1', 'ALDOB_CTNNB1', 'HPRT1_HSP90AB1', 'HPRT1_CTNNB1', 'HSP90AB1_CTNNB1', 'ALDOB_HPRT1_HSP90AB1', 'ALDOB_HPRT1_CTNNB1', 'ALDOB_HSP90AB1_CTNNB1', 'HPRT1_HSP90AB1_CTNNB1']]

# 保存为csv
df.to_csv('/root/autodl-tmp/Wendell/Files/New_wet_all_with_combinations.csv', index=False)