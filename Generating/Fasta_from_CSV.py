import pandas as pd
df = pd.read_csv('/root/autodl-tmp/Wendell/Data/Files/CSV/GB1/New_selected4test.csv', encoding='utf-8-sig') # 读取csv文件

# 检查Variants对应的Sequence缺失行
missing_sequences = df[df['Sequence'].isna()]   # 筛选出 Sequence 列为空的行（NaN）
if not missing_sequences.empty:                 # 如果有缺失数据
    print("⚠️ 以下Variants的Sequence缺失:")
    print(missing_sequences['Variants'].tolist())

# 处理数据
df = df[['Variants', 'Sequence']]               # 保留的列
df = df.dropna(subset=['Sequence'])             # 去除缺失值（如果有）
df = df.drop_duplicates(subset=['Variants'])    # 去除重复项（如果有）

# 输出数据统计信息
print('📊 总数据行数:', len(df))
print('⚠️ 缺失项数:\n', df.isnull().sum())
print('🔍 重复项数:\n', df.duplicated().sum())
print('👀 数据预览:\n', df.head())
print('📝 数据描述:\n', df.describe())
print('ℹ️ 数据类型:\n', df.dtypes)
print('✅ 数据统计:\n', df['Sequence'].apply(len).describe())

# fasta输出部分
with open('/root/autodl-tmp/Wendell/Data/Sequence/Fasta/GB1_New_selected4test.fasta', 'w') as f:
    for idx, row in df.iterrows():
        f.write(f">{row['Variants']}\n{row['Sequence']}\n")
print('🎉 FASTA文件已保存!')