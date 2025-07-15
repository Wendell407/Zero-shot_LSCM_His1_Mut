import pandas as pd

# 读取csv和tsv
df_rank = pd.read_csv('ranking_result_pos(1).csv')
df_blast = pd.read_csv('blastp.tsv', sep='\t', comment='#')

# 合并：将df_blast的Identity列按Alignment与pdb_id对应，添加到df_rank后面
df_merge = pd.merge(df_rank, df_blast[['Alignment', 'Identity']], left_on='pdb_id', right_on='Alignment', how='left')

# 删除多余的Alignment列
df_merge = df_merge.drop(columns=['Alignment'])

# 保存结果
df_merge.to_csv('ranking_result_with_identity_pos.csv', index=False)