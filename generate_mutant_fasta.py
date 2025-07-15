from Bio import SeqIO
import pandas as pd

# 读取野生型序列
wild_seq = str(next(SeqIO.parse("AF-Q99ZW2-F1-model_v4.fasta", "fasta")).seq)

# 读取突变数据
df = pd.read_csv("sp_dms.csv")

# 生成突变序列
with open("mutant_sequences.fasta", "w") as f:
    for _, row in df.iterrows():
        mut = row["mutation"]
        
        try:
            # 解析突变位置 (1-based索引)
            pos = int(mut[1:-1]) - 1  # 转换为0-based索引
            wt = mut[0]
            mt = mut[-1]
        except Exception as e:
            print(f"Invalid mutation format: {mut}")
            continue
        
        # 验证位置有效性
        if pos < 0 or pos >= len(wild_seq):
            print(f"Invalid position {pos+1} for mutation {mut}")
            continue
        
        # 验证野生型氨基酸
        if wild_seq[pos] != wt:
            print(f"Warning: Position {pos+1} has {wild_seq[pos]} instead of {wt}")
            continue
        
        # 创建突变序列
        mutant_seq = list(wild_seq)
        mutant_seq[pos] = mt
        mutant_seq = "".join(mutant_seq)
        
        # 写入FASTA格式
        f.write(f">{mut}\n{mutant_seq}\n")
