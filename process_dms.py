import pandas as pd
from Bio import SeqIO

# 读取所有序列
fasta_path = "/root/autodl-tmp/Wendell/ESM2_embedding/sp_dms/mutant_sequences.fasta"
wild_seq = str(next(SeqIO.parse(fasta_path, "fasta")).seq)

# 读取突变数据
csv_path = "/root/autodl-tmp/Wendell/ESM2_embedding/sp_dms/sp_dms.csv"
df = pd.read_csv(csv_path)

# 读取所有FASTA记录到字典（键为突变名称）
fasta_records = {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}

# 创建结果列表
results = []

# 处理每个突变
for _, row in df.iterrows():
    mutation = row["mutation"]
    score = row["score"]
    
    # 直接获取对应突变的全长序列
    if mutation not in fasta_records:
        print(f"Warning: 未找到突变 {mutation} 的FASTA记录")
        continue
        
    full_sequence = fasta_records[mutation]
    
    # 移除原来的验证和上下文提取逻辑
    results.append(f"{mutation}\t{score}\t{full_sequence}")

# 写入结果文件
with open("mutation_context.tsv", "w") as f:
    f.write("\n".join(results))