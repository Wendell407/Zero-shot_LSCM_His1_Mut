import os
from Bio import SeqIO

# 路径设置
input_fasta = "/root/autodl-tmp/Wendell/Files/Last_new_wet.fasta"
sa_wt = "/root/autodl-tmp/Wendell/Files/SaWT.fasta"
sp_wt = "/root/autodl-tmp/Wendell/Files/SpWT.fasta"
sa_out = "/root/autodl-tmp/Wendell/Files/Sa_seqs.fasta"
sp_out = "/root/autodl-tmp/Wendell/Files/Sp_seqs.fasta"

# 1. 提取Sa和Sp序列
sa_records = []
sp_records = []
for record in SeqIO.parse(input_fasta, "fasta"):
    if record.id.startswith("Sa"):
        sa_records.append(record)
    elif record.id.startswith("Sp"):
        sp_records.append(record)

SeqIO.write(sa_records, sa_out, "fasta")
SeqIO.write(sp_records, sp_out, "fasta")

# 3. 运行blastp
os.system(f"blastp -query {sa_wt} -subject {sa_out} -out Sa_vs_SaWT.blastp")
os.system(f"blastp -query {sp_wt} -subject {sp_out} -out Sp_vs_SpWT.blastp")