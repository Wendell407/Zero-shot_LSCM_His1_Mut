# 用于打印fasta文件中所有序列的条目名称（即以>开头的行）

fasta_path = "SpWT_ABC_combinations.fasta"  # 或替换为你的fasta文件名

with open(fasta_path) as f:
    for line in f:
        if line.startswith(">"):
            print(line.strip())