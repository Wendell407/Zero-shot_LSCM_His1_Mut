# 将fasta文件中的条目和序列导出为csv，两列：条目名, 序列

import csv

fasta_path = "SpWT_ABC_combinations.fasta"
csv_path = "SpWT_ABC_combinations.csv"

with open(fasta_path) as fin, open(csv_path, "w", newline='') as fout:
    writer = csv.writer(fout)
    writer.writerow(["name", "sequence"])
    name = None
    seq_lines = []
    for line in fin:
        line = line.strip()
        if line.startswith(">"):
            if name:
                writer.writerow([name, ''.join(seq_lines)])
            name = line[1:]
            seq_lines = []
        else:
            seq_lines.append(line)
    if name:
        writer.writerow([name, ''.join(seq_lines)])

print(f"已导出为 {csv_path}")