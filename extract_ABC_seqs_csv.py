# 读取stru_split3-replacing.csv和Sp_seqs.fasta，将区间替换为对应序列，输出新csv

import csv
from collections import OrderedDict

def read_fasta(filepath):
    """读取fasta文件，返回{ID: 序列}字典"""
    seqs = OrderedDict()
    with open(filepath) as f:
        seq_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    seqs[seq_id] = ''.join(seq_lines)
                seq_id = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if seq_id:
            seqs[seq_id] = ''.join(seq_lines)
    return seqs

def parse_range(rng):
    """'774-820' -> (773, 820)  # python切片左闭右开"""
    start, end = map(int, rng.split('-'))
    return start-1, end

csv_path = "stru_split3-replacing.csv"
fasta_path = "Sp_seqs.fasta"
output_path = "stru_split3-replacing-seq.csv"

seqs = read_fasta(fasta_path)

with open(csv_path, newline='', encoding='utf-8-sig') as fin, \
     open(output_path, 'w', newline='') as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(['ID', 'A', 'B', 'C'])
    for row in reader:
        sid = row['ID']
        seq = seqs[sid]
        a_start, a_end = parse_range(row['A'])
        b_start, b_end = parse_range(row['B'])
        c_start, c_end = parse_range(row['C'])
        a_seq = seq[a_start:a_end]
        b_seq = seq[b_start:b_end]
        c_seq = seq[c_start:c_end]
        writer.writerow([sid, a_seq, b_seq, c_seq])

print(f"已生成：{output_path}")