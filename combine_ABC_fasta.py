import csv
from itertools import product
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
    """将'774-820'转为Python索引（0-based, 左闭右闭）"""
    start, end = map(int, rng.split('-'))
    return start-1, end  # python切片左闭右开

# 1. 读取表格，收集所有A/B/C片段的来源信息
csv_path = "stru_split3-replacing.csv"
fasta_path = "Sp_seqs.fasta"
output_path = "SpWT_ABC_combinations.fasta"

ABC_list = []
with open(csv_path, newline='', encoding='utf-8-sig') as f:  # 增加 encoding
    reader = csv.DictReader(f)
    for row in reader:
        ABC_list.append({
            'ID': row['ID'],
            'A': row['A'],
            'B': row['B'],
            'C': row['C']
        })

# 2. 读取所有蛋白序列
seqs = read_fasta(fasta_path)

# 3. 提取所有A/B/C片段
A_seqs, B_seqs, C_seqs = [], [], []
for entry in ABC_list:
    sid = entry['ID']
    seq = seqs[sid]
    a_start, a_end = parse_range(entry['A'])
    b_start, b_end = parse_range(entry['B'])
    c_start, c_end = parse_range(entry['C'])
    A_seqs.append((sid, entry['A'], seq[a_start:a_end]))
    B_seqs.append((sid, entry['B'], seq[b_start:b_end]))
    C_seqs.append((sid, entry['C'], seq[c_start:c_end]))

# 4. 获取SpWT模板的前、后片段
spwt_seq = seqs['SpWT']
pre_start = 0
pre_end = 773  # 1-based 773, python索引0-772
post_start = 900  # 1-based 901, python索引900
pre_seq = spwt_seq[pre_start:pre_end]
post_seq = spwt_seq[post_start:]

# 5. 生成所有排列组合并写入fasta
with open(output_path, 'w') as out:
    for (A_id, A_rng, A_seq), (B_id, B_rng, B_seq), (C_id, C_rng, C_seq) in product(A_seqs, B_seqs, C_seqs):
        # Sp07的C片段只允许与Sp07_A和Sp07_B组合，否则跳过
        if C_id == "Sp07" :#and (A_id != "Sp07" or B_id != "Sp07"):
            continue
        if A_id == "Sp19" or C_id == "Sp19":
            continue
        # 组合名称
        name = f"{A_id}_A-{B_id}_B-{C_id}_C"
        # 拼接新序列
        new_seq = pre_seq + A_seq + B_seq + C_seq + post_seq
        # 写入fasta
        out.write(f">{name}\n")
        # 每60字符换行
        for i in range(0, len(new_seq), 60):
            out.write(new_seq[i:i+60] + "\n")

print(f"已生成所有ABC组合的fasta文件：{output_path}")