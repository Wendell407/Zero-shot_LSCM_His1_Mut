import csv
import re

def load_fasta(fasta_path):
    with open(fasta_path) as f:
        sequence = ''.join(line.strip() for line in f if not line.startswith('>'))
    return sequence

sa_cas9_seq = load_fasta("/root/autodl-tmp/Wendell/ESM2_embedding/fold_wt_sacas9_model_4.fasta")
sp_cas9_seq = load_fasta("/root/autodl-tmp/Wendell/ESM2_embedding/fold_wt_spcas9_model_4.fasta")

def process_mutations(input_csv, output_csv):
    with open(input_csv, 'r', encoding='utf-8-sig') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        # 清洗列名并确保包含All_Sequence
        original_fields = [f for f in reader.fieldnames if f]  # 过滤空列
        if 'All_Sequence' not in original_fields:
            original_fields.append('All_Sequence')
            
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        print("CSV 列名：", reader.fieldnames)
        for row_idx, row in enumerate(reader, 1):
            wt_type = row['Wild_Type']
            mut_sites = row['Mut_Site'].strip()
            
            # 选择序列
            if wt_type == 'SaCas9':
                sequence = list(sa_cas9_seq)
            elif wt_type == 'SpCas9':
                sequence = list(sp_cas9_seq)
            else:
                continue

            # 处理突变
            if mut_sites and mut_sites != 'WT':
                for mutation in re.split(r'\+|\s+', mut_sites):  # 支持+号和空格分隔
                    if not mutation:
                        continue
                    
                    try:
                        # 解析带字母的位置（如L1004Q）
                        match = re.match(r"([A-Za-z])(\d+)([A-Za-z])", mutation)
                        if match:
                            orig_aa, pos_str, new_aa = match.groups()
                            position = int(pos_str) - 1
                        else:
                            raise ValueError
                            
                        if 0 <= position < len(sequence):
                            sequence[position] = new_aa
                        else:
                            print(f"行 {row_idx}: 警告：跳过无效位置 {position+1}（总长度 {len(sequence)}）")
                            
                    except (ValueError, AttributeError) as e:
                        print(f"行 {row_idx}: 错误格式的突变 '{mutation}'，已跳过")

            # 写入结果
            row['All_Sequence'] = ''.join(sequence)
            cleaned_row = {k: row.get(k, '') for k in original_fields}
            writer.writerow(cleaned_row)

# 执行处理
process_mutations(
    "/root/autodl-tmp/Wendell/ESM2_embedding/主表：Cas9突变——效率数据整理.csv",
    "/root/autodl-tmp/Wendell/ESM2_embedding/spsa-alldata-frompapers.csv"
)
