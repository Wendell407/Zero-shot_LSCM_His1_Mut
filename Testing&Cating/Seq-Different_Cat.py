def find_sequence_diff(seq1, seq2):
    differences = []                                    # 初始化差异列表用于存储不同位置的字符
    min_len = min(len(seq1), len(seq2))                 # 取两个序列的最短长度
    for i in range(min_len):
        if seq1[i] != seq2[i]:
            differences.append((i, seq1[i], seq2[i]))   # 记录差异位置和字符
    for i in range(min_len, max(len(seq1), len(seq2))): # 处理剩余较长序列的部分
        val1 = seq1[i] if i < len(seq1) else None       # 超出部分用 None 填充
        val2 = seq2[i] if i < len(seq2) else None       # 超出部分用 None 填充
        differences.append((i, val1, val2))             # 记录差异
    return differences                                  # 返回所有差异

if __name__ == "__main__":
    seq1 = "GHJKLGHJK-W"
    seq2 = "GHJKL-JKLO"
    diffs = find_sequence_diff(seq1, seq2)
    print(f"✅ 对比结果（共{len(diffs)}处差异）:")
    print(f"📝 序列1长度: {len(seq1)}")
    print(f"📝 序列2长度: {len(seq2)}")
    print("----------差异位点:----------")
    for pos, val1, val2 in diffs:                       # 遍历所有差异
        print(f"⚠️  位置{pos}: {val1} → {val2}")        # 输出每个差异的详细信息