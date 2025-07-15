import csv, re

input_file = '/root/autodl-tmp/Wendell/Files/LM_Annotation_Seaching_all.csv'  # 文件路径
output_file = '/root/autodl-tmp/Wendell/Files/target_extracted.txt'       # 输出文件
written = set()

# 用于存储已写入的名称，避免重复写入
# 读取CSV文件，提取target列中的名称，并写入到输出文件中
with open(input_file, newline='', encoding='utf-8') as csvfile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        target = row['target'].strip()
        # 第一种格式：AF-xxx-F1-model_v4
        m1 = re.search(r'AF-(.*?)-F1-model_v4', target)
        if m1:
            name = m1.group(1)
            if name not in written:
                outfile.write(name + '\n')
                written.add(name)
            continue
        # 第二种格式：xxx_A_xxx
        m2 = re.match(r'^([A-Za-z0-9]+)_', target)
        if m2:
            name = m2.group(1)
            if name not in written:
                outfile.write(name + '\n')
                written.add(name)
            continue
        # 其他情况不输出