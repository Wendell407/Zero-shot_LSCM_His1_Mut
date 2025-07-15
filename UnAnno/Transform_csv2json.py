import csv, json  # 导入csv和json模块，用于处理CSV和JSON文件

# 定义字典，以将CSV列名映射为JSON键名
col_map = {'Variants': '编号',                     # CSV的'Variants'列映射为JSON的'Sample_ID'
           'ALDOB': 'Norm_ALDOB',             # CSV的'ALDOB'列映射为JSON的'Modified_dali_1ALDOB'
           'HPRT1': 'Norm_HPRT1',             # 依此类推
           'HSP90AB1': 'Norm_HSP90AB1',       # 可按表格需求更换
           'CTNNB1': 'Norm_CTNNB1',
           'FITNESS': 'AVE'}

# 定义函数，将CSV转换为JSON
def csv_to_json(csv_path, json_path):                   # 打开CSV，指定编码为utf-8-sig，防止BOM头影响读取
    with open(csv_path, encoding='utf-8-sig') as f:
        data = []                                       # 用于存储每一行转换后的数据
        for i, row in enumerate(csv.DictReader(f), 1):  # i为行号，从1开始，使用csv.DictReader按行读取CSV，每行是一个字典
            try:                                        # 根据col_map，提取出每列数据组成新字典（键为col_map的key，值为CSV对应列）
                                                        # 'Variants'列保留字符串，其余列转换为float类型
                data.append({k: row[v].strip() if k == 'Variants' else float(row[v]) for k, v in col_map.items()})
            except Exception as e:
                print(f"跳过第{i}无效行，原因：{e}，内容：{row}")
    with open(json_path, 'w', encoding='utf-8') as f:   # 打开JSON，写入转换后的数据，格式化缩进为4，支持中文
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"已保存为JSON：{json_path}")

# 调用函数，指定CSV和JSON文件的路径
csv_to_json(
    '/root/autodl-tmp/Wendell/Files/Wet_1st.csv',      # 输入的CSV文件路径
    '/root/autodl-tmp/Wendell/Data/Json/Wet_1st.json'   # 输出的JSON文件路径
)

# # ****************************定义一个新的函数，处理CSV文件并计算组合的均值***************************

# def csv_to_json_with_combinations(csv_path, json_path):
#     """
#     读取CSV文件，将每行的4个基因列（ALDOB、HPRT1、HSP90AB1、CTNNB1）分别两两、三三组合求平均，
#     并输出到JSON文件。Variants和FITNESS列原样输出，同时保留原始基因值。
#     """
#     from itertools import combinations

#     gene_keys = ['ALDOB', 'HPRT1', 'HSP90AB1', 'CTNNB1']
#     gene_cols = [col_map[k] for k in gene_keys]

#     with open(csv_path, encoding='utf-8-sig') as f:
#         data = []
#         for i, row in enumerate(csv.DictReader(f), 1):
#             try:
#                 # 新建一个字典，先放入Variants和FITNESS（用映射后的字段名）
#                 item = {
#                     'Variants': row[col_map['Variants']].strip(),
#                     'FITNESS': float(row[col_map['FITNESS']])
#                 }
#                 # 取出4个基因的值，转为float，按顺序存入列表，并保留原始值
#                 gene_values = []
#                 for idx, col in enumerate(gene_cols):
#                     val = row[col].strip()
#                     if val == '':
#                         raise ValueError(f"{col} 列为空")
#                     float_val = float(val)
#                     gene_values.append(float_val)
#                     # 保留原始基因值
#                     item[gene_keys[idx]] = float_val

#                 # 生成所有两两组合，求平均
#                 for comb in combinations(range(4), 2):
#                     key = f"{gene_keys[comb[0]]}_{gene_keys[comb[1]]}"
#                     avg = (gene_values[comb[0]] + gene_values[comb[1]]) / 2
#                     item[key] = avg

#                 # 生成所有三三组合，求平均
#                 for comb in combinations(range(4), 3):
#                     key = f"{gene_keys[comb[0]]}_{gene_keys[comb[1]]}_{gene_keys[comb[2]]}"
#                     avg = (gene_values[comb[0]] + gene_values[comb[1]] + gene_values[comb[2]]) / 3
#                     item[key] = avg

#                 data.append(item)
#             except Exception as e:
#                 print(f"跳过第{i}无效行，原因：{e}，内容：{row}")

#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=4, ensure_ascii=False)
#     print(f"已保存为JSON：{json_path}")
#     print("两两、三三组合的均值和原始值已加入输出。")

# # 调用带组合均值的函数，输出到不同的JSON文件
# csv_to_json_with_combinations(
#     '/root/autodl-tmp/Wendell/Files/Last_wet_norm_all.csv',
#     '/root/autodl-tmp/Wendell/Data/Json/New_wet_all_with_combinations.json'
# )