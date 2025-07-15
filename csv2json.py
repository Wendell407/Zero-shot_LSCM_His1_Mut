import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    data = []
    
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        # 添加分隔符参数处理制表符
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        
        for row in csv_reader:
            # 修改字段名称映射
            new_row = {
                'Variants': row['mutation'],
                'FITNESS': float(row['score'])
            }
            data.append(new_row)
    
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        # 保持原有JSON格式输出
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    
    print(f"CSV文件已成功转换为JSON文件：{json_file_path}")

# 示例调用
csv_to_json('/root/autodl-tmp/Wendell/ESM2_embedding/sp_dms/sp_dms.csv', '/root/autodl-tmp/Wendell/ESM2_embedding/sp_dms/sp_dms.json')