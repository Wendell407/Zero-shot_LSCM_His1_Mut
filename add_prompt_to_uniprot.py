import os

# 你的prompt文件路径
prompt_path = '/root/autodl-tmp/Wendell/Files/Prompt.ini'
# 需要批量处理的文件夹
target_folder = '/root/autodl-tmp/Wendell/Files/Uniprot/'

# 读取prompt内容
with open(prompt_path, 'r', encoding='utf-8') as f:
    prompt = f.read().strip() + '\n\n'

# 遍历目标文件夹下所有txt文件
for filename in os.listdir(target_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(target_folder, filename)
        # 读取原文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 写入prompt+原内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(prompt + content)

print("批量添加完成。")