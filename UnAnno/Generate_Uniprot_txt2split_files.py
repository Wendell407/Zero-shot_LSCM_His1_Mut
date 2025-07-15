def split_file_by_marker(input_file, marker='//'):
    import os
    output_dir = '/root/autodl-tmp/Wendell/Files/Uniprot/Split_txt/'
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    parts = []
    current_part = []
    for line in lines:
        if line.strip() == marker:
            if current_part:
                parts.append(current_part)
                current_part = []
        else:
            current_part.append(line)
    if current_part:
        parts.append(current_part)

    for idx, part in enumerate(parts, 1):
        output_filename = os.path.join(output_dir, f'output_{idx}.txt')
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            out_f.writelines(part)
        print(f'Written {output_filename}')

if __name__ == '__main__':
    split_file_by_marker('/root/autodl-tmp/Wendell/Files/Uniprot/idmapping_2025_06_04.txt')