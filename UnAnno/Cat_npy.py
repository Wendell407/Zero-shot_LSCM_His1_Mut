import numpy as np

# 加载.npy文件
data = np.load('/root/autodl-tmp/Wendell/Data/Af2_emb/Last_wet/All_seq/Pair_representation_l2norm/Sp01_D.npy') #, allow_pickle=True) #.item()
# np.set_printoptions(threshold=np.inf, linewidth=200)  # 显示全部内容

# 打印数组信息
print("数组形状:", data.shape)
print("数据类型:", data.dtype)
print("数组内容:")
print(data)

# # 保存数组内容到文件
# with open('/root/autodl-tmp/Wendell/output4.txt', 'w') as f:
#     f.write(repr(data))

# # 如果需要保存多个切片，可以使用以下代码
# # with open('output.txt', 'w') as f:
# #     for i, arr in enumerate(data[:3]):  # 只写前3个[:3]切片
# #         f.write(f"第{i}个切片：\n")
# #         f.write(repr(arr) + "\n\n")

# # print("数组内容已完整保存到 output.txt")

# # print("字典键:", data.keys())
# # print("final_atom_mask 形状:", data['final_atom_mask'].shape)
# # print("final_atom_positions 形状:", data['final_atom_positions'].shape)

# import os
# import numpy as np
# from pathlib import Path
# import sys

# def get_npy_stats(file_path):
#     """获取npy文件的shape、最大值和最小值"""
#     try:
#         data = np.load(file_path)
#         shape = data.shape
        
#         # 计算最大值和最小值
#         max_val = np.max(data)
#         min_val = np.min(data)
        
#         return shape, max_val, min_val
#     except Exception as e:
#         return f"错误: {str(e)}", None, None

# def main():
#     if len(sys.argv) < 2:
#         print("用法: python script.py <npy文件路径> 或 <包含npy的目录>")
#         sys.exit(1)
    
#     path = Path(sys.argv[1])
    
#     if not path.exists():
#         print(f"错误: 路径 {path} 不存在")
#         sys.exit(1)
    
#     if path.is_file() and path.suffix == '.npy':
#         shape, max_val, min_val = get_npy_stats(path)
#         if isinstance(shape, str):
#             print(f"{path}: {shape}")
#         else:
#             print(f"{path}: shape={shape}, 最大值={max_val}, 最小值={min_val}")
    
#     elif path.is_dir():
#         npy_files = sorted(path.glob("*.npy"))
#         if not npy_files:
#             print(f"在目录 {path} 中未找到npy文件")
#             sys.exit(0)
        
#         print(f"在 {path} 中找到 {len(npy_files)} 个npy文件:")
#         for file in npy_files:
#             shape, max_val, min_val = get_npy_stats(file)
#             if isinstance(shape, str):
#                 print(f"{file}: {shape}")
#             else:
#                 print(f"{file}: shape={shape}, 最大值={max_val}, 最小值={min_val}")
    
#     else:
#         print(f"错误: {path} 不是npy文件或目录")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()
