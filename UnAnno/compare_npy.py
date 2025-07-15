# 导入必要的库
import numpy as np

def compare_npy(file1, file2, decimal=6):
    """
    对比两个Numpy二进制文件的核心函数
    
    参数：
    file1 -- 第一个.npy文件路径
    file2 -- 第二个.npy文件路径 
    decimal -- 比较精度（小数点后位数），默认6位
    """
    
    # 加载numpy二进制文件
    arr1 = np.load(file1)  # 加载第一个数组
    arr2 = np.load(file2)  # 加载第二个数组
    
    # 打印数组形状信息
    print(f"文件1形状: {arr1.shape}\n文件2形状: {arr2.shape}")
    
    # 形状一致性检查
    if arr1.shape != arr2.shape:
        print("错误：数组形状不匹配")
        return  # 提前终止函数
    
    # 创建差异掩码（使用numpy的近似比较方法）
    # atol: 绝对容忍度，rtol: 相对容忍度
    diff_mask = ~np.isclose(arr1, arr2) #, 
                        #    atol=10**-decimal,  # 绝对误差容忍度，例如1e-6
                        #    rtol=10**-decimal)   # 相对误差容忍度
    
    # 统计差异总数
    diff_count = np.sum(diff_mask)  # True会被视为1，False为0
    
    # 输出差异分析报告
    print(f"\n差异分析（精度: 1e-{decimal}）:")
    print(f"不同元素数量: {diff_count}/{arr1.size} ({diff_count/arr1.size:.2%})")
    print(f"最大绝对差异: {np.max(np.abs(arr1 - arr2)):.4e}")  # 科学计数法显示
    print(f"平均绝对差异: {np.mean(np.abs(arr1 - arr2)):.4e}")
    
    # 如果有差异则保存详细结果
    if diff_count > 0:
        # 获取所有差异位置的索引
        diff_indices = np.argwhere(diff_mask)
        
        # 将结果保存为压缩的npz文件
        np.savez("diff_results.npz",   # 输出文件名
                indices=diff_indices,  # 差异位置索引
                values1=arr1[diff_mask],  # 文件1中的差异值
                values2=arr2[diff_mask])  # 文件2中的差异值

if __name__ == "__main__":
    # 示例使用（实际路径需要根据具体情况修改）
    compare_npy(
        "/root/autodl-tmp/Wendell/Data/Esm2_emb/Last_wet/Sp07_D.npy",
        "/root/autodl-tmp/Wendell/Data/Esm2_emb/Sp_split_com/Sp07_A-Sp07_B-SpWT_C.npy",
        decimal=6  # 设置比较精度为小数点后6位
    )
