# AlphaFold2的colab批量预测与嵌入(Embedding)提取(Pair_Representation, Structure)
import os                      # 操作系统接口（用于文件/目录操作）
import re                      # 正则表达式（用于字符串清洗）
import hashlib                 # 哈希库（生成唯一标识）
import numpy as np             # 数值计算库（处理多维数组数据）
from pathlib import Path       # 路径操作库（提供面向对象的路径处理方式）# Path: 路径操作对象（比字符串路径更安全的操作方式）

# ColabFold主程序安装
if not os.path.isfile("COLABFOLD_READY"):  # 检查是否已安装ColabFold（通过标志文件判断）
    print("🚀 正在安装ColabFold...")
    os.system("pip install -q --no-warn-conflicts 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'")  # 忽略依赖冲突静默安装ColabFold（不包含JAX）
    if os.environ.get('TPU_NAME', False):           # 如果检测到TPU环境
        os.system("pip uninstall -y jax jaxlib")    # 卸载默认的JAX和jaxlib
        os.system("pip install --no-warn-conflicts --upgrade dm-haiku==0.0.10 'jax[cuda12_pip]'==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")  # 安装兼容TPU的JAX和Haiku
    os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabfold colabfold")  # 创建colabfold软链接
    os.system("ln -s /usr/local/lib/python3.*/dist-packages/alphafold alphafold")  # 创建alphafold软链接
    os.system("touch COLABFOLD_READY")  # 创建标志文件，避免重复安装
from colabfold.download import download_alphafold_params        # 模型参数下载器
from colabfold.batch import get_queries, run, set_model_type    # 核心预测流程组件
from colabfold.utils import setup_logging                       # 日志配置工具

def add_hash(x, y):
    """生成唯一哈希标识（前5位SHA1），用于创建唯一任务目录名
    Python知识点：
        - hashlib.sha1(): 创建SHA1哈希对象
        - .encode(): 将字符串转换为字节（哈希函数需要字节输入）
        - .hexdigest()[:5]: 获取哈希值的前5个字符
    """
    return x + "_" + hashlib.sha1(y.encode()).hexdigest()[:5]   # 返回格式：<清洗后的header>_<序列哈希前5位>

def parse_fasta(filename):
    """解析FASTA文件并返回{header: sequence}字典
    参数：
        filename: 输入FASTA文件路径
    返回：
        字典结构：{序列头标识: 蛋白质序列}
    处理逻辑：
        1. 以'>'开头的行作为新序列头
        2. 连续的非头行合并为蛋白质序列
        3. 自动去除头部的特殊字符和空格
    """
    """解析FASTA文件...（保持原有文档字符串）
    Python文件操作：
        - with open(): 上下文管理器自动处理文件打开/关闭
        - line.strip(): 移除行首尾空白字符
        - .startswith('>'): 检测序列头标识
        - .split()[0]: 取第一个空格前的元素
    """
    sequences = {}  # 初始化空字典用于存储header:sequence对
    with open(filename, 'r') as f:  # 打开FASTA文件，自动关闭
        current_header = None       # 当前序列头初始化为None
        current_sequence = []       # 当前序列内容初始化为空列表
        for line in f:              # 逐行读取文件
            line = line.strip()     # 去除行首尾空白字符
            if line.startswith('>'):  # 检查是否为序列头
                if current_header is not None:  # 如果已有header，保存上一条序列
                    sequences[current_header] = ''.join(current_sequence)  # 合并序列并存入字典
                current_header = line[1:].split()[0]  # 提取header（去掉'>'，仅取第一个单词）
                current_sequence = []  # 重置序列内容
            else:
                current_sequence.append(line)  # 非header行追加到序列内容
        if current_header is not None:  # 文件结束后保存最后一条序列
            sequences[current_header] = ''.join(current_sequence)
    return sequences  # 返回header:sequence字典

# 配置参数（核心参数说明）
fasta_file = "/content/Others_TED.fasta"    # 输入文件路径，需包含标准FASTA格式序列 # 字符串类型
base_model_type = "auto"                    # 模型自动选择策略（根据序列复杂度自动选择单体/复合体模型） # 字符串枚举值（可选"auto", "monomer", "multimer"）
num_models = 1                              # 每个序列运行的模型数量（增加可提升准确性但延长计算时间） # 整数类型，必须大于0
num_recycles = 3                            # 循环迭代次数（3-6为推荐值，迭代次数越多精度可能越高） # 整数类型，典型范围3-6
msa_mode = "mmseqs2_uniref_env"             # MSA生成方式（平衡速度与精度的推荐模式） # 字符串枚举值（控制MSA生成算法）
extract_embeddings = True                   # 启用嵌入提取功能（设为False可加快预测速度） # 布尔类型（True/False）
embedding_types = ["msa", "pair", "structure"]  # 列表类型，包含预定义字符串（"msa", "pair", "structure"）
                                                # 需要提取的嵌入类型说明：
                                                # msa: 多序列比对嵌入
                                                # pair: 残基对表示
                                                # structure: 结构模块嵌入

# 解析Fasta文件
sequences = parse_fasta(fasta_file)     # 返回字典对象，键值对为header:sequence# 字典结构：{序列头标识: 蛋白质序列}
if not sequences:                       # 空字典判断 （无序列时）
    raise ValueError(f"No sequences found in {fasta_file}")  # 抛出异常并终止程序

# 主循环流程（新增循环结构解释）
'''
sequences.items(): 遍历字典的键值对（header, sequence）
enumerate可加：for i, (header, sequence) in enumerate(sequences.items()):
''' 
# 处理每个序列的主流程
for header, sequence in sequences.items():
    print(f"\n🚀 正在处理序列: {header}")  # f-string格式化输出
    
    '''
    目录命名规则：
    re.sub(r'\W+', '', header): 移除非字母数字字符（\W表示非单词字符）
    add_hash(): 添加序列哈希防止名称冲突(!!!!!!本段或许可以删除！！！！！)
    '''
    jobname = add_hash(re.sub(r'\W+', '', header), sequence)    # 创建任务目录（目录名格式：清洗后的header_序列哈希）
    os.makedirs(jobname, exist_ok=True)    # os.makedirs：递归创建目录，exist_ok=True：目录已存在时不报错
    print(f"✅ 任务目录已创建: {jobname}")  # 输出创建的任务目录名

    queries_path = os.path.join(jobname, f"{jobname}.csv")    # 保存查询序列（CSV格式，用于后续处理！！！！！！或许也可以删除！！！！！）# CSV格式：每行表示一条记录，逗号分隔值
    with open(queries_path, "w") as f: # 'w'模式表示写入（会覆盖已有文件）
        '''
        写入CSV文件（新增说明）：
        f.write(f"id,sequence\n{jobname},{sequence}")：写入CSV文件（包含id和sequence两列）
        '''
        f.write(f"id,sequence\n{jobname},{sequence}")   # 写入CSV文件（包含id和sequence两列）
    log_filename = os.path.join(jobname, "log.txt")     # 日志文件路径
    setup_logging(Path(log_filename))                   # 初始化日志系统（记录预测过程中的详细信息）
    print(f"🔍 已保存查询序列到: {queries_path}")    
    queries, is_complex = get_queries(queries_path)     # 解析输入序列
    specific_model_type = set_model_type(is_complex, base_model_type)     # 模型类型判断逻辑（自动检测单体/复合体结构） # 确定最终模型类型
    print(f"🔍 解析结果: {len(queries)} 条查询序列, 是否为复合体: {is_complex}, 模型类型: {specific_model_type}")
    '''
    模型类型判断（新增类型说明）：
    is_complex：布尔值，表示是否为复合体（多个链的相互作用）
    set_model_type：返回具体模型类型字符串（如"model_1_multimer"）
    base_model_type：基础模型类型（"auto"表示自动选择）
    '''
    print(f"🚀 正在下载模型参数，模型类型: {specific_model_type}")
    download_alphafold_params(specific_model_type, Path("."))    # 下载模型参数（从Google Cloud Storage获取预训练权重）    # 下载约数百MB的模型参数文件到当前目录    # 需要稳定的网络连接，首次运行耗时较长
    print("✅ 模型参数下载完成!")  # 下载完成提示
    
    def prediction_callback(protein_obj, length, prediction_result, input_features, mode):
        """
        嵌入提取回调函数
        参数：  prediction_result: 包含模型输出的字典
                mode: 当前模型信息（模型名称，模型编号）
        功能：  从预测结果中提取指定类型的嵌入表示，保存为.npy文件
        注意：  - 该函数在每个模型预测完成后自动调用
                - prediction_result包含模型中间表示
        AlphaFold2机制说明：该函数在每个模型预测完成后自动调用，prediction_result包含模型中间表示

        """
        if not extract_embeddings:
            return

        # 字典操作说明：
        model_name, _ = mode
        embeddings = {} # 创建空字典存储嵌入数据

        # 检查嵌入类型是否在指定列表中
        print("-----------------------")
        print("已调用 prediction_callback，结果包含的键:", prediction_result.keys())  # 打印预测结果的键值（调试用）
        print("prediction_result 是否包含 representations:", "representations" in prediction_result)
        if "msa" in embedding_types and "representations" in prediction_result:
            if "msa" in prediction_result["representations"]:
                embeddings["msa"] = prediction_result["representations"]["msa"]
            else:
                print("警告: 未在 prediction_result['representations'] 中找到 'msa' 嵌入")
        if "pair" in embedding_types and "representations" in prediction_result:
            if "pair" in prediction_result["representations"]:
                embeddings["pair"] = prediction_result["representations"]["pair"]
                print("*****************************************************")
                print(f"Pair 嵌入张量形状: {embeddings['pair'].shape}")
            else:
                print("警告: 未在 prediction_result['representations'] 中找到 'pair' 嵌入")
        if "structure" in embedding_types and "structure_module" in prediction_result:
            embeddings["structure"] = prediction_result["structure_module"]
            print("*****************************************************")
        # 保存嵌入文件（每个模型单独保存）
        if embeddings:
            for embed_type, embed_data in embeddings.items():
                np.save(
                    os.path.join(jobname, f"{jobname}_{embed_type}_embeddings_{model_name}.npy"),
                    embed_data  # 嵌入数据维度说明：
                                # msa: [序列长度, MSA深度, 嵌入维度]
                                # pair: [序列长度, 序列长度, 通道数]
                                # structure: [序列长度, 嵌入维度]
                )
            print(f"已保存 {header} 的 {model_name} 嵌入文件")
        # numpy文件操作：
        # np.save()：将numpy数组保存为二进制文件
        # 加载时使用：np.load("filename.npy")
        
    # 核心预测流程（使用AlphaFold2进行结构预测）
    print(f"🚀 正在运行 {header} 的结构预测...")
    results = run(
        queries=queries, # 从CSV解析的查询列表
        result_dir=jobname,
        use_templates=False,                        # 禁用模板（提升速度但可能影响准确性）
        num_relax=0,                                # 结构松弛次数（0表示不进行，可设为1）
        msa_mode=msa_mode,                          # 控制MSA生成方式（影响速度与精度平衡）
        model_type=specific_model_type,             # 使用自动选择的模型类型
        num_models=num_models,                      # 每个序列运行的模型数量（1-5）
        num_recycles=num_recycles,                  # 循环次数影响最终精度
        num_seeds=1,                                # 随机种子数（增加可提升多样性）
        model_order=[1],                            # 使用的模型编号（AlphaFold提供的不同训练轮次的模型）
        is_complex=is_complex,                      # 是否为复合体预测（影响模型选择）
        data_dir=Path("."),                         # 数据目录（模型参数下载目录）
        keep_existing_results=False,                # 是否保留已有结果（True表示跳过已存在的预测）
        rank_by="auto",                             # 结果排序方式（自动选择pLDDT或ipTM）
        pair_mode="unpaired_paired",                # 配对模式（适用于单体/复合体预测）
        prediction_callback=prediction_callback,    # 嵌入提取回调函数
        zip_results=False,                          # 是否压缩结果（节省存储空间）
        save_all=True,                              # 是否保存中间结果（True会显著增加存储需求）
        use_cluster_profile=True,                   # 使用聚类配置文件（提升MSA质量）
        return_representations=True,                # 返回嵌入表示
        return_predictions=True,                    # 返回完整预测结果
    )

print("\n🎉 所有序列处理完成!")