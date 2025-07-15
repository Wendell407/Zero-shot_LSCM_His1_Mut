#!/bin/bash
: '
脚本功能：使用 FoldX 进行 PDB 文件的修复和自由能计算，注意：
1. 需要安装 GNU Parallel 来实现并行处理
2. 需要安装 FoldX 软件包并配置环境变量
3. FoldX 需要在文件所在目录操作，因此在处理每个 PDB 文件时会切换目录
'
# 检查是否安装了 parallel
if ! command -v parallel &> /dev/null; then
    echo "❌ GNU Parallel 未安装，请安装后重试。"
    exit 1
fi
# 检查FoldX可执行文件是否存在
FOLDX_EXEC="foldx_20251231" # 配置FoldX可执行文件路径（如果FoldX不是全局可执行，修改这里）
# 输出FoldX版本信息
echo -e "
🔍 FoldX 版本信息：
$($FOLDX_EXEC)
开始执行
"

# 路径及硬件配置
NUM_THREADS=$(nproc) # 设定并行线程数（默认为 4，可根据 CPU 核心数调整）
echo "📂 当前目录 -> $(pwd)"
INPUT_DIR="/root/autodl-tmp/Wendell/ESM2_embedding/Alphafold2/pdb" # 设定需要预测的PDB文件路径（默认输出当前路径中）
echo "📂 输入目录：$INPUT_DIR"
OUTPUT_DIR="${INPUT_DIR}/results" # 设定输出目录
mkdir -p "$OUTPUT_DIR" # 若不存在输出目录，则自动创建
echo -e "📂 输出目录：$OUTPUT_DIR \n⚠️  二次执行时可将本脚本下方mv部分注释取消再次运行以分类至不同文件夹"
echo "⚙️  使用 $NUM_THREADS 个线程进行并行处理"
PDB_FILES=($(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.pdb" ! -name "*_Repair.pdb"))
if [ ${#PDB_FILES[@]} -eq 0 ]; then
    echo "❌没有找到PDB文件，请检查 $INPUT_DIR 输入目录。"
    exit 1
fi # 检查是否有PDB文件
PDB_NAMES=($(for f in "${PDB_FILES[@]}"; do basename "$f"; done))
echo -e "✅ 找到${#PDB_FILES[@]}个PDB如下：\n${PDB_NAMES[@]}" # 输出找到的PDB文件数量和名称

# --------------------------------------------主处理过程起始点----------------------------------------------------------------

# 定义处理单个PDB文件的函数
process_pdb() {
    local pdb_file=$1 # 子shell获取通过Parallel传入的PDB文件（含路径）
    local pdb_name=$(basename "$pdb_file" .pdb) # 获取PDB文件名（不带路径和扩展名）
    local original_dir=$(pwd)
    local pdb_dir=$(dirname "$pdb_file")

    cd "$pdb_dir" || return # 切换到PDB文件所在目录
    echo "🔄 切换到目录：$pdb_dir"
    echo "🔧 处理 $pdb_name.pdb..."
    echo "DEBUG: 接收到的参数 -> $1"
    echo "DEBUG: 当前工作目录 -> $(pwd)"
    # 添加修复文件存在性检查
    if [ -f "${pdb_name}_Repair.pdb" ]; then
        echo "⚠️ 发现已存在的修复文件 ${pdb_name}_Repair.pdb，跳过修复步骤"
        REPAIRED_PDB="${pdb_name}_Repair.pdb"
    else
        $FOLDX_EXEC --command=RepairPDB --pdb="$pdb_name.pdb" 2>&1 | tee "${pdb_name}_repair.log"
        REPAIRED_PDB="${pdb_name}_Repair.pdb"
    fi
    if [ ! -f "$REPAIRED_PDB" ]; then
        echo "❌ $pdb_name RepairPDB 失败，跳过..."
        cd "$original_dir" || return
        return
    fi
    echo -e "✅ Repair处理完成：$REPAIRED_PDB \n 🔧开始处理Stability计算自由能..."
    $FOLDX_EXEC --command=Stability --pdb="$REPAIRED_PDB" 2>&1 | tee "${pdb_name}_Repair_stability.log"
    STABILITY_FILE="${pdb_name}_Repair_0_ST.fxout" # 计算自由能的输出文件名
    if [ -f "$STABILITY_FILE" ]; then
        echo "✅ Stability处理完成: $STABILITY_FILE"
    else
        echo "❌ $pdb_name Stability 计算失败"
    fi
    Repairfxout="${pdb_name}_Repair.fxout"
    repairlog="${pdb_name}_repair.log"
    stabilitylog="${pdb_name}_Repair_stability.log"
    # 检查修复文件和自由能文件是否存在
    if [ ! -f "$REPAIRED_PDB" ]; then
        echo "❌ $REPAIRED_PDB 文件不存在，跳过移动"
        cd "$original_dir" || return
        return
    fi
    echo "📄 处理完成的文件: $REPAIRED_PDB, $Repairfxout, $STABILITY_FILE"
    echo "📄 日志文件: $repairlog, $stabilitylog"
    # # 移动修复文件和自由能文件以及log文件到输出目录（再次运行时取消注释即可）
    # echo "📁 移动文件到 $OUTPUT_DIR"
    # mkdir -p "$OUTPUT_DIR/logs" # 创建日志目录
    # mkdir -p "$OUTPUT_DIR/repaired" # 创建修复文件目录
    # mv "$STABILITY_FILE" "$OUTPUT_DIR/"
    # mv "$REPAIRED_PDB" "$OUTPUT_DIR/repaired"
    # mv "$Repairfxout" "$OUTPUT_DIR/repaired"
    # mv "$repairlog" "$OUTPUT_DIR/logs"
    # mv "$stabilitylog" "$OUTPUT_DIR/logs"
    # echo "✅ 移动完成：$REPAIRED_PDB, $Repairfxout, $repairlog, $stabilitylog"
    cd "$original_dir" || return # 最终统一返回原目录
    echo "🔚 该PDB所有处理完成，返回主目录。"
    echo "🔄 准备处理下一个 PDB 文件..."
}
export -f process_pdb
export FOLDX_EXEC OUTPUT_DIR

# 使用 GNU Parallel 进行并行处理所有PDB文件
echo "🚀 开始并行处理 PDB 文件..."
parallel -j "$NUM_THREADS" process_pdb ::: "${PDB_FILES[@]}"
# 处理完成后输出提示
echo "🎉 所有PDB文件处理完成！结果存入 $OUTPUT_DIR"