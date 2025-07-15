import pandas as pd
from pathlib import Path
import re

base_dir = Path("/root/autodl-tmp/lzs/Func_eval")
seeds = [0, 1, 24, 42, 2025]

# 1. 找到所有以 _0_650m 结尾的目录，提取前缀
seed0_dirs = sorted(base_dir.glob("*_0_650m"))
for seed0_dir in seed0_dirs:
    # 提取前缀
    prefix = re.sub(r"_0_650m$", "", seed0_dir.name)
    # 构造5个seed目录和merged.csv路径
    dfs = []
    for seed in seeds:
        sub_dir = base_dir / f"{prefix}_{seed}_650m"   # <-- 修正这里
        merged_csv = sub_dir / "labeled_csv" / f"{sub_dir.name}_merged.csv"
        if merged_csv.exists():
            df = pd.read_csv(merged_csv)
            df.columns = [f"{col}_seed{seed}" for col in df.columns]
            # 添加空列分隔
            df[f"PSS{seed}"] = "_"
            dfs.append(df)
        else:
            print(f"Warning: {merged_csv} not found, skip.")
   
    if dfs:
        # dfs[-1] = dfs[-1].iloc[:, :-1] # 去掉最后一个空列
        merged_df = pd.concat(dfs, axis=1)
        out_path = base_dir / f"{prefix}_650m_merged.csv"
        merged_df.to_csv(out_path, index=False)
        print(f"Saved horizontally merged CSV to: {out_path}")