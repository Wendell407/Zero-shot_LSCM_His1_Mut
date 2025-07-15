import pandas as pd
from pathlib import Path
import glob

base_dir = Path("/root/autodl-tmp/Wendell/Data")
pattern = "results_v5_0703_*"

# 1. 处理和转移csv
for csv_path in glob.glob(str(base_dir / pattern / "fold_*_ranking*.csv")):
    csv_path = Path(csv_path)
    labeled_dir = csv_path.parent / "labeled_csv"
    labeled_dir.mkdir(exist_ok=True)
    out_path = labeled_dir / (csv_path.stem + "_label_sorted.csv")
    if out_path.exists():
        print(f"Already exists: {out_path}, skip.")
        continue
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="label", ascending=False).reset_index(drop=True)
    df['label_rank'] = df.index + 1
    df['pos'] = df['label'].apply(lambda x: 1 if x > 0.1 else '')
    df['eff_identi'] = [
        1 if row['pos'] == 1 and df[(df['pos'] != 1) & (df['avg_rank'] <= row['avg_rank'])].empty else ''
        for _, row in df.iterrows()
    ]
    df.to_csv(out_path, index=False)
    print(f"Processed: {out_path}")

# 2. 合并所有 labeled_csv 下的 _label_sorted.csv 文件
for result_dir in base_dir.glob(pattern):
    labeled_dir = result_dir / "labeled_csv"
    if not labeled_dir.exists():
        continue
    csv_files = sorted(labeled_dir.glob("*_label_sorted.csv"))
    if not csv_files:
        continue
    merged = []
    for f in csv_files:
        df = pd.read_csv(f)
        merged.append(df)
        # 插入空行
        merged.append(pd.DataFrame({col: [""] for col in df.columns}))
    # 去掉最后多余的空行
    merged_df = pd.concat(merged, ignore_index=True).iloc[:-1]
    merged_path = labeled_dir / f"{result_dir.name}_merged.csv"
    merged_df.to_csv(merged_path, index=False)
    print(f"Merged CSV saved to: {merged_path}")