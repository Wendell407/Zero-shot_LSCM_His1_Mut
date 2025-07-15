"""
Embed protein sequences with ESM-2-15B on **3×40 GB A100** GPUs
===============================================================
• 自动把模型权重切分到 3 块 GPU（`device_map="auto"` + `max_memory`）
• 长序列按 `chunk_size` 切片，避免激活 OOM
• 从 CSV 读取 `编号, seq` 两列，输出 *.npy 文件到指定目录

Usage
-----
CUDA_VISIBLE_DEVICES=0,1,2 \
python Generate_esm_emb_distributed.py \
       --csv /path/to/nothing.csv \
       --out /path/to/out_dir \
       --model /root/autodl-tmp/lzs/models/esm2_t48_3B_UR50D 

依赖: transformers>=4.40, accelerate>=0.25, torch>=2.1
"""

import argparse, os, math, json, torch, pandas as pd, numpy as np
from pathlib import Path
from transformers import EsmModel, EsmTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()  # 静默 HF log

# -------------------- util --------------------
@torch.inference_mode()
def embed_sequence(seq: str, tokenizer, model, chunk_size: int = 1024):
    """Return mean-pooled embedding (numpy) for a single AA sequence."""
    seq = seq.replace(" ", "").replace("\n", "").upper()
    token_ids = tokenizer(seq, add_special_tokens=False)["input_ids"]
    # chunk
    chunks = [token_ids[i : i + chunk_size] for i in range(0, len(token_ids), chunk_size)]
    reps = []
    for ids in chunks:
        ids = torch.tensor([[tokenizer.cls_token_id] + ids + [tokenizer.eos_token_id]], dtype=torch.long)
        # send to **first device of model**; accelerate 会自动分发
        ids = ids.to(next(model.parameters()).device)
        rep = model(ids).last_hidden_state.mean(dim=1)  # (1, H)
        reps.append(rep.cpu())
    emb = torch.mean(torch.stack(reps, dim=0), dim=0).squeeze(0)  # (H,)
    return emb.numpy()

# -------------------- main --------------------

def main(args):
    os.makedirs(args.out, exist_ok=True)
    print("[INFO] loading tokenizer & model …")
    tokenizer = EsmTokenizer.from_pretrained(args.model, do_lower_case=False)

    # split weights across 3×40 GB GPU (≈38GiB each to leave margin)
    max_mem = {f"cuda:{i}": "38GiB" for i in range(torch.cuda.device_count())}
    model = EsmModel.from_pretrained(
        args.model,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True,
    ).eval()

    df = pd.read_csv(args.csv)
    print(f"[INFO] {len(df)} sequences to embed …")

    for idx, row in df.iterrows():
        vid, seq = str(row["编号"]), str(row["seq"])
        outfile = Path(args.out) / f"{vid}.npy"
        if outfile.exists():
            continue
        emb = embed_sequence(seq, tokenizer, model, chunk_size=args.chunk)
        np.save(outfile, emb)
        if (idx + 1) % 10 == 0 or idx + 1 == len(df):
            print(f"  [{idx+1}/{len(df)}] saved → {outfile.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with columns 编号, seq")
    parser.add_argument("--out", required=True, help="directory to save *.npy")
    parser.add_argument("--model", required=True, help="ESM-2 checkpoint path or HF id")
    parser.add_argument("--chunk", type=int, default=1024, help="tokens per chunk (<=1024 建议)")
    args = parser.parse_args()
    main(args)
