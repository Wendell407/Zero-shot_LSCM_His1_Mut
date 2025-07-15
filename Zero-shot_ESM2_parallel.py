from transformers import EsmForMaskedLM, EsmTokenizer
import torch
import torch.multiprocessing as mp
import csv
import os
import math
from tqdm import tqdm


def compute_batch_log_likelihood(model, tokenizer, sequences, device, max_len):
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len
    ).to(device)
    with torch.inference_mode():  # 节省显存
        logits = model(**inputs).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    input_ids = inputs["input_ids"]
    ll_avg_list, ll_total_list = [], []

    for i in range(len(sequences)):
        # 忽略 pad 部分
        seq_len = (input_ids[i] != tokenizer.pad_token_id).sum().item()
        ll_total = 0.0
        count = 0
        for j in range(1, seq_len - 1):
            token_id = input_ids[i, j]
            ll_total += log_probs[i, j, token_id].item()
            count += 1
        ll_avg = ll_total / count
        ll_avg_list.append(ll_avg)
        ll_total_list.append(ll_total)

    return ll_avg_list, ll_total_list


def read_fasta(fasta_path):
    seqs = {}
    with open(fasta_path, 'r') as f:
        curr_id, curr_seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if curr_id is not None:
                    seqs[curr_id] = ''.join(curr_seq)
                curr_id = line[1:]
                curr_seq = []
            else:
                curr_seq.append(line)
        if curr_id is not None:
            seqs[curr_id] = ''.join(curr_seq)
    return seqs


def worker(rank, world_size, model_dir, task_chunks, batch_size, ll_orig_avg, ll_orig_total, return_list, max_len):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = EsmTokenizer.from_pretrained(model_dir)
    model = EsmForMaskedLM.from_pretrained(model_dir).to(device)
    model.eval()

    task_list = task_chunks[rank]
    pbar = tqdm(total=len(task_list), position=rank, desc=f"[GPU {rank}]", leave=True)

    for i in range(0, len(task_list), batch_size):
        batch = task_list[i:i + batch_size]
        seq_ids = [sid for sid, _ in batch]
        seqs = [s for _, s in batch]

        try:
            ll_avg_list, ll_total_list = compute_batch_log_likelihood(model, tokenizer, seqs, device, max_len)
            for seq_id, ll_mut_avg, ll_mut_total in zip(seq_ids, ll_avg_list, ll_total_list):
                return_list.append({
                    "id": seq_id,
                    "avg": ll_mut_avg - ll_orig_avg,
                    "total": ll_mut_total - ll_orig_total
                })
        except Exception as e:
            print(f"[GPU {rank}] ❌ Error processing batch {seq_ids}: {e}")
            for seq_id in seq_ids:
                return_list.append({
                    "id": seq_id,
                    "avg": None,
                    "total": None
                })
        pbar.update(len(batch))
    pbar.close()


def main_spawn(fasta_path, seq_orig, model_dir, output_csv_path, batch_size=16):
    sequences = read_fasta(fasta_path)

    all_tasks = [(seq_id, seq) for seq_id, seq in sequences.items()]
    world_size = torch.cuda.device_count()
    chunk_size = math.ceil(len(all_tasks) / world_size)
    task_chunks = [all_tasks[i * chunk_size:(i + 1) * chunk_size] for i in range(world_size)]

    # 主进程加载 tokenizer + baseline
    tokenizer = EsmTokenizer.from_pretrained(model_dir)
    model = EsmForMaskedLM.from_pretrained(model_dir).cuda()
    model.eval()

    # 固定 padding 长度
    max_len = max(len(seq) for _, seq in all_tasks + [("orig", seq_orig)])

    ll_orig_avg, ll_orig_total = compute_batch_log_likelihood(model, tokenizer, [seq_orig], torch.device("cuda"), max_len)
    ll_orig_avg = ll_orig_avg[0]
    ll_orig_total = ll_orig_total[0]
    del model
    torch.cuda.empty_cache()

    manager = mp.Manager()
    return_list = manager.list()

    # 启动多卡进程
    mp.spawn(
        worker,
        args=(world_size, model_dir, task_chunks, batch_size, ll_orig_avg, ll_orig_total, return_list, max_len),
        nprocs=world_size,
        join=True
    )

    # 写入排序后的 CSV
    sorted_results = sorted(return_list, key=lambda x: x["avg"] if x["avg"] is not None else 1e9)
    with open(output_csv_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "avg", "total"])
        writer.writeheader()
        writer.writerows(sorted_results)


if __name__ == "__main__":
    target_fasta_path = "/root/autodl-tmp/PairNet/data/SpWet_Dali.fasta"
    sp_wet_sequences = read_fasta(target_fasta_path)
    seq_orig = sp_wet_sequences["SpWT"]

    output_path = "/root/autodl-tmp/PairNet/results_wet/esm2_zero_shot"
    os.makedirs(output_path, exist_ok=True)

    main_spawn(
        fasta_path=target_fasta_path,
        seq_orig=seq_orig,
        model_dir="/root/autodl-tmp/PairNet/model_weights/esm2_t33_650M_UR50D",
        output_csv_path=f"{output_path}/esm2_zero_shot_spwet_dali_parallel_fixed.csv",
        batch_size=16
    )
