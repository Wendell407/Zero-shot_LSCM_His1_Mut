import torch
from ESM_embedding import EsmEmbedding
import pandas as pd
import numpy as np
import os


csv_path = "/root/autodl-tmp/Wendell/Data/Files/CSV/Last_others_sp.csv"
em_path = "/root/autodl-tmp/Wendell/Data/Esm2_emb/Last_others_sp/"
model_name = "facebook/esm2_t33_650M_UR50D"
# model_name = "/root/autodl-tmp/Wendell/GitNprojects/esm/weights/esm2_t33_650M_UR50D.pt"


if not os.path.exists(em_path):
    os.mkdir(em_path)


max_length = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例
Esm_model = EsmEmbedding(model_name, max_length, device)
Esm_model = Esm_model.to(device)

# 定义函数来获取新序列的 embedding
def get_embeddings(model, sequences, device, max_length):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for idx, sequence in enumerate(sequences):
            # 调用 EsmEmbedding 类的 forward 方法
            emb = model([sequence], device)
            emb = emb.squeeze(0)
            embeddings.append(emb.cpu().numpy())
            print(f"已生成第 {idx + 1} 个序列的 embedding，序列长度: {len(sequence)}")
    return embeddings

# 从 CSV 文件读取数据并进行 embedding 保存
def process_csv_and_save_embeddings(model, csv_path, device, max_length, em_path):
    df = pd.read_csv(csv_path)
    variant_ids = df['unique_ID'].tolist()
    sequences = df['assemble_seq_Dali'].tolist()
    embeddings = get_embeddings(model, sequences, device, max_length)
    for variant_id, embedding in zip(variant_ids, embeddings):
        np.save(f"{em_path}{variant_id}.npy", embedding)
        print(f"{em_path}{variant_id}.npy Embedding 保存完成！")

# 替换为实际的 CSV 文件路径
process_csv_and_save_embeddings(Esm_model, csv_path, device, max_length, em_path)
