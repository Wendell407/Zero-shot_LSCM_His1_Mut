import os
import random
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data as GraphData, Batch

# ----------------- Utilities -----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------- Dataset -----------------
class CustomPairwiseDataset(Dataset):
    def __init__(self, pdb_ids, labels, data_dir: Path, structure_dir: Path, metric_name, diff_threshold=0.0):
        self.pdb_ids = pdb_ids
        self.labels = labels
        self.embeddings = []
        self.graphs = []
        self.values = []
        self.embeddings_sup = []

        metric_df = pd.read_json(data_dir / "data_point.json")

        for pdb_id in pdb_ids:
            val = np.mean([metric_df.loc[metric_df['Variants'] == pdb_id, m].values[0] for m in metric_name])
            self.values.append(val)

            emb = np.load(data_dir / f"{pdb_id}.npy")
            emb = torch.tensor(emb, dtype=torch.float32)
            emb = emb[1:-1] if emb.ndim > 1 else emb
            emb = F.normalize(emb, p=2, dim=-1)

            # structure
            dist_mat = np.load(structure_dir / f"{pdb_id}.npy")  # (L,L)
            edge_index = np.array(dist_mat.nonzero())
            edge_weight = dist_mat[edge_index[0], edge_index[1]]

            graph = GraphData(
                x=emb,
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_weight, dtype=torch.float32)
            )
            self.graphs.append(graph)

            order = metric_df.loc[metric_df['Variants'] == pdb_id, 'active_numbers'].values[0]
            emb_sup = torch.stack([emb[order[0]], emb[order[1]], emb[order[2]]], dim=0).mean(0)
            emb_sup = F.normalize(emb_sup, p=2, dim=-1)
            self.embeddings_sup.append(emb_sup)

        self.pairs = self._generate_pairs(self.values, diff_threshold)

    def _generate_pairs(self, values, diff_threshold):
        pairs = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if abs(values[i] - values[j]) >= diff_threshold:
                    pairs.append((i, j))
                    pairs.append((j, i))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        g1 = self.graphs[i]
        g2 = self.graphs[j]
        emb1_sup = self.embeddings_sup[i]
        emb2_sup = self.embeddings_sup[j]
        label = 1 if self.values[i] > self.values[j] else -1
        return g1, g2, emb1_sup, emb2_sup, torch.tensor(label, dtype=torch.float32)

# ----------------- collate_fn -----------------
def pair_collate_fn(batch):
    g1_list, g2_list, sup1_list, sup2_list, y_list = zip(*batch)
    batch_g1 = Batch.from_data_list(g1_list)
    batch_g2 = Batch.from_data_list(g2_list)
    sup1 = torch.stack(sup1_list, dim=0)
    sup2 = torch.stack(sup2_list, dim=0)
    y = torch.stack(y_list, dim=0)
    return batch_g1, batch_g2, sup1, sup2, y

# ----------------- GCN Encoder -----------------
class GCNEncoder(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.input_linear(x)
        h = F.relu(self.gcn1(x, edge_index, edge_weight))
        h = F.relu(self.gcn2(h, edge_index, edge_weight))
        pooled = global_mean_pool(h, data.batch)
        return pooled

# ----------------- Pair Model -----------------
class PairNet(nn.Module):
    def __init__(self, gcn_hidden=256, emb_sup_dim=1280, dropout=0.2):
        super().__init__()
        self.gcn = GCNEncoder(input_dim=1280, hidden_dim=gcn_hidden)
        self.sup_linear = nn.Linear(emb_sup_dim, gcn_hidden)
        self.net = nn.Sequential(
            nn.Linear(2 * (gcn_hidden + gcn_hidden), gcn_hidden),
            nn.BatchNorm1d(gcn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden, gcn_hidden // 2),
            nn.BatchNorm1d(gcn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden // 2, 1)
        )

    def forward(self, g1, g2, sup1, sup2):
        z1 = self.gcn(g1)
        z2 = self.gcn(g2)
        sup1 = self.sup_linear(sup1)
        sup2 = self.sup_linear(sup2)
        feat1 = torch.cat([z1, sup1], dim=-1)
        feat2 = torch.cat([z2, sup2], dim=-1)
        x = torch.cat([feat1, feat2], dim=-1)
        return self.net(x).squeeze(-1)

# ----------------- Training -----------------
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for g1, g2, sup1, sup2, y in loader:
        g1, g2 = g1.to(device), g2.to(device)
        sup1, sup2, y = sup1.to(device), sup2.to(device), y.to(device)
        scores = model(g1, g2, sup1, sup2)
        loss = loss_fn(scores, torch.zeros_like(scores), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)

# ----------------- Evaluation -----------------
def eval_metrics(model, loader, loss_fn, device):
    model.eval()
    true_labels = []
    pred_diff = []
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for g1, g2, sup1, sup2, y in loader:
            g1, g2 = g1.to(device), g2.to(device)
            sup1, sup2, y = sup1.to(device), sup2.to(device), y.to(device)
            scores = model(g1, g2, sup1, sup2)
            loss = loss_fn(scores, torch.zeros_like(scores), y)
            total_loss += loss.item() * y.size(0)
            preds = torch.sign(scores).cpu().numpy()
            gt = y.cpu().numpy()
            correct += (preds == gt).sum()
            true_labels.extend(gt.tolist())
            pred_diff.extend(scores.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    tau = kendalltau(true_labels, pred_diff).correlation
    rho = spearmanr(true_labels, pred_diff).correlation
    return avg_loss, acc, tau, rho

# ----------------- Main -----------------
if __name__ == "__main__":
    seeds = [0, 1, 24, 42, 2025]
    config = '650m'
    for seed in seeds:
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DATA_DIR = Path("/root/autodl-tmp/lzs/Func_eval/cwt_data_domain/embeddings_0609")
        STRUCTURE_DIR = Path("/root/autodl-tmp/Wendell/Data/GCN_AF3/sp")
        RESULTS_DIR = Path(f"/root/autodl-tmp/Wendell/Data/results_v5_0703_dali_pos_neg_kfold_point_struc_{seed}_{config}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        metric = ["FITNESS"]
        POS_DIR = Path("/root/autodl-tmp/lzs/Func_eval/cwt_data_domain/sp_pdb_kfold/pos")
        NEG_DIR = Path("/root/autodl-tmp/lzs/Func_eval/cwt_data_domain/sp_pdb_kfold/neg_all")

        pos_ids = [f.stem for f in POS_DIR.glob("*.npy")]
        neg_ids = [f.stem for f in NEG_DIR.glob("*.npy")]

        CONFIG = {
            "batch_size": 5,
            "lr": 3e-5,
            "weight_decay": 1e-4,
            "margin": 0.5,
            "diff_threshold": 0.0,
            "hidden_dim": 256,
            "epochs": 100,
            "patience": 10,
            "folds": 5
        }

        kf = KFold(n_splits=CONFIG["folds"], shuffle=True, random_state=seed)
        val_accs = []
        fold_splits = {}

        for fold, (pos_train_idx, pos_val_idx) in enumerate(kf.split(pos_ids)):
            print(f"Fold {fold+1}/{CONFIG['folds']}")
            pos_train = [pos_ids[i] for i in pos_train_idx]
            pos_val = [pos_ids[i] for i in pos_val_idx]

            neg_kf = KFold(n_splits=CONFIG["folds"], shuffle=True, random_state=fold)
            neg_train_idx, neg_val_idx = list(neg_kf.split(neg_ids))[fold]
            neg_train = [neg_ids[i] for i in neg_train_idx[:2*len(pos_train)]]
            neg_val = [neg_ids[i] for i in neg_val_idx[:2*len(pos_val)]]

            train_ids = pos_train + neg_train
            train_labels = [1]*len(pos_train) + [0]*len(neg_train)
            val_ids = pos_val + neg_val
            val_labels = [1]*len(pos_val) + [0]*len(neg_val)

            fold_splits[f"fold_{fold+1}"] = {"train_ids": train_ids, "val_ids": val_ids}

            train_ds = CustomPairwiseDataset(train_ids, train_labels, DATA_DIR, STRUCTURE_DIR, metric)
            val_ds = CustomPairwiseDataset(val_ids, val_labels, DATA_DIR, STRUCTURE_DIR, metric)

            train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=pair_collate_fn)
            val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=pair_collate_fn)

            model = PairNet().to(device)
            loss_fn = nn.MarginRankingLoss(margin=CONFIG["margin"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            best_val_loss = float('inf')
            no_improve = 0
            best_val_acc = 0.0

            for epoch in range(1, CONFIG["epochs"] + 1):
                train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
                val_loss, val_acc, tau, rho = eval_metrics(model, val_loader, loss_fn, device)
                scheduler.step(val_loss)
                print(f"Fold {fold+1} | Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={val_acc:.4f}, tau={tau:.4f}, rho={rho:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    no_improve = 0
                    torch.save(model.state_dict(), RESULTS_DIR / f"best_model_fold{fold+1}.pth")
                else:
                    no_improve += 1
                    if no_improve >= CONFIG["patience"]:
                        print("Early stopping.")
                        break

            val_accs.append(best_val_acc)

                        # === 生成 win-rate 排序 CSV ===
            print(f"Generating win-rate ranking for fold {fold+1}...")

            # reload best model
            model.load_state_dict(torch.load(RESULTS_DIR / f"best_model_fold{fold+1}.pth", map_location=device))
            model.eval()

            val_graphs = val_ds.graphs
            val_emb_sup = val_ds.embeddings_sup
            val_pdb_ids = val_ds.pdb_ids
            val_values = val_ds.values
            n_val = len(val_graphs)

            # calculate embedding
            val_gcn_embeddings = []
            with torch.no_grad():
                for g in val_graphs:
                    g = g.to(device)
                    z = model.gcn(g)
                    val_gcn_embeddings.append(z.cpu())

            wins1 = [0 for _ in range(n_val)]
            wins2 = [0 for _ in range(n_val)]

            with torch.no_grad():
                for i in range(n_val):
                    for j in range(i+1, n_val):
                        e1 = val_gcn_embeddings[i].to(device)
                        e2 = val_gcn_embeddings[j].to(device)
                        s1 = model.sup_linear(val_emb_sup[i].unsqueeze(0).to(device))
                        s2 = model.sup_linear(val_emb_sup[j].unsqueeze(0).to(device))
                        feat1 = torch.cat([e1.squeeze(0), s1.squeeze(0)], dim=-1)
                        feat2 = torch.cat([e2.squeeze(0), s2.squeeze(0)], dim=-1)

                        score = model.net(torch.cat([feat1, feat2], dim=-1).unsqueeze(0)).item()
                        if score > 0:
                            wins1[i] += 1
                        else:
                            wins1[j] += 1
                for i in range(n_val):
                    for j in range(i+1, n_val):
                        e1 = val_gcn_embeddings[j].to(device)
                        e2 = val_gcn_embeddings[i].to(device)
                        s1 = model.sup_linear(val_emb_sup[j].unsqueeze(0).to(device))
                        s2 = model.sup_linear(val_emb_sup[i].unsqueeze(0).to(device))
                        feat1 = torch.cat([e1.squeeze(0), s1.squeeze(0)], dim=-1)
                        feat2 = torch.cat([e2.squeeze(0), s2.squeeze(0)], dim=-1)
                        score = model.net(torch.cat([feat1, feat2], dim=-1).unsqueeze(0)).item()

                        if score > 0:
                            wins2[j] += 1
                        else:
                            wins2[i] += 1

            win_rates1 = [w/(n_val-1) for w in wins1]
            win_rates2 = [w/(n_val-1) for w in wins2]

            sorted_idx1 = sorted(range(n_val), key=lambda k: win_rates1[k], reverse=True)
            sorted_idx2 = sorted(range(n_val), key=lambda k: win_rates2[k], reverse=True)

            # 计算综合排序（index 相加后取平均）
            avg_ranks = [(sorted_idx1.index(i) + 1 + sorted_idx2.index(i) + 1) / 2 for i in range(n_val)]
            final_sorted_indices = sorted(range(n_val), key=lambda k: avg_ranks[k])

            # 构建 DataFrame 保存
            df = pd.DataFrame({
                "pdb_id": [val_pdb_ids[i] for i in final_sorted_indices],
                "label": [val_values[i] for i in final_sorted_indices],
                "avg_rank": [avg_ranks[i] for i in final_sorted_indices],
                "rank_pos": [sorted_idx1.index(i) + 1 for i in final_sorted_indices],
                "rank_neg": [sorted_idx2.index(i) + 1 for i in final_sorted_indices],
            })
            
            df.to_csv(RESULTS_DIR / f"fold_{fold+1}_ranking.csv", index=False)
            print(f"Saved ranking CSV to fold_{fold+1}_ranking.csv")


        print("\n===== Cross-Validation Summary =====")
        for i, acc in enumerate(val_accs):
            print(f"Fold {i+1} Accuracy: {acc:.4f}")
        print(f"Average Accuracy: {np.mean(val_accs):.4f}")

        with open(RESULTS_DIR / "kfold_split_ids.json", "w") as f:
            json.dump(fold_splits, f, indent=4)

