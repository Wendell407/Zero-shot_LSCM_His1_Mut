#   使用fitness 预测label 和fitness
#   边分类边回归
#   70cath数据训练 1280
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from ESM_embedding import EsmEmbedding
import pandas as pd
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# 单个 CSV 文件路径
csv_path = '/root/autodl-tmp/lc/code_for_contrast/train-data-3nt.csv'
# train_csv_path = '/root/autodl-tmp/lc/data/indel.result_train.csv'
# test_csv_path = '/root/autodl-tmp/lc/data/indel.result_test.csv'

resultname = 'usdata-more527-3nt-sp'
target_row = 'fit_score'
# loss_file_path = '/root/autodl-tmp/lc/results/train_loss_cls_fitness_under_dropout.txt'
# prediction_file_path = '/root/autodl-tmp/lc/results/prediction_cls_fitness_under_dropout.csv'
# model_save_path = "/root/autodl-tmp/lc/code_for_contrast/model/trained_model_fitness_under_dropout.pth"
# conv_save_path = "/root/autodl-tmp/lc/code_for_contrast/model/conv_layer_under_dropout_t6.pth"
# model_name = "facebook/esm2_t6_8M_UR50D"




loss_file_path = f'/root/autodl-tmp/lc/results/train_loss_cls_fitness_under_dropout_t33_for_sasp_{resultname}-.txt'
prediction_file_path = f'/root/autodl-tmp/lc/results/prediction_cls_fitness_under_dropout_t33_for_sasp_{resultname}-.csv'
model_save_path = f"/root/autodl-tmp/lc/code_for_contrast/model/trained_model_under_dropout_t33_for_sasp_{resultname}-.pth"
classifier_save_path = f"/root/autodl-tmp/lc/code_for_contrast/model/classifier_under_dropout_t33_for_sasp_{resultname}-.pth"
conv_save_path = f"/root/autodl-tmp/lc/code_for_contrast/model/conv_layer_under_dropout_t33_for_sasp_{resultname}-.pth"
em_path = f"/root/autodl-tmp/lc/embedding_1280-lossmi1_{resultname}-/"

model_name = "facebook/esm2_t33_650M_UR50D"
print(em_path)


input_size = 1280
# input_size = 320

local_model_path = "model_get_esm/"
num_epochs = 100
batchsize = 1
neg_pos_radio = 3

dropout_rate = 0.3

learning_rate = 5e-7
test_size = 0.1
weight_decay = 1e-6

dis_pos_neg = 0.001

# 自定义数据集类
class SequenceDataset(Dataset):
    def __init__(self, sequences, scores,dis_pos_neg):
        self.sequences = sequences
        self.scores = scores
        self.dis_pos_neg = dis_pos_neg

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        score = self.scores[idx]
        # 根据 fit_score 区分正负样本
        label = 1 if score >= dis_pos_neg else 0
        return sequence, score, label


class ClassifyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ClassifyLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, labels, scores):
        # 对 features 进行维度调整
        features = features.mean(dim=1)  # 对 sequence_length 维度求平均，得到 (batch_size, hidden_size)

        # 计算相似度矩阵
        similarities = torch.matmul(features, features.T) / self.temperature

        # 创建标签，正样本对的标签为 1，负样本对的标签为 0
        batch_size = features.size(0)
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        negative_mask = ~positive_mask

        # 生成对比学习的标签
        contrastive_labels = torch.arange(batch_size).to(features.device)

        # 对相似度矩阵进行掩码操作，只保留正样本对和负样本对的相似度
        masked_similarities = similarities.masked_fill(negative_mask, -float('inf'))

        # 计算对比损失
        contrastive_loss = self.criterion(masked_similarities, contrastive_labels)

        # 根据 scores 计算权重
        # weights = scores / scores.sum()
        weights = scores / (scores.sum() + 1e-10)  # 添加极小值 1e-8
        # # 加权对比损失
        weighted_contrastive_loss = (contrastive_loss * weights).sum()
        # weighted_contrastive_loss = (contrastive_loss).sum()

        return weighted_contrastive_loss


# 回归损失函数
class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()

    def forward(self, features, scores):
        # 修改此处，对 features 进行维度调整
        features = features.mean(dim=1)  # 对 sequence_length 维度求平均，得到 (batch_size, hidden_size)
        features = features.mean(dim=1)
        predictions = features
        # 计算绝对值误差损失
        loss = torch.mean(torch.abs((predictions - scores)**2))
        return loss

# 使用最大池化替代平均操作
import torch.nn.functional as F

class RegressionLossMaxPool(nn.Module):
    def __init__(self):
        super(RegressionLossMaxPool, self).__init__()

    def forward(self, features, scores):
        # 对 sequence_length 维度进行最大池化
        features = F.max_pool1d(features.permute(0, 2, 1), kernel_size=features.size(1)).squeeze(-1)
        # 添加全连接层调整维度
        fc_layer = nn.Linear(features.size(1), scores.size(0)).to(features.device)
        predictions = fc_layer(features)
        # 计算绝对值误差损失
        loss = torch.mean(torch.abs((predictions - scores)**2))
        return loss

class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        attn_weights = self.attention(features)
        weighted_features = features * attn_weights
        return weighted_features.sum(dim=1)

class RegressionLossAttention(nn.Module):
    def __init__(self, input_size):
        super(RegressionLossAttention, self).__init__()
        self.attention_layer = AttentionLayer(input_size)

    def forward(self, features, scores):
        # 使用注意力机制处理特征
        # 对 sequence_length 维度进行最大池化
        # features = F.max_pool1d(features.permute(0, 2, 1), kernel_size=features.size(1)).squeeze(-1)
        features = F.avg_pool1d(features.permute(0, 2, 1), kernel_size=features.size(1)).squeeze(-1)
        attended_features = self.attention_layer(features)
        predictions = attended_features
        # 计算绝对值误差损失
        loss = torch.mean(torch.abs((predictions - scores)**2))
        return loss
    
# 从 CSV 文件加载数据
def load_data_undersampling(csv_path):
    df = pd.read_csv(csv_path)
    # 筛选 fit_score 大于 1 的行
    positive_df = df[df[target_row] >= 1]
    lensofpos = len(positive_df)
    print("Len of pos samples:  ", lensofpos)
    # 筛选 fit_score 小于 1 的行
    negative_df = df[df[target_row] < 1]
    print("Len of neg samples:  ", len(negative_df))

    # 合并正样本和负样本用于欠采样
    combined_df = pd.concat([positive_df, negative_df])
    X = combined_df[['All_Sequence', target_row]]
    y = (combined_df[target_row] > 1).astype(int)

    # 计算需要的负样本数量
    target_neg_count = min(len(negative_df), lensofpos * neg_pos_radio)
    target_counts = {0: target_neg_count, 1: lensofpos}

    # 创建欠采样器
    sampler = RandomUnderSampler(sampling_strategy=target_counts)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # 从欠采样结果中获取序列和分数
    sequences = X_resampled['All_Sequence'].tolist()
    scores = X_resampled[target_row].tolist()
    return sequences, scores


def load_data_oversampling(csv_path, neg_pos_radio):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 筛选 fit_score 大于 1 的行作为正样本
    positive_df = df[df['fit_score'] > 1]
    lensofpos = len(positive_df)
    print("Len of pos samples:  ", lensofpos)

    # 筛选 fit_score 小于 1 的行作为负样本
    negative_df = df[df['fit_score'] < 1]
    lenofneg = len(negative_df)
    print("Len of neg samples:  ", lenofneg)

    # 计算需要过采样的正样本数量，确保正样本数量为负样本数量的三分之一
    target_positive_count = lenofneg // neg_pos_radio

    # 合并正样本和负样本
    combined_df = pd.concat([positive_df, negative_df])
    X = combined_df[['variant_sequence', 'fit_score']]
    y = (combined_df['fit_score'] > 1).astype(int)

    # 确定采样策略
    target_counts = {0: lenofneg, 1: target_positive_count}

    # 创建过采样器
    ros = RandomOverSampler(sampling_strategy=target_counts)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # 从过采样结果中获取序列和分数
    sequences = X_resampled['variant_sequence'].tolist()
    scores = X_resampled['fit_score'].tolist()
    return sequences, scores

def load_data(csv_path):
    df = pd.read_csv(csv_path,sep=',')
    # 直接获取序列和分数
    sequences = df['All_Sequence'].tolist()
    scores = df[target_row].tolist()
    return sequences, scores

# 加载数据
sequences, scores = load_data(csv_path)
# sequences, scores = load_data_undersampling(csv_path)
print(f"Training Data:  {len(sequences)}")
# 随机划分训练集和测试集
train_sequences, test_sequences, train_scores, test_scores = train_test_split(
    sequences, scores, test_size=test_size, random_state=418)
# 不随机划分训练集和测试集
# train_sequences,train_scores = load_data(train_csv_path)
# print(f"Training Data:  {len(train_sequences)}")

# test_sequences,test_scores = load_data(test_csv_path)
# print(f"Testing Data:  {len(test_sequences)}")

# 创建数据集和数据加载器
train_dataset = SequenceDataset(train_sequences, train_scores, dis_pos_neg)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

test_dataset = SequenceDataset(test_sequences, test_scores, dis_pos_neg)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

# ---3、导入获取 Esm 嵌入的模型：：后期可能使用 lora 模型
print("MODEL BUILDING......")

max_length = 2048
device = torch.device("cuda")
print(device)
Esm_model = EsmEmbedding(model_name, max_length, device)
Esm_model = Esm_model.to(device)
dropout = nn.Dropout(dropout_rate).to(device)  # 在训练时添加 Dropout 层

# 添加卷积层
hidden_size = Esm_model(train_sequences[:1], device).size(-1)
conv_layer = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1).to(device)

# 添加二分类线性层
classifier = nn.Linear(hidden_size, 2).to(device)

# 初始化损失函数和优化器
classify_criterion = ClassifyLoss()
# regression_criterion = RegressionLoss()
# regression_criterion = RegressionLossMaxPool()
# 初始化回归损失函数

regression_criterion = RegressionLossAttention(input_size).to(device)
# 添加 weight_decay 参数进行 L2 正则化
optimizer = optim.Adam(list(Esm_model.parameters()) + list(conv_layer.parameters()) + list(classifier.parameters()), 
                       lr=learning_rate, weight_decay=weight_decay)

# 第一次反向传播前进行梯度清零
optimizer.zero_grad()

# 初始化最小损失为无穷大
min_loss = float('inf')

# 训练过程
for epoch in range(num_epochs):
    running_loss = 0.0
    # 使用 tqdm 包装 train_dataloader 来显示进度条
    for sequences, scores, labels in tqdm(train_dataloader, desc=f'Epoch {epoch + 1} Training'):
        scores = scores.float().to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 分类对比学习 - 前向传播
        contrastive_features = Esm_model(sequences, device)
        contrastive_features = dropout(contrastive_features)  # 在训练时应用 Dropout
        # 使用卷积层处理特征
        contrastive_features = contrastive_features.permute(0, 2, 1)  # 调整维度以适应卷积层输入
        contrastive_features = conv_layer(contrastive_features)
        contrastive_features = contrastive_features.permute(0, 2, 1)  # 调整回原来的维度

        class_labels = torch.arange(len(contrastive_features)).to(device)
        contrastive_loss = classify_criterion(contrastive_features, class_labels, scores)

        # 正样本回归对比学习
        positive_indices = labels == 1
        if positive_indices.sum() > 0:
            positive_sequences = [sequences[i] for i in torch.where(positive_indices)[0]]
            regression_features = Esm_model(positive_sequences, device)
            regression_features = dropout(regression_features)  # 在训练时应用 Dropout
            # 使用卷积层处理特征
            regression_features = regression_features.permute(0, 2, 1)  # 调整维度以适应卷积层输入
            regression_features = conv_layer(regression_features)
            regression_features = regression_features.permute(0, 2, 1)  # 调整回原来的维度
            positive_scores = scores[positive_indices]
            regression_loss = regression_criterion(regression_features, positive_scores)

            # 定义两个损失的权重
            alpha = 1  # 对比损失的权重
            beta = 0.5   # 回归损失的权重

            # 加权求和损失
            combined_loss = alpha * contrastive_loss + beta * regression_loss
        else:
            # 只有对比损失
            combined_loss = contrastive_loss

        # 反向传播
        combined_loss.backward()
        optimizer.step()
        running_loss += combined_loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')
    with open(loss_file_path, 'a+') as loss_file:
        loss_file.write(f'{epoch + 1}   {epoch_loss}\n')

    # 如果当前损失小于最小损失，保存模型并更新最小损失
    if epoch_loss < min_loss:
        min_loss = epoch_loss
        torch.save(Esm_model.state_dict(), model_save_path)
        torch.save(conv_layer.state_dict(), conv_save_path)
        torch.save(classifier.state_dict(), classifier_save_path)
        print(f"✅ Epoch {epoch + 1}: 最小损失更新为 {min_loss},模型已保存。")


# 测试过程
Esm_model.load_state_dict(torch.load(model_save_path, map_location=device))
conv_layer.load_state_dict(torch.load(conv_save_path, map_location=device))
classifier.load_state_dict(torch.load(classifier_save_path, map_location=device))
Esm_model.eval()
conv_layer.eval()
classifier.eval()
correct = 0
total = 0
pre_labels = []

# 用于存储预测结果和真实值的列表
all_sequences = []
all_truescores = []
all_prescores = []
all_truelabels = []
all_prelabels = []
# 使用 tqdm 包装 test_dataloader 来显示进度条

with torch.no_grad():
    for sequences, scores, labels in tqdm(test_dataloader, desc='Testing'):
        scores = scores.float().to(device)
        labels = labels.to(device)

        features = Esm_model(sequences, device)
        features = dropout(features)
        # 使用卷积层处理特征
        features = features.permute(0, 2, 1)  # 调整维度以适应卷积层输入
        features = conv_layer(features)
        features = features.permute(0, 2, 1)  # 调整回原来的维度
        features = features.mean(dim=1)  # 对 sequence_length 维度求平均

        logits = classifier(features)
        _, predicted = torch.max(logits, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 将预测结果转换为 numpy 数组
        predicted_numpy = predicted.cpu().numpy()
        features_numpy = features.cpu().numpy()

        for i, (label, feature, pred) in enumerate(zip(labels.cpu().numpy(), features_numpy, predicted_numpy)):
            all_prelabels.append(pred)

            if pred == 1:
                # 正样本，这里简单假设使用特征向量的最大值作为预测分数，可根据实际情况调整
                pred_score = feature.max()
            else:
                pred_score = feature.max()

            all_prescores.append(pred_score)

        # 收集预测结果和真实值
        all_sequences.extend(sequences)
        all_truescores.extend(scores.cpu().numpy().flatten())
        all_truelabels.extend(labels.cpu().numpy().flatten())

loss = np.mean((np.array(all_truescores) - np.array(all_prescores)) ** 2)
print(f"Testing Loss: {loss}\n")

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
with open(loss_file_path, 'a+') as loss_file:
    loss_file.write(f'Test Accuracy:    {accuracy * 100:.2f}%\n')
    loss_file.write(f'Test Loss:    {loss}\n')
# print("pre_labels", pre_labels)

# 创建 DataFrame
data = {
    'truescore': all_truescores,
    'prescore': all_prescores,
    'truelabel': all_truelabels,
    'prelabels': all_prelabels,
    'sequence': all_sequences
}
df = pd.DataFrame(data)

# 保存为 CSV 文件
df.to_csv(prediction_file_path, index=False, sep=',')


# 后续获取 embedding 的代码保持不变
import os
if not os.path.exists(em_path):
    os.mkdir(em_path)

# 创建模型实例
Esm_model = EsmEmbedding(model_name, max_length, device)
Esm_model = Esm_model.to(device)

# 加载训练好的模型参数
try:
    Esm_model.load_state_dict(torch.load(model_save_path, map_location=device))
    Esm_model.eval()
    print("模型参数加载成功！")
except FileNotFoundError:
    print(f"错误: 未找到模型文件 {model_save_path}。")
except Exception as e:
    print(f"错误: 加载模型参数时出现问题: {e}")

# 定义函数来获取新序列的 embedding
def get_embeddings(model, conv_layer, sequences, device, max_length):
    # target_seq_length = 56
    model.eval()
    conv_layer.eval()
    embeddings = []
    with torch.no_grad():
        for sequence in sequences:
            # 调用 EsmEmbedding 类的 forward 方法
            emb = model([sequence], device)
            emb = emb.permute(0, 2, 1)  # 调整维度以适应卷积层输入
            emb = conv_layer(emb)
            emb = emb.permute(0, 2, 1)  # 调整回原来的维度
            emb = emb.squeeze(0)
            embeddings.append(emb.cpu().numpy())
    return embeddings

# 从 CSV 文件读取数据并进行 embedding 保存
def process_csv_and_save_embeddings(model, conv_layer, csv_path, device, max_length, em_path):
    df = pd.read_csv(csv_path)
    variant_ids = df['ID'].tolist()
    sequences = df['seq'].tolist()
    embeddings = get_embeddings(model, conv_layer, sequences, device, max_length)
    for variant_id, embedding in zip(variant_ids, embeddings):
        np.save(f"{em_path}{variant_id}.npy", embedding)
        # print(f"{em_path}{variant_id}.npy Embedding 保存完成！")

# 替换为实际的 CSV 文件路径
process_csv_and_save_embeddings(Esm_model, conv_layer, '/root/autodl-tmp/lc/data/indel.result.csv', 
                                device, max_length, em_path)

# 导入所需的包
import numpy as np

# 导入npy文件路径位置
test = np.load(f'{em_path}Sp1.npy')

print(test.shape)
    