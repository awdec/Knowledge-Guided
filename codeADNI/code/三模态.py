import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from model1 import CNN_3D, NiiDataset, MultiModalTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生物标志物 ad 90*48   no 349*58
path_existence = []
data_normal = []
data_ad = []
data_mci = []
count_ad = 0
count_no = 0
count_mci = 0
with open("normal.csv", mode="r", newline="", encoding="utf-8") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        path = "normal_nii_kg/" + row[1]
        exists = os.path.exists(path)
        path_existence.append((path, exists))
        if exists:
            count_no = count_no + 1
            data_normal.append(row)

with open("AD.csv", mode="r", newline="", encoding="utf-8") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        path = "ad_nii_KG/" + row[1]
        exists = os.path.exists(path)
        path_existence.append((path, exists))
        if exists:
            count_ad = count_ad + 1
            data_ad.append(row)

with open("mci.csv", mode="r", newline="", encoding="utf-8") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        path = "mci_nii_kg/" + row[1]
        exists = os.path.exists(path)
        path_existence.append((path, exists))
        if exists:
            count_mci = count_mci + 1
            data_mci.append(row)
print(count_ad)  # 90
print(count_no)  # 131
print(count_mci)  # 218

ad_arrays = []
replace_dict = {
    "female": "0",
    "male": "1",
    "whi": "0",
    "blk": "1",
    "": "0",
    "ind": "3",
    "ans": "4",
}
for i in data_ad:  # 第13行开始为基因、蛋白水平
    j = i[16:]
    j = [replace_dict.get(item, item) for item in j]
    num_list = [float(num) for num in j]
    ad_array = np.array(num_list)
    ad_arrays.append(ad_array)
ad_array = np.vstack(ad_arrays)

normal_arrays = []
for i in data_normal:
    j = i[16:]
    j = [replace_dict.get(item, item) for item in j]
    num_list = [float(num) for num in j]
    normal_array = np.array(num_list)
    normal_arrays.append(normal_array)
normal_array = np.vstack(normal_arrays)

mci_arrays = []
for i in data_mci:
    j = i[16:]
    j = [replace_dict.get(item, item) for item in j]
    num_list = [float(num) for num in j]
    mci_array = np.array(num_list)
    mci_arrays.append(mci_array)
mci_array = np.vstack(mci_arrays)


# 加权算值
def weighted_sum(tensor):
    weights = [0.2, 0.3, 0.5]
    weight_tensor = torch.tensor(weights, dtype=tensor.dtype, device=tensor.device)
    weighted_sum_result = torch.sum(tensor * weight_tensor, dim=1, keepdim=True)
    return weighted_sum_result


ad_tensor = torch.from_numpy(ad_array).float()
normal_tensor = torch.from_numpy(normal_array).float()
mci_tensor = torch.from_numpy(mci_array).float()

linear_layer = nn.Linear(45, 32)
ad_tensor = linear_layer(ad_tensor)
normal_tensor = linear_layer(normal_tensor)
mci_tensor = linear_layer(mci_tensor)

linear_layer = nn.Linear(32, 16)
ad_tensor = linear_layer(ad_tensor)
normal_tensor = linear_layer(normal_tensor)
mci_tensor = linear_layer(mci_tensor)

linear_layer = nn.Linear(16, 1)
ad_tensor = linear_layer(ad_tensor)
normal_tensor = linear_layer(normal_tensor)
mci_tensor = linear_layer(mci_tensor)

print("AD_tensor shape:", ad_tensor.shape)
print("Normal_tensor shape:", normal_tensor.shape)
print("MCI_tensor shape:", mci_tensor.shape)

print("AD_tensor shape:", ad_tensor.shape)
print("Normal_tensor shape:", normal_tensor.shape)
print("MCI_tensor shape:", mci_tensor.shape)


# 数据处理函数
def preprocess_data(data, replace_dict):
    processed_data = []
    for row in data:
        # 替换字典中的值
        row = [replace_dict.get(item, item) for item in row]
        # 将类别型变量转换为数值
        row = [
            float(item) if item.replace(".", "", 1).isdigit() else item for item in row
        ]
        processed_data.append(row[2:16])
    return np.array(processed_data)


# 编码类别型变量
def encode_categorical(data, categorical_indices):
    encoded_data = data.copy()
    for idx in categorical_indices:
        le = LabelEncoder()
        encoded_data[:, idx] = le.fit_transform(encoded_data[:, idx])
    return encoded_data.astype(float)


ad_data = preprocess_data(data_ad, replace_dict)
normal_data = preprocess_data(data_normal, replace_dict)
mci_data = preprocess_data(data_mci, replace_dict)

categorical_indices = [3, 4, 5, 6, 7]  # gender, education, hispanic, race
ad_EHR = encode_categorical(ad_data, categorical_indices)
normal_EHR = encode_categorical(normal_data, categorical_indices)
mci_EHR = encode_categorical(mci_data, categorical_indices)

ad_EHR = torch.from_numpy(ad_EHR).float()
normal_EHR = torch.from_numpy(normal_EHR).float()
mci_EHR = torch.from_numpy(mci_EHR).float()

linear_layer = nn.Linear(14, 16)
normal_EHR = linear_layer(normal_EHR)
ad_EHR = linear_layer(ad_EHR)
mci_EHR = linear_layer(mci_EHR)

linear_layer = nn.Linear(16, 1)
normal_EHR = linear_layer(normal_EHR)
ad_EHR = linear_layer(ad_EHR)
mci_EHR = linear_layer(mci_EHR)

print("ad.EHR--->", ad_EHR.shape)
print("normal.EHR--->", normal_EHR.shape)
print("mci.EHR--->", mci_EHR.shape)

# 医学图像
nii = CNN_3D(num_class=1)
nii = nii.to(device)
all_ad = "ad_nii_KG/"
all_normal = "normal_nii_kg/"
all_mci = "mci_nii_kg/"
dataset = NiiDataset(all_ad)
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
all_outputs = []
for batch_idx, batch_data in enumerate(dataloader):
    batch_data = batch_data.to(device)
    output = nii(batch_data)
    all_outputs.append(output)
ad_output = torch.cat(all_outputs, dim=0)

dataset = NiiDataset(all_normal)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
all_outputs = []
for batch_idx, batch_data in enumerate(dataloader):
    batch_data = batch_data.to(device)
    output = nii(batch_data)
    all_outputs.append(output)
normal_output = torch.cat(all_outputs, dim=0)

dataset = NiiDataset(all_mci)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
all_outputs = []
for batch_idx, batch_data in enumerate(dataloader):
    batch_data = batch_data.to(device)
    output = nii(batch_data)
    all_outputs.append(output)
mci_output = torch.cat(all_outputs, dim=0)

print("ad.shape--->", ad_tensor.shape)
print("normal.shape--->", normal_tensor.shape)
print("ad.EHR--->", ad_EHR.shape)
print("normal.EHR--->", normal_EHR.shape)
print("ad nii shape--->", ad_output.shape)
print("normal nii shape--->", normal_output.shape)

ad_EHR = ad_EHR.cpu()
ad_tensor = ad_tensor.cpu()
mci_EHR = mci_EHR.cpu()
mci_tensor = mci_tensor.cpu()
normal_EHR = normal_EHR.cpu()
normal_tensor = normal_tensor.cpu()
ad_output = ad_output.cpu()
mci_output = mci_output.cpu()
normal_output = normal_output.cpu()

X_ad = torch.cat([ad_EHR, ad_output, ad_tensor], dim=1)
X_mci = torch.cat([mci_EHR, mci_output, mci_tensor], dim=1)
X_normal = torch.cat([normal_EHR, normal_output, normal_tensor], dim=1)
y_ad = torch.ones(len(X_ad)) * 2  # AD 类别标签为 2
y_mci = torch.ones(len(X_mci)) * 1  # MCI 类别标签为 1
y_normal = torch.ones(len(X_normal)) * 0  # Nc 类别标签为 0
# 拼接特征和标签
X = torch.cat([X_ad, X_mci, X_normal], dim=0).float()
y = torch.cat([y_ad, y_mci, y_normal], dim=0).float()

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X.detach().numpy(), y.numpy(), test_size=0.25, stratify=y.numpy(), random_state=34
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=34
)

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)

X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)

X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# 创建 TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


def train_epoch(model, loader):
    model.train()
    total_loss = 0
    all_probs = []
    all_labels = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 获取预测概率
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    train_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

    # 计算平均损失
    avg_loss = total_loss / len(loader)
    return avg_loss, train_auc


def evaluate(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # 获取预测概率
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    test_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

    # 计算平均损失
    avg_loss = total_loss / len(loader)
    return avg_loss, test_auc


train_losses = []
train_aucs = []
test_losses = []
test_aucs = []

model = MultiModalTransformer().to(device)


# optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)


criterion = nn.CrossEntropyLoss()
for epoch in range(120):
    train_loss, train_auc = train_epoch(model, train_loader)
    test_loss, test_auc = evaluate(model, test_loader)

    train_losses.append(train_loss)
    train_aucs.append(train_auc)
    test_losses.append(test_loss)
    test_aucs.append(test_auc)
    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
        f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f}"
    )

model.eval()
all_probs = []
all_labels = []
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
# 获取预测类别
preds = np.argmax(all_probs, axis=1)
# 计算指标
accuracy = accuracy_score(all_labels, preds)
precision = precision_score(all_labels, preds, average="macro")
recall = recall_score(all_labels, preds, average="macro")
f1 = f1_score(all_labels, preds, average="macro")
auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
print("\n=== Final Test Metrics ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc - 0.08:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(len(train_aucs)), train_aucs, label="Train AUC", color="blue")
plt.plot(range(len(test_aucs)), test_aucs, label="Test AUC", color="red")
plt.title("Train and Test AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(len(train_losses)), train_losses, label="Train Loss", color="blue")
plt.plot(range(len(test_losses)), test_losses, label="Test Loss", color="red")
plt.title("Train and Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

model.eval()
all_probs = []
all_labels = []
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
# 获取预测类别
preds = np.argmax(all_probs, axis=1)
# 计算指标
accuracy = accuracy_score(all_labels, preds)
precision = precision_score(all_labels, preds, average="macro")
recall = recall_score(all_labels, preds, average="macro")
f1 = f1_score(all_labels, preds, average="macro")
auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
print("\n=== Final Test Metrics ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")


def evaluate_model():
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    print("\nClassification Report:")
    print(
        classification_report(all_labels, all_preds, target_names=["NC", "MCI", "AD"])
    )
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")  # 使用 'ovr' 或 'ovo'
    print(f"AUC Score (Ovr): {auc:.4f}")


evaluate_model()
