# -*- coding: utf-8 -*-
import os

import seaborn as sns
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# =================== 1. 数据集定义 ===================
class FundusDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历 disease 和 normal 文件夹
        for label, folder in enumerate(['disease', 'normal']):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for img_name in os.listdir(folder_path):
                self.images.append(os.path.join(folder_path, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# =================== 2. 数据增强与预处理 ===================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# =================== 3. 加载数据 ===================
train_dataset = FundusDataset(root_dir="./data/2-MedImage-TrainSet", transform=data_transforms['val'])
val_dataset = FundusDataset(root_dir="./data/2-MedImage-TestSet", transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# =================== 4. 模型构建 ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)  # 二分类输出
model.to(device)

# =================== 5. 训练循环 ===================
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

train_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

torch.save(model.state_dict(), './model.pth')

# 绘制Loss变化图
plt.plot(train_losses, label='Training Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# =================== 6. 验证并预测 ===================
model.eval()
y_true = []
y_scores = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.float().to(device)
        outputs = model(inputs).squeeze()
        y_true.extend(labels.cpu().numpy())
        y_scores.extend(torch.sigmoid(outputs).cpu().numpy())

# =================== 7. 绘制 ROC 曲线 ===================
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"ROC AUC Score: {roc_auc:.4f}")

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

y_pred = [1 if score >= 0.5 else 0 for score in y_scores]

conf_matrix = confusion_matrix(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print("Confusion Matrix: \n", conf_matrix)
print("F1 Score: ", f1)
print("Accuracy: ", acc)
print("Precision: ", precision)
print("Recall: ", recall)

# ====== 绘制混淆矩阵图 ======
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # 调整字体大小

# 创建热力图
sns.heatmap(
    conf_matrix,
    annot=True,  # 显示数字
    fmt='d',  # 整数格式
    cmap='Blues',  # 颜色主题
    xticklabels=[str(i) for i in range(2)],
    yticklabels=[str(i) for i in range(2)]
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
