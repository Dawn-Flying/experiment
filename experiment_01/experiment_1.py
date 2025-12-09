import glob
import os
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

def plot_loss_curve(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 自定义Dataset
class MyDataset(Dataset):
    def __init__(self, data_path: str):
        self.image_files = list(Path(data_path).glob("*.bmp"))

        # self.img_2_tensor = transforms.ToTensor()
        # self.tensor_2_img = transforms.ToPILImage()

        self.transforms = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # 灰度图 (28,28)
        image_tensor = self.transforms(image)
        label = int(img_path.stem.split("_")[0])
        # print(label)
        return image_tensor, label


if __name__ == '__main__':
    # 1.创建数据集
    train_dataset = MyDataset("./data/TrainingSet")
    test_dataset = MyDataset("./data/TestSet")
    model_path = "./model.pth"
    misclassified = './misclassified'
    batch_size = 512

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=0)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=0)

    # 2.创建模型
    model = LeNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not os.path.exists(model_path):

        # 3. 模型训练
        lr = 1e-3
        num_epochs = 20
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses = []

        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)

            acc = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {acc:.2f}%")

        plot_loss_curve(train_losses)
        torch.save(model.state_dict(), model_path)
    else:
        model.eval()
        model.load_state_dict(torch.load(model_path))

    # 输入为 (1, 28, 28) 的图像
    summary(model, input_size=(1, 28, 28))

    # 4. 模型评估
    # 创建保存错误图像的目录
    # 存储所有预测和标签
    all_labels = []
    all_preds = []
    os.makedirs(misclassified, exist_ok=True)

    with torch.no_grad():
        count = 0
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # 保存错分图像
            wrong_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in wrong_indices:
                img = images[idx].cpu()
                true_label = labels[idx].cpu().item()
                pred_label = predicted[idx].cpu().item()
                count += 1
                filename = f'misclassified/true_{true_label}_pred_{pred_label}_{count}.png'
                save_image(img, filename)
    print(f"错判总数：{count}")
    # 转为数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)])

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("/nConfusion Matrix:/n", cm)
    print("/nClassification Report:/n", report)

    # ====== 绘制混淆矩阵图 ======
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # 调整字体大小

    # 创建热力图
    sns.heatmap(
        cm,
        annot=True,  # 显示数字
        fmt='d',  # 整数格式
        cmap='Blues',  # 颜色主题
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)]
    )

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    output_dir = './visualized_misclassified'
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有错判图像文件
    image_files = glob.glob(os.path.join(misclassified, "*.png"))

    # 字典：按真实标签分类
    label_groups = {}

    for file_path in image_files:
        filename = os.path.basename(file_path)
        # 解析文件名：true_X_pred_Y_*.png
        parts = filename.split('_')
        true_label = int(parts[1])  # true_后第一个数字
        pred_label = int(parts[3])  # pred_后第一个数字
        label = f"{true_label}_{pred_label}"
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(file_path)

    # 遍历每个真实标签组，每8张一组显示
    for label, files in label_groups.items():
        print(f"Processing label: {label}, total images: {len(files)}")

        # 每次处理8张
        for i in range(0, len(files), 8):
            batch = files[i:i + 8]
            fig, axes = plt.subplots(1, min(8, len(batch)), figsize=(20, 4))
            if len(batch) == 1:
                axes = [axes]  # 如果只有一个子图，转成列表

            for idx, img_path in enumerate(batch):
                img = Image.open(img_path).convert('RGB')  # 确保是RGB
                axes[idx].imshow(img)
                axes[idx].set_title(f'pred: {img_path.split("_")[3]}')
                axes[idx].axis('off')

            plt.suptitle(f'Label: {label} (batch {i // 8 + 1})')
            plt.tight_layout()
            output_file = os.path.join(output_dir, f'true_{label}_batch_{i // 8 + 1}.png')
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close()
