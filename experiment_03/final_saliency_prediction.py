import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
# 设置Matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import metric

# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# 设置数据路径
TRAIN_PATH = "D:/博/机器学习基础/data/3-Saliency-TrainSet/3-Saliency-TrainSet"
TEST_PATH = "D:/博/机器学习基础/data/3-Saliency-TestSet/3-Saliency-TestSet"

# 设置参数
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4

# 自定义数据集类
class SaliencyDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = []
        self.saliency_paths = []
        
        # 获取所有类别
        categories = os.listdir(os.path.join(data_path, "Stimuli"))
        
        for category in categories:
            # 获取当前类别的图像和显著图路径
            stimuli_path = os.path.join(data_path, "Stimuli", category)
            fixation_path = os.path.join(data_path, "FIXATIONMAPS", category)
            
            # 检查FIXATIONMAPS目录是否存在
            if not os.path.exists(fixation_path):
                print(f"警告：FIXATIONMAPS目录不存在 {fixation_path}")
                continue
            
            # 获取所有图像文件名
            image_files = os.listdir(stimuli_path)
            
            # 获取FIXATIONMAPS目录中的所有文件
            fixation_files = os.listdir(fixation_path)
            fixation_files_set = set(fixation_files)
            
            for file in image_files:
                if file.endswith(".jpg"):
                    # 检查显著图文件是否存在
                    if file not in fixation_files_set:
                        # 尝试其他可能的扩展名
                        possible_extensions = [".png", ".bmp", ".jpeg"]
                        found = False
                        for ext in possible_extensions:
                            alt_file = file.replace(".jpg", ext)
                            if alt_file in fixation_files_set:
                                file = alt_file
                                found = True
                                break
                        if not found:
                            print(f"警告：显著图文件不存在 {os.path.join(fixation_path, file)}")
                            continue
                    
                    # 保存图像和显著图路径
                    img_path = os.path.join(stimuli_path, file)
                    saliency_path = os.path.join(fixation_path, file)
                    self.image_paths.append(img_path)
                    self.saliency_paths.append(saliency_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        img_path = img_path.replace('\\', '/')
        
        try:
            # 使用np.fromfile和cv2.imdecode来处理中文路径
            img_data = np.fromfile(img_path, dtype=np.uint8)
            image = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
            
            # 调整大小
            image = cv2.resize(image, IMAGE_SIZE)
            
            # 如果是灰度图，转换为RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # 如果是BGR格式，转换为RGB
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"警告：无法读取图像 {img_path}，错误：{str(e)}")
            # 返回一个默认图像
            image = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
        
        # 归一化到[0, 1]
        image = image.astype(np.float32) / 255.0
        
        # 加载对应的显著图
        saliency_path = self.saliency_paths[idx]
        saliency_path = saliency_path.replace('\\', '/')
        
        try:
            # 使用np.fromfile和cv2.imdecode来处理中文路径
            saliency_data = np.fromfile(saliency_path, dtype=np.uint8)
            saliency = cv2.imdecode(saliency_data, cv2.IMREAD_GRAYSCALE)
            
            # 调整大小
            saliency = cv2.resize(saliency, IMAGE_SIZE)
        except Exception as e:
            print(f"警告：无法读取显著图 {saliency_path}，错误：{str(e)}")
            # 返回一个默认显著图
            saliency = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        
        # 归一化到[0, 1]
        saliency = saliency.astype(np.float32) / 255.0
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            saliency = self.target_transform(saliency)
        
        return image, saliency

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor并归一化到[0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
])

target_transform = transforms.Compose([
    transforms.ToTensor()  # 转换为Tensor
])

# 最终修复版模型架构 - 更简单、更可靠的U-Net结构
class FinalSaliencyModel(nn.Module):
    def __init__(self):
        super(FinalSaliencyModel, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 中间层
        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 1, 1)
        )
    
    def forward(self, x):
        # 编码器
        x1 = self.encoder[:5](x)
        x2 = self.encoder[5:10](x1)
        x3 = self.encoder[10:15](x2)
        x4 = self.encoder[15:20](x3)
        
        # 中间层
        x = self.middle(x4)
        
        # 解码器
        x = self.decoder[:4](x)
        x = self.decoder[4:8](x)
        x = self.decoder[8:12](x)
        x = self.decoder[12:16](x)
        x = self.decoder[16:](x)
        
        return x

# 训练模型
def train_model():
    # 创建数据集
    print("加载训练数据...")
    train_dataset = SaliencyDataset(TRAIN_PATH, transform=transform, target_transform=target_transform)
    test_dataset = SaliencyDataset(TEST_PATH, transform=transform, target_transform=target_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"训练数据集大小：{len(train_dataset)}")
    print(f"测试数据集大小：{len(test_dataset)}")
    
    # 构建模型
    model = FinalSaliencyModel().to(device)
    print(model)
    
    # 定义损失函数和优化器
    # 使用BCEWithLogitsLoss损失函数，结合了Sigmoid和BCE
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 使用Adam优化器，降低学习率
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    # 训练模型
    print("开始训练...")
    import sys
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"GPU内存使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        
        print(f'\n====== Epoch {epoch+1}/{EPOCHS} ======')
        for i, (images, saliency_maps) in enumerate(train_loader):
            # 调试信息
            if i % 10 == 0:
                print(f"处理批次 {i}/{len(train_loader)}")
                if torch.cuda.is_available():
                    print(f"GPU内存使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    print(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            # 将数据移到设备上
            images = images.to(device)
            saliency_maps = saliency_maps.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, saliency_maps)
            # 应用Sigmoid激活函数计算MAE
            outputs_sigmoid = torch.sigmoid(outputs)
            mae = nn.L1Loss()(outputs_sigmoid, saliency_maps)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 累加损失
            running_loss += loss.item() * images.size(0)
            running_mae += mae.item() * images.size(0)
            
            # 打印批次信息
            if (i+1) % 10 == 0 or (i+1) == len(train_loader):
                print(f'[Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}]')
                print(f'  损失值: {loss.item():.6f}')
                print(f'  MAE值: {mae.item():.6f}')
                print(f'  输出值范围: {outputs_sigmoid.min().item():.4f} - {outputs_sigmoid.max().item():.4f}')
            
            # 保存日志（每10个批次保存一次）
            if (i+1) % 10 == 0 or (i+1) == len(train_loader):
                with open('final_training_log.txt', 'a') as f:
                    f.write(f'[Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}]\n')
                    f.write(f'  损失值: {loss.item():.6f}\n')
                    f.write(f'  MAE值: {mae.item():.6f}\n')
                    f.write(f'  模型输出值范围: {outputs.min().item():.4f} 到 {outputs.max().item():.4f}\n')
                    f.write(f'  模型输出均值: {outputs.mean().item():.4f}\n')
                    f.write(f'  模型输出标准差: {outputs.std().item():.4f}\n')
                    f.write(f'  Sigmoid后输出范围: {outputs_sigmoid.min().item():.4f} 到 {outputs_sigmoid.max().item():.4f}\n')
                    f.write(f'  Sigmoid后输出均值: {outputs_sigmoid.mean().item():.4f}\n')
            
            # 清理内存
            del images, saliency_maps, outputs, outputs_sigmoid, loss, mae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 计算平均损失
        epoch_loss = running_loss / len(train_dataset)
        epoch_mae = running_mae / len(train_dataset)
        train_losses.append(epoch_loss)
        train_maes.append(epoch_mae)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_mae = 0.0
        
        with torch.no_grad():
            for batch_idx, (images, saliency_maps) in enumerate(test_loader):
                # 调试信息
                if batch_idx % 10 == 0:
                    print(f"处理验证批次 {batch_idx}/{len(test_loader)}")
                    if torch.cuda.is_available():
                        print(f"GPU内存使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                        print(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                
                # 将数据移到设备上
                images = images.to(device)
                saliency_maps = saliency_maps.to(device)
                
                # 前向传播
                outputs = model(images)
                
                # 计算损失
                loss = criterion(outputs, saliency_maps)
                # 应用Sigmoid激活函数计算MAE
                outputs_sigmoid = torch.sigmoid(outputs)
                mae = nn.L1Loss()(outputs_sigmoid, saliency_maps)
                
                # 累加损失
                val_running_loss += loss.item() * images.size(0)
                val_running_mae += mae.item() * images.size(0)
                
                # 清理内存
                del images, saliency_maps, outputs, outputs_sigmoid, loss, mae
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 计算平均验证损失
        val_epoch_loss = val_running_loss / len(test_dataset)
        val_epoch_mae = val_running_mae / len(test_dataset)
        val_losses.append(val_epoch_loss)
        val_maes.append(val_epoch_mae)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Train Loss: {epoch_loss:.4f}, Train MAE: {epoch_mae:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val MAE: {val_epoch_mae:.4f}")
    
    # 保存模型
    model_path = os.path.join(os.getcwd(), "final_saliency_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型保存完成：{model_path}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练和验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制MAE曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='训练MAE')
    plt.plot(val_maes, label='验证MAE')
    plt.title('训练和验证MAE曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    curve_path = os.path.join(os.getcwd(), "final_training_curves.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"训练曲线保存完成：{curve_path}")
    
    return model, test_loader

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    
    # 存储预测结果
    all_y_test = []
    all_y_pred = []
    
    with torch.no_grad():
        for images, saliency_maps in test_loader:
            # 将数据移到设备上
            images = images.to(device)
            saliency_maps = saliency_maps.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 转换为numpy数组
            saliency_maps_np = saliency_maps.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            
            # 存储结果
            all_y_test.extend(saliency_maps_np)
            all_y_pred.extend(outputs_np)
    
    # 计算性能指标
    cc_scores = []
    kl_scores = []
    
    for i in range(len(all_y_test)):
        gt = all_y_test[i].squeeze()
        pred = all_y_pred[i].squeeze()
        
        # 计算相关系数
        cc = metric.calc_cc_score(gt, pred)
        cc_scores.append(cc)
        
        # 计算KL散度
        kl = metric.KLD(gt, pred)
        kl_scores.append(kl)
    
    # 计算平均指标
    avg_cc = np.mean(cc_scores)
    avg_kl = np.mean(kl_scores)
    
    print(f"平均相关系数 (CC): {avg_cc:.4f}")
    print(f"平均KL散度: {avg_kl:.4f}")
    
    # 保存评估结果
    with open('final_evaluation_results.txt', 'w') as f:
        f.write(f"平均相关系数 (CC): {avg_cc:.4f}\n")
        f.write(f"平均KL散度: {avg_kl:.4f}\n")
    
    # 分析预测结果分布
    print("\n预测结果分布分析：")
    all_pred_flat = np.array(all_y_pred).flatten()
    print(f"预测值均值: {np.mean(all_pred_flat):.4f}")
    print(f"预测值标准差: {np.std(all_pred_flat):.4f}")
    print(f"预测值最小值: {np.min(all_pred_flat):.4f}")
    print(f"预测值最大值: {np.max(all_pred_flat):.4f}")
    print(f"预测值大于0.1的比例: {np.sum(all_pred_flat > 0.1) / len(all_pred_flat):.4f}")
    
    return np.array(all_y_test), np.array(all_y_pred)

# 可视化结果
def visualize_results(test_loader, y_test, y_pred, num_samples=5):
    plt.figure(figsize=(15, 3 * num_samples))
    
    # 获取原始图像（未经过变换的）
    test_dataset = test_loader.dataset
    
    for i in range(num_samples):
        # 获取原始图像
        img_path = test_dataset.image_paths[i]
        img_data = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, IMAGE_SIZE)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 真实显著图
        gt = y_test[i].squeeze()
        
        # 预测显著图
        pred = y_pred[i].squeeze()
        
        # 绘制原始图像
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title(f'原始图像 {i+1}')
        plt.axis('off')
        
        # 绘制真实显著图
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(gt, cmap='gray')
        plt.title(f'真实显著图 {i+1}')
        plt.axis('off')
        
        # 绘制预测显著图
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred, cmap='gray')
        plt.title(f'预测显著图 {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('final_saliency_results.png')
    plt.close()
    print("结果可视化完成：final_saliency_results.png")

# 主函数
if __name__ == "__main__":
    # 训练模型
    model, test_loader = train_model()
    
    # 评估模型
    y_test, y_pred = evaluate_model(model, test_loader)
    
    # 可视化结果
    visualize_results(test_loader, y_test, y_pred)
    
    print("最终修复版显著性预测模型训练和评估完成！")