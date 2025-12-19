import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os

from experiment_03.SaliencyLoss import SaliencyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_kl_histogram(kl_scores, save_path="kl_divergence_hist.png"):
    """
    绘制 KL 散度的对数直方图 + KDE + 均值/中位数线
    """
    kl_scores = np.array(kl_scores)
    log_kl = np.log10(kl_scores + 1e-8)  # 避免 log(0)

    plt.figure(figsize=(8, 5))
    ax = sns.histplot(log_kl, kde=True, stat="density", bins=30, alpha=0.6, color='skyblue')

    mean_val = np.mean(log_kl)
    median_val = np.median(log_kl)

    plt.axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    plt.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')

    plt.xlabel(r'$\log_{10}$(KL Divergence)')
    plt.ylabel('Density')
    plt.title('Distribution of KL Divergence (log scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"KL 分布图已保存: {save_path}")


def plot_cc_vs_kl_scatter(cc_scores, kl_scores, save_path="cc_vs_kl_scatter.png"):
    """
    绘制 CC vs KL 的散点图，颜色表示点密度
    """
    cc_scores = np.array(cc_scores)
    kl_scores = np.array(kl_scores)

    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(cc_scores, np.log10(kl_scores), gridsize=50, cmap='inferno')
    cb = plt.colorbar(hb)
    cb.set_label('counts in bin')

    plt.xlabel('CC Score')
    plt.ylabel(r'$\log_{10}$(KL Divergence)')
    plt.title('CC vs KL Divergence (colored by point density)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"CC vs KL 散点图已保存: {save_path}")


# 修改 eval 函数以调用绘图函数
def eval(model):
    save_path = SAVE_PATH  # 最佳模型保存路径
    # 加载最佳模型并测试
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\n加载最佳模型（Epoch {checkpoint['epoch'] + 1}）")
    avg_cc, avg_kl, all_cc, all_kl = test_and_evaluate(model, './data/3-Saliency-TestSet',
                                                       save_dir="./mse/saliency_results")  # 修改test_and_evaluate函数返回all_cc, all_kl
    print("测试完成！结果已保存至 saliency_results 目录")

    # 绘制KL分布图和CC vs KL散点图
    plot_kl_histogram(all_kl)
    plot_cc_vs_kl_scatter(all_cc, all_kl)


class SaliencyDataset(Dataset):
    def __init__(self, root_dir, img_size=(256, 256), is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train

        # 递归获取所有图像路径
        self.img_paths = []
        img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
        for root, _, files in os.walk(os.path.join(root_dir, "Stimuli")):
            for file in files:
                if file.lower().endswith(img_extensions):
                    self.img_paths.append(os.path.join(root, file))

        # 匹配掩码路径
        self.mask_paths = []
        for img_path in self.img_paths:
            mask_path = img_path.replace("Stimuli", "FIXATIONMAPS")
            mask_path = os.path.splitext(mask_path)[0]
            found = False
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                candidate = mask_path + ext
                if os.path.exists(candidate):
                    self.mask_paths.append(candidate)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"未找到{img_path}对应的掩码文件")

        # 数据增强
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
        ]) if is_train else None

        print(f"成功加载{len(self.img_paths)}个样本")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取原图并记录尺寸
        img_path = self.img_paths[idx]
        img_ori = cv2.imread(img_path)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img_ori.shape[:2]  # 保存原图尺寸

        # 预处理输入图像（resize到256*256）
        img = cv2.resize(img_ori, self.img_size)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 读取并预处理掩码
        mask_path = self.mask_paths[idx]
        mask_ori = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask_ori, self.img_size)
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        # 数据增强
        if self.transform and self.is_train:
            seed = torch.randint(0, 1000000, (1,)).item()
            torch.manual_seed(seed)
            img = self.transform(img)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        # 返回原图尺寸和原始掩码（用于测试指标计算）
        return img, mask, (ori_h, ori_w), mask_ori, img_ori

import torch.nn as nn
import torchvision.models as models


class ResNet18Saliency(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练ResNet18并拆分编码器
        resnet = models.resnet18(pretrained=pretrained)
        model_file_dir = "./model.pth"
        if os.path.exists(model_file_dir):
            resnet.load_state_dict(torch.load(model_file_dir, map_location=device))
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64通道, 1/2
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64通道, 1/4
        self.encoder3 = resnet.layer2  # 128通道, 1/8
        self.encoder4 = resnet.layer3  # 256通道, 1/16
        self.encoder5 = resnet.layer4  # 512通道, 1/32

        # 解码器：上采样+特征融合
        self.decoder5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.ConvTranspose2d(256 + 256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(64 + 64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.ConvTranspose2d(64 + 64, 1, kernel_size=2, stride=2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器提取多尺度特征
        feat1 = self.encoder1(x)
        feat2 = self.encoder2(feat1)
        feat3 = self.encoder3(feat2)
        feat4 = self.encoder4(feat3)
        feat5 = self.encoder5(feat4)

        # 解码器融合与上采样
        dec5 = self.decoder5(feat5)
        fuse4 = torch.cat([dec5, feat4], dim=1)
        dec4 = self.decoder4(fuse4)

        fuse3 = torch.cat([dec4, feat3], dim=1)
        dec3 = self.decoder3(fuse3)

        fuse2 = torch.cat([dec3, feat2], dim=1)
        dec2 = self.decoder2(fuse2)

        fuse1 = torch.cat([dec2, feat1], dim=1)
        out = self.decoder1(fuse1)

        return self.sigmoid(out)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calc_cc_score(gtsAnn, resAnn):
    # gtsAnn: Ground-truth saliency map
    # resAnn: Predicted saliency map

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]


EPSILON = np.finfo('float').eps


def KLD(p, q):
    # q: Predicted saliency map
    # p: Ground-truth saliency map
    p = normalize(p, method='sum')
    q = normalize(q, method='sum')
    return np.sum(np.where(p != 0, p * np.log((p + EPSILON) / (q + EPSILON)), 0))


def normalize(x, method='standard', axis=None):
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="training", leave=False)
    for imgs, masks, _, _, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"batch_loss": loss.item()})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="validation", position=1, leave=False)
    for imgs, masks, _, _, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def test_and_evaluate(model, test_dir, save_dir="./saliency_results", img_size=(256, 256)):
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    test_dataset = SaliencyDataset(test_dir, img_size=img_size, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_cc = []
    all_kl = []

    # 用于暂存拼接图像
    combined_images = []
    batch_index = 0  # 用于命名文件：comparison_batch_0.png, comparison_batch_1.png...

    pbar = tqdm(test_loader, desc="测试与评估")
    for idx, (img, _, (ori_h, ori_w), mask_ori, img_ori) in enumerate(pbar):
        category = os.path.basename(os.path.dirname(test_dataset.img_paths[idx]))
        cate_save_dir = os.path.join(save_dir, category)
        os.makedirs(cate_save_dir, exist_ok=True)

        # 模型预测
        img = img.to(device)
        saliency_pred = model(img).squeeze().cpu().numpy()  # [256, 256]
        saliency_pred_ori = cv2.resize(saliency_pred, (ori_w.item(), ori_h.item()))  # 回原尺寸
        mask_ori = mask_ori.squeeze().cpu().numpy()

        # 计算指标
        cc_score = calc_cc_score(mask_ori, saliency_pred_ori)
        kl_score = KLD(mask_ori, saliency_pred_ori)
        all_cc.append(cc_score)
        all_kl.append(kl_score)

        # --- 开始拼接原图 + 显著图 ---
        # 原图：HWC, uint8, RGB
        img_ori_np = img_ori.squeeze().cpu().numpy()  # shape: (H, W, 3)
        if img_ori_np.dtype != np.uint8:
            img_ori_np = (img_ori_np * 255).astype(np.uint8)

        # 显著图：转换为 uint8 灰度图（0~255）
        sal_map_uint8 = (saliency_pred_ori * 255).astype(np.uint8)  # shape: (H, W)

        # 将灰度图扩展为三通道（保持黑白，非伪彩色）
        sal_gray_rgb = np.stack([sal_map_uint8] * 3, axis=-1)  # shape: (H, W, 3)

        # 调整尺寸一致（以防 resize 后不一致）
        h, w = img_ori_np.shape[:2]
        sal_gray_resized = cv2.resize(sal_gray_rgb, (w, h))

        # 横向拼接：[原图 | 显著图（灰度）]
        combined = np.concatenate([img_ori_np, sal_gray_resized], axis=1)  # (H, 2W, 3)
        combined_pil = Image.fromarray(combined)

        combined_images.append(combined_pil)

        # 每满12张，保存一张大图
        if len(combined_images) == 1:
            # 创建 3 行 × 4 列 网格（每张子图高度可能不同，这里统一 resize 到相同尺寸）
            grid_h, grid_w = 1, 1
            # 统一子图尺寸（取第一张的尺寸）
            sub_h, sub_w = combined_images[0].height, combined_images[0].width
            canvas = Image.new('RGB', (grid_w * sub_w, grid_h * sub_h), color=(255, 255, 255))

            for i, img_pil in enumerate(combined_images):
                # 如果尺寸不一致，强制 resize（可选）
                if img_pil.size != (sub_w, sub_h):
                    img_pil = img_pil.resize((sub_w, sub_h))
                row = i // grid_w
                col = i % grid_w
                canvas.paste(img_pil, (col * sub_w, row * sub_h))

            # 保存
            grid_path = os.path.join(cate_save_dir, f"comparison_batch_{batch_index}.png")
            canvas.save(grid_path)
            print(f"✅ 已保存对比图: {grid_path}")

            # 清空并更新批次索引
            combined_images.clear()
            batch_index += 1

    # 处理最后不足12张的剩余图像（可选）
    if combined_images:
        grid_h = int(np.ceil(len(combined_images) / 4))
        grid_w = 4
        sub_h, sub_w = combined_images[0].height, combined_images[0].width
        canvas = Image.new('RGB', (grid_w * sub_w, grid_h * sub_h), color=(255, 255, 255))
        for i, img_pil in enumerate(combined_images):
            if img_pil.size != (sub_w, sub_h):
                img_pil = img_pil.resize((sub_w, sub_h))
            row = i // grid_w
            col = i % grid_w
            canvas.paste(img_pil, (col * sub_w, row * sub_h))
        grid_path = os.path.join(cate_save_dir, f"comparison_batch_{batch_index}_last.png")
        canvas.save(grid_path)
        print(f"✅ 已保存最后一批对比图: {grid_path}")

    # 计算平均指标
    avg_cc = np.mean(all_cc)
    avg_kl = np.mean(all_kl)
    print(f"\n测试集平均CC系数：{avg_cc:.4f}")
    print(f"测试集平均KL散度：{avg_kl:.4f}")

    # 保存指标
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"平均CC系数：{avg_cc:.4f}\n")
        f.write(f"平均KL散度：{avg_kl:.4f}\n")

    return avg_cc, avg_kl, all_cc, all_kl

import torchvision.transforms.functional as F
def combine_images(img_ori, saliency_pred_save):
    """
    将原始图像和显著性图横向拼接。

    :param img_ori: 原始RGB图像（numpy array）
    :param saliency_pred_save: 显著性图（灰度图，numpy array）
    :return: 拼接后的图像（PIL Image）
    """
    # 转换为PIL Image以方便操作
    img_ori_pil = F.to_pil_image(img_ori)
    saliency_pred_pil = F.to_pil_image(saliency_pred_save).convert('L')  # 确保是灰度图像

    # 将显著性图转换为伪彩色图以便于观察
    saliency_pred_color = saliency_pred_pil.convert('RGB')
    saliency_pred_color_np = np.array(saliency_pred_color)
    saliency_pred_color_np[:, :, :] = cv2.applyColorMap(np.uint8(saliency_pred_save), cv2.COLORMAP_JET)
    saliency_pred_color = F.to_pil_image(saliency_pred_color_np)

    # 拼接图像
    combined_img = F.hcat([img_ori_pil, saliency_pred_color])

    return combined_img


from PIL import Image
def create_comparison_grid(images_list, grid_size=(3, 4)):
    """
    创建一个用于比较的图像网格。

    :param images_list: 图像列表（每个元素都是一个PIL Image）
    :param grid_size: 网格大小 (rows, cols)
    :return: 组合后的比较图像（PIL Image）
    """
    rows, cols = grid_size
    w, h = images_list[0].size
    grid_image = Image.new('RGB', (cols * w, rows * h))

    for idx, img in enumerate(images_list):
        row_idx = idx // cols
        col_idx = idx % cols
        grid_image.paste(img, (col_idx * w, row_idx * h))

    return grid_image


# ===================== 主函数 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "./mse/resnet18_saliency_best.pth"  # 最佳模型保存路径

def train(model, train_loader, val_loader):
    EPOCHS = 10  # 训练轮数
    LR = 1e-3  # 学习率
    # criterion = nn.MSELoss()  # 损失函数
    criterion = SaliencyLoss(alpha=0.5, beta=0.5).to(device)
    train_losses = []
    val_losses = []

    optimizer = optim.Adam(model.parameters(), lr=LR)  # 优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率衰减

    # 训练主循环
    best_val_loss = float("inf")
    for epoch in tqdm(range(EPOCHS), position=0):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, SAVE_PATH)
            print(f"保存最佳模型（验证损失：{best_val_loss:.4f}）")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    plt.show()


def eval(model):
    save_path = SAVE_PATH  # 最佳模型保存路径
    # 加载最佳模型并测试
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\n加载最佳模型（Epoch {checkpoint['epoch'] + 1}）")
    avg_cc, avg_kl, all_cc, all_kl = test_and_evaluate(model, './data/3-Saliency-TestSet', save_dir="./mse/saliency_results")
    print("测试完成！结果已保存至 saliency_results 目录")

    # 绘制KL分布图和CC vs KL散点图
    plot_kl_histogram(all_kl)
    plot_cc_vs_kl_scatter(all_cc, all_kl)


def start():
    dummy_input = torch.rand(10, 3, 256, 256).to(device)
    model = ResNet18Saliency(pretrained=False).to(device)

    print("输入尺寸:")
    print(f"{dummy_input.shape}")
    print()

    # 打印网络结构
    print("网络结构:")
    print(model)

    # 计算并打印参数量
    total_params, trainable_params = count_parameters(model)
    print("参数量:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print()

    # 前向传播
    output = model(dummy_input)

    IMG_SIZE = (256, 256)  # 图像尺寸
    BATCH_SIZE = 16  # 批次大小

    train_dataset = SaliencyDataset('./data/3-Saliency-TrainSet', img_size=IMG_SIZE, is_train=True)
    val_dataset = SaliencyDataset('./data/3-Saliency-TestSet', img_size=IMG_SIZE, is_train=False)  # 若有独立验证集可替换路径
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 打印输出尺寸和结果
    print("输出尺寸:")
    print(f"{output.shape}")
    print()
    print("输出结果示例:")
    print(f"样本 {0 + 1}示例 = {output[0]}")

    # train(model, train_loader, val_loader)
    eval(model)


if __name__ == "__main__":
    start()