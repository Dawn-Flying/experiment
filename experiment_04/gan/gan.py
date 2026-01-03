import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

# ----------------------------
# 全局配置
# ----------------------------
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
nclass = len(classes)
nz = 100      # 噪声维度
ngf = 64      # 生成器特征图基数
ndf = 64      # 判别器特征图基数
nc = 3        # 通道数 (RGB)


# ----------------------------
# Step 1: 保存真实图像（按类别）
# ----------------------------
def prepare_real_images():
    real_root = "./cifar10_real"
    os.makedirs(real_root, exist_ok=True)

    for cls in classes:
        os.makedirs(os.path.join(real_root, cls), exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = torchvision.datasets.CIFAR10(
        root='./CIFARdata', train=True, download=True, transform=transform
    )

    print(f"Saving {len(dataset)} real CIFAR-10 images to {real_root} (by class)...")
    class_counts = {cls: 0 for cls in classes}

    for i, (img_tensor, label) in enumerate(dataset):
        cls_name = classes[label]
        img_pil = transforms.ToPILImage()(img_tensor)
        idx = class_counts[cls_name]
        img_pil.save(os.path.join(real_root, cls_name, f"{idx:05d}.png"))
        class_counts[cls_name] += 1

        if (i + 1) % 5000 == 0:
            print(f"Saved {i + 1}/{len(dataset)} images")

    print(f"✅ Real images saved by class to: {os.path.abspath(real_root)}")
    return real_root


# ----------------------------
# Step 2: 权重初始化函数
# ----------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ----------------------------
# Step 3: 条件 DCGAN 模型
# ----------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入: (nz + nclass) x 1 x 1
            nn.ConvTranspose2d(nz + nclass, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出: (nc) x 32 x 32
        )

    def forward(self, noise, labels_onehot):
        # noise: [B, nz]
        # labels_onehot: [B, nclass]
        gen_input = torch.cat((noise, labels_onehot), dim=1)  # [B, nz + nclass]
        return self.main(gen_input.unsqueeze(2).unsqueeze(3))  # reshape to [B, C, 1, 1]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入: (nc + nclass) x 32 x 32
            nn.Conv2d(nc + nclass, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出: 1
        )

    def forward(self, imgs, labels_onehot):
        # imgs: [B, 3, 32, 32]
        # labels_onehot: [B, nclass]
        B, _, H, W = imgs.shape
        label_maps = labels_onehot.view(B, nclass, 1, 1).expand(-1, -1, H, W)  # [B, nclass, 32, 32]
        d_input = torch.cat((imgs, label_maps), dim=1)  # [B, 3+nclass, 32, 32]
        output = self.main(d_input)
        return output.view(-1, 1).squeeze(1)  # [B]


# ----------------------------
# Step 4: 训练并保存生成图像（带类别名）
# ----------------------------
import matplotlib.pyplot as plt  # 新增导入

def train_gan_and_save_generated(num_epochs=20, save_every=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    lr = 0.0002
    beta1 = 0.5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./CIFARdata', train=True, download=True, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    print(netG)
    print(netD)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    # 固定噪声用于可视化
    fixed_noise = torch.randn(100, nz, device=device)
    fixed_labels = torch.arange(10).repeat_interleave(10).to(device)
    fixed_labels_onehot = torch.zeros(100, nclass, device=device)
    fixed_labels_onehot.scatter_(1, fixed_labels.unsqueeze(1), 1)

    gen_dir = "./generated_images"
    os.makedirs(gen_dir, exist_ok=True)

    # 📈 新增：用于记录损失
    G_losses = []
    D_losses = []

    print("🚀 Starting conditional GAN training...")
    for epoch in range(num_epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        num_batches = 0

        for i, (real_imgs, real_labels) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            b_size = real_imgs.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            netD.zero_grad()
            real_labels_onehot = torch.zeros(b_size, nclass, device=device)
            real_labels_onehot.scatter_(1, real_labels.unsqueeze(1), 1)

            output_real = netD(real_imgs, real_labels_onehot)
            errD_real = criterion(output_real, torch.ones_like(output_real))
            errD_real.backward()

            noise = torch.randn(b_size, nz, device=device)
            fake_labels = torch.randint(0, nclass, (b_size,), device=device)
            fake_labels_onehot = torch.zeros(b_size, nclass, device=device)
            fake_labels_onehot.scatter_(1, fake_labels.unsqueeze(1), 1)
            fake_imgs = netG(noise, fake_labels_onehot)
            output_fake = netD(fake_imgs.detach(), fake_labels_onehot)
            errD_fake = criterion(output_fake, torch.zeros_like(output_fake))
            errD_fake.backward()

            optimizerD.step()
            d_loss = errD_real + errD_fake

            # -----------------
            # Train Generator
            # -----------------
            netG.zero_grad()
            output_fake_g = netD(fake_imgs, fake_labels_onehot)
            errG = criterion(output_fake_g, torch.ones_like(output_fake_g))
            errG.backward()
            optimizerG.step()

            # 累加损失
            g_loss_epoch += errG.item()
            d_loss_epoch += d_loss.item()
            num_batches += 1

        # 计算平均损失
        avg_g_loss = g_loss_epoch / num_batches
        avg_d_loss = d_loss_epoch / num_batches
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss_D: {avg_d_loss:.4f}, Loss_G: {avg_g_loss:.4f}")

        # 保存生成图像
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                fake_imgs = netG(fixed_noise, fixed_labels_onehot).detach().cpu()
            fake_imgs = (fake_imgs + 1) / 2.0
            for idx in range(fake_imgs.size(0)):
                cls_name = classes[fixed_labels[idx].item()]
                filename = f"gen_{epoch+1:03d}_{cls_name}_{idx:04d}.png"
                save_image(fake_imgs[idx], os.path.join(gen_dir, filename))

    # 📊 绘制并保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(G_losses, label='Generator Loss', color='blue')
    plt.plot(D_losses, label='Discriminator Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Conditional DCGAN Loss Curves on CIFAR-10')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = "loss_curve_20.png"
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"✅ Loss curve saved to: {os.path.abspath(loss_plot_path)}")

    return gen_dir

# ----------------------------
# Step 5: 使用 torch-fidelity 计算指标
# ----------------------------
def fidelity_metric(generated_images_path, real_images_path):
    try:
        import torch_fidelity
    except ImportError:
        print("⚠️ torch-fidelity not installed. Skipping FID/IS calculation.")
        print("Install it via: pip install torch-fidelity")
        return {"fid": "N/A", "isc": "N/A"}

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=generated_images_path,
        input2=real_images_path,
        cuda=torch.cuda.is_available(),
        isc=True,
        fid=True,
        kid=False,
        verbose=False,
        samples_find_deep=True
    )
    return metrics_dict


# ----------------------------
# Main
# ----------------------------
def main():
    # Step 1: 准备真实图像（按类别）
    if not os.path.exists("./cifar10_real") or sum(len(files) for _, _, files in os.walk("./cifar10_real")) < 100:
        real_dir = prepare_real_images()
    else:
        real_dir = "./cifar10_real"
        print(f"📁 Using existing real images at: {os.path.abspath(real_dir)}")

    # Step 2: 训练条件 GAN
    gen_dir = train_gan_and_save_generated(num_epochs=100, save_every=20)

    # Step 3: 计算指标
    print("\n🔍 Calculating fidelity metrics (this may take a few minutes)...")
    metrics = fidelity_metric(gen_dir, real_dir)

    print("\n📊 Evaluation Results:")
    print("-" * 30)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.upper()}: {value:.4f}")
        else:
            print(f"{key.upper()}: {value}")


if __name__ == "__main__":
    main()
