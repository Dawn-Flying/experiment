"分批读取CIFAR-10图片并将部分批次保存为图片文件"
import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = CIFAR10(root='./data', download=True,
                  transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # 喂入大小是把原来数据集中的多少图片组合成一张图片
batch_size = 64
for batch_idx, data in enumerate(dataloader):
    if batch_idx == len(dataloader) - 1:
        continue
    real_images, _ = data

    print('#{} has {} images.'.format(batch_idx, batch_size))
    if batch_idx % 100 == 0:
        path = './data/CIFAR10_shuffled_batch{:03d}.png'.format(batch_idx)
        save_image(real_images, path, normalize=True)

"搭建生成网络和鉴别网络"
"隐藏的卷积层(即除了最后的输出卷积层外)的输出都需要经过规范化操作"
import torch.nn as nn

# 搭建生成网络
latent_size = 64  # 潜在大小
n_channel = 3  # 输出通道数
n_g_feature = 64  # 生成网络隐藏层大小
"生成网络采用了四层转置卷积操作"
gnet = nn.Sequential(
    # 输入大小 = (64, 1, 1)
    # 有点像互相关的反操作，(x-4)/1=1-->x=4
    nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4,
                       bias=False),
    nn.BatchNorm2d(4 * n_g_feature),
    nn.ReLU(),
    # 大小 = (256, 4, 4)
    # {x+2(填充)-4(核尺寸)+2(步长)}/2=4-->x=8
    nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4,
                       stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_g_feature),
    nn.ReLU(),
    # 大小 = (128, 8, 8)
    nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4,
                       stride=2, padding=1, bias=False),
    nn.BatchNorm2d(n_g_feature),
    nn.ReLU(),
    # 大小 = (64, 16, 16)
    nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4,
                       stride=2, padding=1),
    nn.Sigmoid(),
    # 图片大小 = (3, 32, 32)
)
print(gnet)

# 搭建鉴别网络
n_d_feature = 64  # 鉴别网络隐藏层大小
"鉴别网络采用了4层互相关操作"
dnet = nn.Sequential(
    # 图片大小 = (3, 32, 32)
    nn.Conv2d(n_channel, n_d_feature, kernel_size=4,
              stride=2, padding=1),
    nn.LeakyReLU(0.2),
    # 大小 = (64, 16, 16)
    nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4,
              stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_d_feature),
    nn.LeakyReLU(0.2),
    # 大小 = (128, 8, 8)
    nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4,
              stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4 * n_d_feature),
    nn.LeakyReLU(0.2),
    # 大小 = (256, 4, 4)
    nn.Conv2d(4 * n_d_feature, 1, kernel_size=4),
    # 对数赔率张量大小 = (1, 1, 1)
)
print(dnet)

gnet = gnet.to(device)
dnet = dnet.to(device)

"初始化权重值"
import torch.nn.init as init
import matplotlib.pyplot as plt

def weights_init(m):  # 用于初始化权重值的函数
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)


# 调用apply()函数，torch.nn.Module类实例会递归地让自己成为weights_init()里面函数的m
gnet.apply(weights_init)
dnet.apply(weights_init)



if __name__ == '__main__':
    "训练生成网络和鉴别网络并输出图片"
    import torch
    import torch.optim

    # 损失
    criterion = nn.BCEWithLogitsLoss()

    # 优化器
    #Adam优化器的默认学习率n=0.01,过高，应减小为0.002，动量参数默认0.9，会造成震荡，减小为0.5
    goptimizer = torch.optim.Adam(gnet.parameters(),
        lr=0.0002, betas=(0.5, 0.999))
    doptimizer = torch.optim.Adam(dnet.parameters(),
        lr=0.0002, betas=(0.5, 0.999))

    # 用于测试的固定噪声,用来查看相同的潜在张量在训练过程中生成图片的变换
    batch_size = 64
    fixed_noises = torch.randn(batch_size, latent_size, 1, 1, device=device)

    # 创建保存目录
    os.makedirs('./generated_images', exist_ok=True)
    os.makedirs('./real_images_for_fid', exist_ok=True)

    # === 先保存所有真实图像（用于 FID）===
    print("Saving real CIFAR-10 images for FID...")
    real_img_count = 0
    for batch_idx, (real_imgs, _) in enumerate(
            DataLoader(CIFAR10(root='./data', train=True, transform=transforms.ToTensor()), batch_size=64)):
        for i in range(real_imgs.size(0)):
            save_image(real_imgs[i], f'./real_images_for_fid/real_{real_img_count:05d}.png')
            real_img_count += 1
        if real_img_count >= 50000:
            break
    print(f"Saved {real_img_count} real images to ./real_images_for_fid")

    # 训练过程
    # === 训练记录 ===
    G_losses = []
    D_losses = []

    epoch_num = 10
    for epoch in range(epoch_num):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        num_batches = 0

        for batch_idx, data in enumerate(dataloader):
            if batch_idx==len(dataloader)-1: #剔除最后一张是(16,3,32,32)
                continue
            # 载入本批次数据
            real_images, _ = data#real_images(64,3,32,32)
            real_images = real_images.to(device)  # 👈 关键：数据上 GPU

            # 训练鉴别网络
            labels = torch.ones(batch_size, device=device) # 真实数据对应标签为1(64,)
            preds = dnet(real_images) # 对真实数据进行判别(64,1,1,1)

            outputs = preds.reshape(-1)#(64,)
            dloss_real = criterion(outputs, labels) # 真实数据的鉴别器损失
            dmean_real = outputs.sigmoid().mean() # 计算鉴别器将多少比例的真数据判定为真,仅用于输出显示

            noises = torch.randn(batch_size, latent_size, 1, 1, device=device) # 潜在噪声(64,64,1,1)
            fake_images = gnet(noises) # 生成假数据(64,3,32,32)
            labels = torch.zeros(batch_size, device=device) # 假数据对应标签为0
            fake = fake_images.detach()# 使得梯度的计算不回溯到生成网络,可用于加快训练速度.删去此步结果不变
            preds = dnet(fake) # 对假数据进行鉴别
            outputs = preds.view(-1)

            dloss_fake = criterion(outputs, labels) # 假数据的鉴别器损失
            dmean_fake = outputs.sigmoid().mean()
                    # 计算鉴别器将多少比例的假数据判定为真,仅用于输出显示

            dloss = dloss_real + dloss_fake # 总的鉴别器损失
            dnet.zero_grad()
            dloss.backward()
            doptimizer.step()

            # 训练生成网络
            labels = torch.ones(batch_size, device=device)
                    # 生成网络希望所有生成的数据都被认为是真数据
            preds = dnet(fake_images) # 把假数据通过鉴别网络
            outputs = preds.view(-1)
            gloss = criterion(outputs, labels) # 真数据看到的损失
            gmean_fake = outputs.sigmoid().mean()
                    # 计算鉴别器将多少比例的假数据判定为真,仅用于输出显示
            gnet.zero_grad()
            gloss.backward()
            goptimizer.step()

            # --- Accumulate losses ---
            g_loss_epoch += gloss.item()
            d_loss_epoch += dloss.item()
            num_batches += 1

            # 输出本步训练结果
            if batch_idx % 100 == 0:
                print('[{}/{}]'.format(epoch, epoch_num) +
                        '[{}/{}]'.format(batch_idx, len(dataloader)) +
                        '鉴别网络损失:{:g} 生成网络损失:{:g}'.format(dloss, gloss) +
                        '真数据判真比例:{:g} 假数据判真比例:{:g}/{:g}'.format(
                        dmean_real, dmean_fake, gmean_fake))
                fake = gnet(fixed_noises) # 由固定潜在张量生成假数据
                save_image(fake, # 保存假数据
                        './data/images_epoch{:02d}_batch{:03d}.png'.format(
                        epoch, batch_idx))

        # --- Epoch 结束 ---
        avg_g = g_loss_epoch / num_batches
        avg_d = d_loss_epoch / num_batches
        G_losses.append(avg_g)
        D_losses.append(avg_d)

        print(f"[Epoch {epoch}/{epoch_num}] G_loss: {avg_g:.4f}, D_loss: {avg_d:.4f}")

        # === 保存 50,000 张生成图像用于 FID ===
        print("Generating 50,000 fake images for FID...")
        gen_img_count = 0
        with torch.no_grad():
            while gen_img_count < 50000:
                noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
                fake_imgs = gnet(noise)
                for i in range(fake_imgs.size(0)):
                    if gen_img_count >= 50000:
                        break
                    save_image(fake_imgs[i].cpu(), f'./generated_images/fake_{gen_img_count:05d}.png', normalize=True)
                    gen_img_count += 1
        print(f"Saved {gen_img_count} generated images to ./generated_images")

        # === 绘制损失曲线 ===
        plt.figure(figsize=(10, 5))
        plt.plot(G_losses, label='Generator Loss')
        plt.plot(D_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('DCGAN Training Loss (BCEWithLogitsLoss)')
        plt.legend()
        plt.grid(True)
        plt.savefig('./data/gan_loss_curve.png')
        plt.show()


        # === 调用 FID/IS 评估 ===
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


        print("Computing FID and IS...")
        results = fidelity_metric('./generated_images', './real_images_for_fid')
        print("Evaluation Results:")
        print(f"  FID: {results.get('frechet_inception_distance', 'N/A')}")
        print(f"  IS (mean): {results.get('inception_score_mean', 'N/A')}")
        print(f"  IS (std): {results.get('inception_score_std', 'N/A')}")

