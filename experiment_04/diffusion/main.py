"main.py"
import torch
import numpy as np

from torch import nn, optim
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from scheduler import Scheduler
from UNet import UNet


def plot_training_loss(epoch_losses):
    """
    绘制训练过程中每个epoch的平均损失。

    参数:
        epoch_losses: 每个epoch的平均损失列表。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.xticks(range(1, len(epoch_losses) + 1))
    plt.tight_layout()
    plt.savefig('training_loss_curve.png', dpi=300)
    plt.show()


if __name__ == '__main__':

    device = torch.device("cuda")
    batch_size = 512
    lr = 2e-5
    epochs = 2000
    denoise_steps = 250

    train_dataset = datasets.CIFAR10(
        root='../data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5, inplace=True)
        ])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = UNet(
        in_channels=3,
        out_channels=3,
        block_channels=[64, 128, 256],
        use_attention=[False, False, False],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = Scheduler(model, denoise_steps)

    epoch_losses = []
    model.train()
    for epoch in range(epochs):

        print('*' * 40)

        train_loss = []

        for i, data in enumerate(train_loader, 1):

            x, _ = data
            x = Variable(x).to(device)

            t = torch.randint(low=0, high=denoise_steps, size=(x.shape[0],)).to(device)
            training_loss = scheduler.training_losses(x, t)

            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
            train_loss.append(training_loss.item())

        avg_loss = np.mean(train_loss)
        epoch_losses.append(avg_loss)

        torch.save(model.state_dict(), "unet-cifar10.pth")
        print('Finish  {}  Loss: {:.6f}'.format(epoch + 1, np.mean(train_loss)))

    plot_training_loss(epoch_losses)

    model.eval()

    xs = np.array(scheduler.sample((16, 3, 32, 32), device))

    step_25 = xs[24]
    step_50 = xs[49]
    step_75 = xs[74]
    step_100 = xs[99]
    step_125 = xs[124]
    step_150 = xs[149]
    step_175 = xs[174]
    step_200 = xs[199]
    step_225 = xs[224]
    step_250 = xs[-1]

    x = np.concatenate([step_25, step_50, step_75, step_100, step_125,
                        step_150, step_175, step_200, step_225, step_250], axis=-1)
    x = x.transpose(0, 2, 3, 1)
    x = x.reshape(-1, 32 * 10, 3).clip(-1, 1)
    x = (x + 1) / 2
    x = x.astype(np.float32)

    plt.imsave('result1.png', x)


