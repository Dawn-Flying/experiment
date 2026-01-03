import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import cfg
from unet import UNet
from diffusion import Diffusion


if __name__ == '__main__':
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.sample_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(root='./CIFARdata', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    model = UNet(base_channels=cfg.base_channels, dim_mults=cfg.channel_mults, channels=cfg.channels).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    diffusion = Diffusion(timesteps=cfg.timesteps, beta_schedule=cfg.beta_schedule, device=cfg.device)

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(dataloader)
        for x, _ in pbar:
            x = x.to(cfg.device)
            t = torch.randint(0, cfg.timesteps, (x.shape[0],), device=cfg.device).long()
            loss = diffusion.p_losses(model, x, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                samples = diffusion.ddim_sample(model, shape=(16, cfg.channels, cfg.image_size, cfg.image_size), ddim_steps=cfg.ddim_steps, eta=cfg.eta)
                samples = (samples + 1) / 2  # [-1,1] → [0,1]
                grid = torchvision.utils.make_grid(samples, nrow=4)
                torchvision.utils.save_image(grid, f"{cfg.sample_dir}/epoch_{epoch}.png")
            torch.save(model.state_dict(), f"{cfg.ckpt_dir}/model_epoch_{epoch}.pth")