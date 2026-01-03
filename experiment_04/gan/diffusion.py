import torch
import torch.nn.functional as F
from beta_schedule import cosine_beta_schedule

class Diffusion:
    def __init__(self, timesteps=1000, beta_schedule="cosine", device="cpu"):
        self.timesteps = timesteps
        self.device = device

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise NotImplementedError()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        # calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def ddim_sample(self, model, shape, ddim_steps=50, eta=0.0):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        times = torch.linspace(-1, self.timesteps - 1, steps=ddim_steps + 1).int().to(device)
        times = list(reversed(times))[1:]  # skip t=-1

        for i in range(len(times)):
            t = times[i].repeat(b)
            t_next = times[i + 1].repeat(b) if i + 1 < len(times) else torch.full_like(t, -1)

            at = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            at_next = self.alphas_cumprod[t_next].view(-1, 1, 1, 1) if t_next[0] >= 0 else torch.ones_like(at)

            xt = img
            et = model(xt, t)

            x0_t = (xt - torch.sqrt(1 - at) * et) / torch.sqrt(at)
            sigma_t = eta * torch.sqrt((1 - at / at_next) * (1 - at_next) / (1 - at))
            c2 = torch.sqrt(1 - at_next - sigma_t**2)

            img = torch.sqrt(at_next) * x0_t + c2 * et + sigma_t * torch.randn_like(xt)

        return img