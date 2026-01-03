"scheduler.py"
import numpy as np

import torch

import torch.nn.functional as F

def extract_into_tensor(arr, timesteps, broadcast_shape):

    res = torch.from_numpy(arr).to(torch.float32).to(device=timesteps.device)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)

class Scheduler:

    def __init__(self, denoise_model, denoise_steps, beta_start=1e-4, beta_end=2e-2):

        self.model = denoise_model

        betas = np.array(
            np.linspace(beta_start, beta_end, denoise_steps),
            dtype=np.float64
        )

        self.denoise_steps = denoise_steps

        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        alphas = 1.0 - betas

        self.sqrt_alphas = np.sqrt(alphas)
        self.one_minus_alphas = 1.0 - alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

    def gaussian_q_sample(self, x0, t, noise):

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def training_losses(self, x, t):

        noise = torch.randn_like(x)
        x_t = self.gaussian_q_sample(x, t, noise)

        predict_noise = self.model(x_t, t)

        return F.mse_loss(predict_noise, noise)

    @torch.no_grad()
    def gaussian_p_sample(self, x_t, t):

        t_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        z = torch.randn_like(x_t) * t_mask

        predict_noise = self.model(x_t, t)

        x = x_t - (
                extract_into_tensor(self.one_minus_alphas, t, x_t.shape)
                * predict_noise
                / extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        )

        x = x / extract_into_tensor(self.sqrt_alphas, t, x_t.shape)

        sigma = torch.sqrt(
            extract_into_tensor(self.one_minus_alphas, t, x_t.shape)
            * (1.0 - extract_into_tensor(self.alphas_cumprod_prev, t, x_t.shape))
            / (1.0 - extract_into_tensor(self.alphas_cumprod, t, x_t.shape))
        )

        x = x + sigma * z

        return x

    @torch.no_grad()
    def sample(self, x_shape, device):

        xs = []

        x = torch.randn(*x_shape, device=device)

        for t in reversed(range(0, self.denoise_steps)):

            t = torch.tensor([t], device=device).repeat(x_shape[0])

            x = self.gaussian_p_sample(x, t)

            xs.append(x.detach().cpu().numpy())

        return xs
