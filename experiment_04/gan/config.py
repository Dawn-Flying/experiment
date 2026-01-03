import torch

class CFG:
    # Data
    image_size = 32
    channels = 3
    batch_size = 128
    num_workers = 1

    # Diffusion
    timesteps = 1000
    beta_schedule = "cosine"  # or "linear"

    # Model
    base_channels = 128
    channel_mults = [1, 2, 2, 4]  # downsample 3 times: 32 → 16 → 8 → 4

    # Training
    epochs = 1000
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sampling
    ddim_steps = 50
    eta = 0.0  # deterministic DDIM

    # Paths
    ckpt_dir = "outputs/checkpoints"
    sample_dir = "outputs/samples"

cfg = CFG()