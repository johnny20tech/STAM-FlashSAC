import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def compute_tanh_jacobian_correction(pre_tanh_action: torch.Tensor) -> torch.Tensor:
    """Computes the log det jacobian of the tanh transform safely."""
    return 2.0 * (math.log(2.0) - pre_tanh_action - F.softplus(-2.0 * pre_tanh_action))


def generate_zeta_distribution(max_k: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Generates choices and probabilities for Zeta noise repetition."""
    k_values = np.arange(1, max_k + 1, dtype=np.float32)
    pmf = k_values ** (-2.0)
    probabilities = pmf / np.sum(pmf)
    return np.arange(1, max_k + 1), probabilities


def build_lr_scheduler(optimizer, peak_lr, end_lr, warmup_steps, total_decay_steps, init_lr=None):
    """Warmup + cosine decay, matching the official FlashSAC scheduler semantics."""
    init_lr = peak_lr if init_lr is None else init_lr

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            lr = init_lr + (peak_lr - init_lr) * (step / warmup_steps)
            return lr / peak_lr
        elif step < total_decay_steps:
            progress = (step - warmup_steps) / max(total_decay_steps - warmup_steps, 1)
            cosine_decay = end_lr + (peak_lr - end_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            return cosine_decay / peak_lr
        return end_lr / peak_lr

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
