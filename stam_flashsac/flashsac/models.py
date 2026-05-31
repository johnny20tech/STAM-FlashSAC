import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from stam_flashsac.stam import STAMContextModule
from stam_flashsac.flashsac.utils import compute_tanh_jacobian_correction


class SphericalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias=False)
        nn.init.orthogonal_(self.layer.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    def project_weights(self):
        self.layer.weight.data.copy_(F.normalize(self.layer.weight.data, p=2, dim=-1, eps=1e-8))

class SphericalBatchNorm(nn.Module):
    def __init__(self, channels: int, momentum: float = 0.01):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.register_buffer("rm", torch.zeros(channels))
        self.register_buffer("rv", torch.ones(channels))
        self.momentum = momentum

    def forward(self, x: torch.Tensor, is_training: bool) -> torch.Tensor:
        return F.batch_norm(
            x, self.rm, self.rv, self.gamma, self.beta,
            training=is_training, momentum=self.momentum, eps=1e-5
        )

    def project_weights(self):
        dim = self.gamma.shape[-1]
        norm_scale = math.sqrt(dim) * torch.rsqrt(torch.sum(self.gamma**2 + self.beta**2, dim=-1, keepdim=True) + 1e-8)
        self.gamma.data.copy_(self.gamma.data * norm_scale)
        self.beta.data.copy_(self.beta.data * norm_scale)

class SphericalRMSNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.scale.shape, self.scale, eps=1e-6)

    def project_weights(self):
        dim = self.scale.shape[-1]
        norm_scale = math.sqrt(dim) * torch.rsqrt(torch.sum(self.scale**2, dim=-1, keepdim=True) + 1e-8)
        self.scale.data.copy_(self.scale.data * norm_scale)


# ---------------------------------------------------------
# Vectorized Ensemble Modules (For Critic)
# ---------------------------------------------------------
class ParallelSphericalLinear(nn.Module):
    def __init__(self, n_models: int, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_models, out_features, in_features))
        for i in range(n_models):
            nn.init.orthogonal_(self.W.data[i], gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nbi,noi->nbo", x, self.W)

    def project_weights(self):
        self.W.data.copy_(F.normalize(self.W.data, p=2, dim=-1, eps=1e-8))

class ParallelSphericalBatchNorm(nn.Module):
    def __init__(self, n_models: int, channels: int, momentum: float = 0.01):
        super().__init__()
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(n_models, channels))
        self.beta = nn.Parameter(torch.zeros(n_models, channels))
        self.register_buffer("rm", torch.zeros(n_models, channels))
        self.register_buffer("rv", torch.ones(n_models, channels))

    def forward(self, x: torch.Tensor, is_training: bool) -> torch.Tensor:
        if is_training:
            batch_mean = x.mean(dim=1, keepdim=True)
            batch_var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                b_size = x.shape[1]
                self.rm.lerp_(batch_mean.squeeze(1).float(), self.momentum)
                self.rv.lerp_((batch_var.squeeze(1) * (b_size / (b_size - 1))).float(), self.momentum)
            x_norm = (x - batch_mean) * torch.rsqrt(batch_var + 1e-5)
        else:
            x_norm = (x - self.rm.unsqueeze(1)) * torch.rsqrt(self.rv.unsqueeze(1) + 1e-5)
        return x_norm * self.gamma.unsqueeze(1) + self.beta.unsqueeze(1)

    def project_weights(self):
        dim = self.gamma.shape[-1]
        norm_scale = math.sqrt(dim) * torch.rsqrt(torch.sum(self.gamma**2 + self.beta**2, dim=-1, keepdim=True) + 1e-8)
        self.gamma.data.copy_(self.gamma.data * norm_scale)
        self.beta.data.copy_(self.beta.data * norm_scale)

class ParallelSphericalRMSNorm(nn.Module):
    def __init__(self, n_models: int, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_models, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
        return (x / rms) * self.scale.unsqueeze(1)

    def project_weights(self):
        dim = self.scale.shape[-1]
        norm_scale = math.sqrt(dim) * torch.rsqrt(torch.sum(self.scale**2, dim=-1, keepdim=True) + 1e-8)
        self.scale.data.copy_(self.scale.data * norm_scale)


# ---------------------------------------------------------
# Architecture Building Blocks
# ---------------------------------------------------------
class ResidualControlBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.expand = SphericalLinear(dim, dim * expansion)
        self.squeeze = SphericalLinear(dim * expansion, dim)
        self.bn1 = SphericalBatchNorm(dim * expansion)
        self.bn2 = SphericalBatchNorm(dim)

    def forward(self, x, is_training):
        identity = x
        out = F.relu(self.bn1(self.expand(x), is_training=is_training))
        out = F.relu(self.bn2(self.squeeze(out), is_training=is_training))
        return out + identity

class ParallelResidualBlock(nn.Module):
    def __init__(self, n_models: int, dim: int, expansion: int = 4):
        super().__init__()
        self.expand = ParallelSphericalLinear(n_models, dim, dim * expansion)
        self.squeeze = ParallelSphericalLinear(n_models, dim * expansion, dim)
        self.bn1 = ParallelSphericalBatchNorm(n_models, dim * expansion)
        self.bn2 = ParallelSphericalBatchNorm(n_models, dim)

    def forward(self, x, is_training):
        identity = x
        out = F.relu(self.bn1(self.expand(x), is_training=is_training))
        out = F.relu(self.bn2(self.squeeze(out), is_training=is_training))
        return out + identity

class GaussianActionHead(nn.Module):
    def __init__(self, dim: int, action_dim: int):
        super().__init__()
        self.mean_proj = SphericalLinear(dim, action_dim)
        self.mean_b = nn.Parameter(torch.zeros(action_dim))
        self.std_proj = SphericalLinear(dim, action_dim)
        self.std_b = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mu = F.linear(x, self.mean_proj.layer.weight, self.mean_b)
        raw_std = F.linear(x, self.std_proj.layer.weight, self.std_b)
        log_std = -10.0 + 6.0 * (1 + torch.tanh(raw_std))
        return mu, torch.exp(log_std)

class CategoricalValueHead(nn.Module):
    def __init__(self, n_models: int, dim: int, n_atoms: int, v_min: float, v_max: float):
        super().__init__()
        self.proj = ParallelSphericalLinear(n_models, dim, n_atoms)
        self.b = nn.Parameter(torch.zeros(n_models, n_atoms))
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms).reshape(1, 1, -1))

    def forward(self, x):
        logits = self.proj(x) + self.b.unsqueeze(1)
        log_probs = F.log_softmax(logits, dim=-1)
        expected_q = torch.sum(torch.exp(log_probs) * self.support, dim=-1)
        return expected_q, log_probs

# ---------------------------------------------------------
# Actor & Critic Networks
# ---------------------------------------------------------
class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, stam_cfg: dict | None = None):
        super().__init__()
        self.stem_bn = SphericalBatchNorm(state_dim)
        self.stem_linear = SphericalLinear(state_dim, hidden_dim)
        self.backbone = nn.ModuleList([ResidualControlBlock(hidden_dim) for _ in range(2)])
        self.stam_gate = None
        if stam_cfg and stam_cfg.get('use_stam', False) and stam_cfg.get('stam_apply_to', 'both') in {'both', 'actor'}:
            self.stam_gate = STAMContextModule(
                stam_cfg['stam_context_dim'],
                stam_cfg['stam_hidden_dim'],
                hidden_dim,
                stam_cfg['stam_context_len'],
                encoder=stam_cfg.get('stam_encoder', 'lstm'),
                fusion=stam_cfg.get('stam_fusion', 'gate'),
            )
        self.head_norm = SphericalRMSNorm(hidden_dim)
        self.action_head = GaussianActionHead(hidden_dim, action_dim)

    def get_distribution_params(self, x, context=None, is_training=True):
        x = self.stem_linear(self.stem_bn(x, is_training))
        for layer in self.backbone:
            x = layer(x, is_training)
        if self.stam_gate is not None:
            x = self.stam_gate(context, x)
        return self.action_head(self.head_norm(x))

    def forward(self, x, context=None, is_training=True):
        mu, sigma = self.get_distribution_params(x, context, is_training)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - compute_tanh_jacobian_correction(z)
        return action, log_prob.sum(dim=-1, keepdim=True)

class DistributionalCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_atoms: int = 101, v_range: tuple = (-5.0, 5.0), stam_cfg: dict | None = None):
        super().__init__()
        self.stem_bn = ParallelSphericalBatchNorm(2, input_dim)
        self.stem_linear = ParallelSphericalLinear(2, input_dim, hidden_dim)
        self.backbone = nn.ModuleList([ParallelResidualBlock(2, hidden_dim) for _ in range(2)])
        self.stam_gate = None
        if stam_cfg and stam_cfg.get('use_stam', False) and stam_cfg.get('stam_apply_to', 'both') in {'both', 'critic'}:
            self.stam_gate = STAMContextModule(
                stam_cfg['stam_context_dim'],
                stam_cfg['stam_hidden_dim'],
                hidden_dim,
                stam_cfg['stam_context_len'],
                n_models=2,
                encoder=stam_cfg.get('stam_encoder', 'lstm'),
                fusion=stam_cfg.get('stam_fusion', 'gate'),
            )
        self.head_norm = ParallelSphericalRMSNorm(2, hidden_dim)
        self.q_head = CategoricalValueHead(2, hidden_dim, n_atoms, v_range[0], v_range[1])

    def forward(self, state, action, context=None, is_training=True):
        x = torch.cat([state, action], dim=-1).unsqueeze(0).expand(2, -1, -1)
        x = self.stem_linear(self.stem_bn(x, is_training))
        for layer in self.backbone:
            x = layer(x, is_training)
        if self.stam_gate is not None:
            x = self.stam_gate(context, x)
        return self.q_head(self.head_norm(x))
