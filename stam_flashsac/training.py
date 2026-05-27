import os
import sys
import csv
import json
import math
import copy
import warnings
import argparse
from collections import deque

# ---------------------------------------------------------
# Environment & Runtime Configuration
# ---------------------------------------------------------
package_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(package_dir)
runtime_dir = os.path.join(repo_dir, "runtime")
tmp_dir = os.path.join(runtime_dir, "tmp")
cache_dir = os.path.join(runtime_dir, "cache")
mpl_dir = os.path.join(runtime_dir, "matplotlib")
inductor_cache_dir = os.path.join(runtime_dir, "torchinductor_compile_no_cudagraphs")
for path in (tmp_dir, cache_dir, mpl_dir, inductor_cache_dir):
    os.makedirs(path, exist_ok=True)

os.environ.update({
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "JAX_DEFAULT_MATMUL_PRECISION": "float32",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "TMPDIR": tmp_dir,
    "TEMP": tmp_dir,
    "TMP": tmp_dir,
    "XDG_CACHE_HOME": cache_dir,
    "MPLCONFIGDIR": mpl_dir,
    "TORCHINDUCTOR_CACHE_DIR": inductor_cache_dir,
    "TORCHINDUCTOR_USE_CUDAGRAPHS": "0",
})

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch._inductor.config as inductor_config
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore")
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False

# Enable TF32 for RTX 3090/4090 performance boost
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from stam_flashsac.vendor.flash_rl.envs.isaaclab import make_isaaclab_env


# ---------------------------------------------------------
# Math & Utility Functions
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Spherical/Orthogonal Neural Network Modules
# ---------------------------------------------------------
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

class RISEContextGate(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int, feature_dim: int, n_models: int | None = None):
        super().__init__()
        self.n_models = n_models
        self.lstm = nn.LSTM(context_dim, hidden_dim, batch_first=True)
        if n_models is None:
            self.proj = nn.Linear(hidden_dim, feature_dim)
        else:
            self.proj = nn.Linear(hidden_dim, n_models * feature_dim)

    def forward(self, context: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if context is None:
            return features
        _, (h_n, _) = self.lstm(context)
        gate = torch.sigmoid(self.proj(h_n[-1]))
        if self.n_models is None:
            return features * gate
        gate = gate.view(context.shape[0], self.n_models, -1).permute(1, 0, 2)
        return features * gate


# ---------------------------------------------------------
# Actor & Critic Networks
# ---------------------------------------------------------
class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, rise_cfg: dict | None = None):
        super().__init__()
        self.stem_bn = SphericalBatchNorm(state_dim)
        self.stem_linear = SphericalLinear(state_dim, hidden_dim)
        self.backbone = nn.ModuleList([ResidualControlBlock(hidden_dim) for _ in range(2)])
        self.rise_gate = None
        if rise_cfg and rise_cfg.get('use_rise', False):
            self.rise_gate = RISEContextGate(
                rise_cfg['rise_context_dim'], rise_cfg['rise_hidden_dim'], hidden_dim
            )
        self.head_norm = SphericalRMSNorm(hidden_dim)
        self.action_head = GaussianActionHead(hidden_dim, action_dim)

    def get_distribution_params(self, x, context=None, is_training=True):
        x = self.stem_linear(self.stem_bn(x, is_training))
        for layer in self.backbone:
            x = layer(x, is_training)
        if self.rise_gate is not None:
            x = self.rise_gate(context, x)
        return self.action_head(self.head_norm(x))

    def forward(self, x, context=None, is_training=True):
        mu, sigma = self.get_distribution_params(x, context, is_training)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - compute_tanh_jacobian_correction(z)
        return action, log_prob.sum(dim=-1, keepdim=True)

class DistributionalCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_atoms: int = 101, v_range: tuple = (-5.0, 5.0), rise_cfg: dict | None = None):
        super().__init__()
        self.stem_bn = ParallelSphericalBatchNorm(2, input_dim)
        self.stem_linear = ParallelSphericalLinear(2, input_dim, hidden_dim)
        self.backbone = nn.ModuleList([ParallelResidualBlock(2, hidden_dim) for _ in range(2)])
        self.rise_gate = None
        if rise_cfg and rise_cfg.get('use_rise', False):
            self.rise_gate = RISEContextGate(
                rise_cfg['rise_context_dim'], rise_cfg['rise_hidden_dim'], hidden_dim, n_models=2
            )
        self.head_norm = ParallelSphericalRMSNorm(2, hidden_dim)
        self.q_head = CategoricalValueHead(2, hidden_dim, n_atoms, v_range[0], v_range[1])

    def forward(self, state, action, context=None, is_training=True):
        x = torch.cat([state, action], dim=-1).unsqueeze(0).expand(2, -1, -1)
        x = self.stem_linear(self.stem_bn(x, is_training))
        for layer in self.backbone:
            x = layer(x, is_training)
        if self.rise_gate is not None:
            x = self.rise_gate(context, x)
        return self.q_head(self.head_norm(x))


# ---------------------------------------------------------
# Core Components
# ---------------------------------------------------------
class EmpiricalReturnTracker:
    """Tracks global return variance to normalize incoming rewards."""
    def __init__(self, gamma: float, limit: float, device: torch.device):
        self.gamma, self.limit, self.device = gamma, limit, device
        self.running_ret = torch.zeros(1, device=device)
        self.max_ret_seen = torch.zeros(1, device=device)
        
        self.mu = torch.zeros(1, device=device)
        self.sigma2 = torch.ones(1, device=device)
        self.n_samples = torch.tensor(0.0, device=device)

    def process_step(self, r: float, is_terminal: bool):
        with torch.no_grad():
            r_t = torch.tensor([r], dtype=torch.float32, device=self.device)
            d_t = torch.tensor([float(is_terminal)], dtype=torch.float32, device=self.device)
            
            self.running_ret = self.gamma * (1.0 - d_t) * self.running_ret + r_t
            self.max_ret_seen = torch.maximum(self.max_ret_seen, torch.abs(self.running_ret))
            
            delta = self.running_ret - self.mu
            self.n_samples += 1.0
            weight = 1.0 / self.n_samples
            
            self.mu += delta * weight
            m2 = self.sigma2 * (self.n_samples - 1.0 + 1e-4) + (delta ** 2) * (self.n_samples - 1.0) * weight
            self.sigma2 = m2 / self.n_samples

    def process_batch(self, rewards: np.ndarray, terminals: np.ndarray):
        with torch.no_grad():
            r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1)
            d_t = torch.as_tensor(terminals, dtype=torch.float32, device=self.device).view(-1)

            if self.running_ret.numel() != r_t.numel():
                self.running_ret = torch.zeros_like(r_t)

            self.running_ret = self.gamma * (1.0 - d_t) * self.running_ret + r_t
            self.max_ret_seen = torch.maximum(self.max_ret_seen, torch.max(torch.abs(self.running_ret)).view(1))

            batch = self.running_ret
            batch_count = torch.tensor(float(batch.numel()), device=self.device)
            batch_mean = batch.mean()
            batch_var = batch.var(unbiased=False) if batch.numel() > 1 else torch.zeros((), device=self.device)

            delta = batch_mean - self.mu
            total = self.n_samples + batch_count
            new_mu = self.mu + delta * batch_count / torch.clamp(total, min=1.0)
            m_a = self.sigma2 * self.n_samples
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta.pow(2) * self.n_samples * batch_count / torch.clamp(total, min=1.0)

            self.mu = new_mu.view(1)
            self.sigma2 = (m2 / torch.clamp(total, min=1.0)).view(1)
            self.n_samples = total.view(1)

    def scale(self, batch_rewards: torch.Tensor) -> torch.Tensor:
        dynamic_scale = torch.maximum(torch.sqrt(self.sigma2 + 1e-8), self.max_ret_seen / self.limit)
        return batch_rewards / dynamic_scale

class TrajectoryBuffer:
    """CPU-bound cyclic buffer for IsaacLab vectorized rollouts."""
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int = int(1e6),
        rise_context_dim: int = 0,
        rise_context_len: int = 1,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        self.capacity, self.idx, self.current_size = capacity, 0, 0
        self.rise_context_dim = rise_context_dim
        self.rise_context_len = rise_context_len
        self.n_step = max(int(n_step), 1)
        self.gamma = float(gamma)
        self.n_step_transitions = deque(maxlen=self.n_step)
        self.states = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, act_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_states = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        self.contexts = torch.zeros((capacity, rise_context_len, rise_context_dim), dtype=torch.float32)
        self.next_contexts = torch.zeros((capacity, rise_context_len, rise_context_dim), dtype=torch.float32)

    def store(self, s, a, r, ns, d, terminal=None, context=None, next_context=None):
        self.store_batch(
            np.asarray(s, dtype=np.float32).reshape(1, -1),
            np.asarray(a, dtype=np.float32).reshape(1, -1),
            np.asarray(r, dtype=np.float32).reshape(1),
            np.asarray(ns, dtype=np.float32).reshape(1, -1),
            np.asarray(d, dtype=np.float32).reshape(1),
            None if terminal is None else np.asarray(terminal, dtype=np.float32).reshape(1),
            None if context is None else np.asarray(context, dtype=np.float32).reshape(1, self.rise_context_len, -1),
            None if next_context is None else np.asarray(next_context, dtype=np.float32).reshape(1, self.rise_context_len, -1),
        )

    def store_batch(self, s, a, r, ns, d, terminal=None, context=None, next_context=None):
        transition = {
            "s": np.asarray(s, dtype=np.float32),
            "a": np.asarray(a, dtype=np.float32),
            "r": np.asarray(r, dtype=np.float32).reshape(-1),
            "ns": np.asarray(ns, dtype=np.float32),
            "d": np.asarray(d, dtype=np.float32).reshape(-1),
            "terminal": np.asarray(terminal if terminal is not None else d, dtype=np.float32).reshape(-1),
            "context": None if context is None else np.asarray(context, dtype=np.float32),
            "next_context": None if next_context is None else np.asarray(next_context, dtype=np.float32),
        }
        self.n_step_transitions.append(transition)
        if len(self.n_step_transitions) < self.n_step:
            return

        transition = self._build_n_step_transition()
        self._store_ready_batch(
            transition["s"],
            transition["a"],
            transition["r"],
            transition["ns"],
            transition["d"],
            context=transition["context"],
            next_context=transition["next_context"],
        )

    def _build_n_step_transition(self):
        first = self.n_step_transitions[0]
        current = self.n_step_transitions[-1]
        n_reward = current["r"].copy()
        n_done = current["d"].copy()
        n_next_state = current["ns"].copy()
        n_next_context = None if current["next_context"] is None else current["next_context"].copy()

        for idx in reversed(range(self.n_step - 1)):
            transition = self.n_step_transitions[idx]
            terminal = transition["terminal"].astype(bool)
            n_reward = transition["r"] + self.gamma * n_reward * (1.0 - terminal.astype(np.float32))
            if np.any(terminal):
                n_done[terminal] = transition["d"][terminal]
                n_next_state[terminal] = transition["ns"][terminal]
                if n_next_context is not None and transition["next_context"] is not None:
                    n_next_context[terminal] = transition["next_context"][terminal]

        return {
            "s": first["s"],
            "a": first["a"],
            "r": n_reward,
            "ns": n_next_state,
            "d": n_done,
            "context": first["context"],
            "next_context": n_next_context,
        }

    def _store_ready_batch(self, s, a, r, ns, d, context=None, next_context=None):
        batch_size = int(np.asarray(s).shape[0])
        indices = (torch.arange(batch_size) + self.idx) % self.capacity
        self.states[indices] = torch.as_tensor(s, dtype=torch.float32)
        self.actions[indices] = torch.as_tensor(a, dtype=torch.float32)
        self.rewards[indices] = torch.as_tensor(r, dtype=torch.float32).view(-1, 1)
        self.next_states[indices] = torch.as_tensor(ns, dtype=torch.float32)
        self.dones[indices] = torch.as_tensor(d, dtype=torch.float32).view(-1, 1)
        if self.rise_context_dim > 0:
            self.contexts[indices] = torch.as_tensor(context, dtype=torch.float32)
            self.next_contexts[indices] = torch.as_tensor(next_context, dtype=torch.float32)
        self.idx = (self.idx + batch_size) % self.capacity
        self.current_size = min(self.current_size + batch_size, self.capacity)

    def sample_batch(self, batch_size: int, device: torch.device):
        indices = torch.randint(0, self.current_size, (batch_size,))
        return (
            self.states[indices].to(device, non_blocking=True),
            self.actions[indices].to(device, non_blocking=True),
            self.rewards[indices].to(device, non_blocking=True),
            self.next_states[indices].to(device, non_blocking=True),
            self.dones[indices].to(device, non_blocking=True),
            self.contexts[indices].to(device, non_blocking=True),
            self.next_contexts[indices].to(device, non_blocking=True)
        )
        
    # [新增] Buffer 存讀機制，防範災難性遺忘
    def save(self, path: str):
        state = {
            'idx': self.idx,
            'current_size': self.current_size,
            'states': self.states[:self.current_size].clone(),
            'actions': self.actions[:self.current_size].clone(),
            'rewards': self.rewards[:self.current_size].clone(),
            'next_states': self.next_states[:self.current_size].clone(),
            'dones': self.dones[:self.current_size].clone(),
            'rise_context_dim': self.rise_context_dim,
            'rise_context_len': self.rise_context_len,
            'contexts': self.contexts[:self.current_size].clone(),
            'next_contexts': self.next_contexts[:self.current_size].clone()
        }
        torch.save(state, path)

    def load(self, path: str):
        if not os.path.isfile(path):
            print(f"[Warning] Replay buffer not found at '{path}'")
            return False
        print(f"[System] Loading replay buffer from '{path}' (Takes a few seconds)...")
        state = torch.load(path, map_location='cpu', weights_only=True)
        if self.rise_context_dim > 0 and 'contexts' not in state:
            print("[Warning] Replay buffer is incompatible with state-based RISE context. Regenerating buffer.")
            return False
        if state.get('rise_context_dim', self.rise_context_dim) != self.rise_context_dim:
            print("[Warning] Replay buffer RISE context size mismatch. Regenerating buffer.")
            return False
        if state.get('rise_context_len', self.rise_context_len) != self.rise_context_len:
            print("[Warning] Replay buffer RISE context length mismatch. Regenerating buffer.")
            return False
        sz = state['current_size']
        self.idx = state['idx']
        self.current_size = sz
        self.states[:sz] = state['states']
        self.actions[:sz] = state['actions']
        self.rewards[:sz] = state['rewards']
        self.next_states[:sz] = state['next_states']
        self.dones[:sz] = state['dones']
        if self.rise_context_dim > 0:
            self.contexts[:sz] = state['contexts']
            self.next_contexts[:sz] = state['next_contexts']
        print(f"[System] ✔️ Replay buffer loaded with {sz} transitions.")
        return True


# ---------------------------------------------------------
# Controller Agent
# ---------------------------------------------------------
class ContinuousControlAgent:
    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        
        self.v_min, self.v_max, self.n_atoms = -5.0, 5.0, 101
        self.atom_step = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms, device=device).view(1, -1)
        self.use_rise = cfg.get('use_rise', False)
        
        # Instantiate Architectures
        self.actor = ActorNetwork(cfg['s_dim'], cfg['a_dim'], rise_cfg=cfg).to(device)
        self.critic = DistributionalCritic(cfg['s_dim'] + cfg['a_dim'], n_atoms=self.n_atoms, v_range=(self.v_min, self.v_max), rise_cfg=cfg).to(device)
        
        # Initial Projection
        self._project_actor_weights()
        self._project_critic_weights()
        
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        self.target_params = list(self.target_critic.parameters())
        
        self.return_tracker = EmpiricalReturnTracker(cfg['gamma'], self.v_max, device)
        
        # Optimizers (Pure FP32)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=cfg['lr'], fused=True)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg['lr'], fused=True)
        
        # Schedulers
        total_updates = cfg.get('num_update_steps', cfg['max_steps'] - cfg['explore_steps'])
        self.sch_critic = build_lr_scheduler(self.opt_critic, cfg['lr'], cfg['min_lr'], 1, total_updates)
        self.sch_actor = build_lr_scheduler(self.opt_actor, cfg['lr'], cfg['min_lr'], 1, total_updates // 2)
        
        # Entropy Tuning
        target_entropy_sigma = 0.15
        self.heuristic_entropy = 0.5 * cfg['a_dim'] * math.log(2 * math.pi * math.e * (target_entropy_sigma ** 2))
        self.log_temp = nn.Parameter(torch.tensor([math.log(0.01)], dtype=torch.float32, device=device))
        self.opt_temp = optim.Adam([self.log_temp], lr=cfg['lr'], fused=True)
        self.sch_temp = build_lr_scheduler(self.opt_temp, cfg['lr'], cfg['min_lr'], 1, total_updates // 2)
        
        # Noise settings
        self.noise_choices, self.noise_probs = generate_zeta_distribution()
        self.repeats_left = 0
        self.active_noise = None
        self.optimization_steps = 0

        self._compile_models()

    def make_context_token(self, state: np.ndarray, prev_action: np.ndarray) -> torch.Tensor | None:
        if not self.use_rise:
            return None
        if self.cfg.get('rise_include_prev_action', True):
            token = np.concatenate([state, prev_action], axis=-1)
        else:
            token = state
        return torch.as_tensor(token, dtype=torch.float32, device=self.device)

    def make_initial_context(self, state: np.ndarray, prev_action: np.ndarray) -> torch.Tensor | None:
        if not self.use_rise:
            return None
        token = self.make_context_token(state, prev_action)
        if token.ndim == 1:
            token = token.unsqueeze(0)
        return token.unsqueeze(1).repeat(1, self.cfg['rise_context_len'], 1)

    def append_context_token(self, context: torch.Tensor | None, token: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_rise:
            return None
        if context is None:
            raise ValueError("context must be initialized before appending RISE tokens")
        if token.ndim == 1:
            token = token.unsqueeze(0)
        context = torch.roll(context, shifts=-1, dims=1)
        context[:, -1] = token
        return context

    def context_to_numpy(self, context: torch.Tensor | None) -> np.ndarray | None:
        if context is None:
            return None
        return context.detach().cpu().numpy()

    def make_empty_context(self, batch_size: int = 1) -> torch.Tensor | None:
        if not self.use_rise:
            return None
        return torch.zeros(
            (batch_size, self.cfg['rise_context_len'], self.cfg['rise_context_dim']),
            dtype=torch.float32,
            device=self.device
        )

    def _compile_models(self):
        if not self.cfg.get('use_compile', False):
            print("[System] Torch Compiler disabled (eager mode)")
            return
        try:
            if sys.platform.startswith('linux'):
                torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
                mode = "max-autotune" if torch_version >= (2, 9) else "reduce-overhead"
                self.actor = torch.compile(self.actor, mode=mode)
                self.critic = torch.compile(self.critic, mode=mode)
                self.target_critic = torch.compile(self.target_critic, mode=mode)
                print(f"[System] Torch Compiler activated ({mode}, cudagraphs disabled)")
        except Exception as e: 
            print(f"[Warning] Compiler bypass: {e}")

    def _project_actor_weights(self):
        for layer in self.actor.modules():
            if hasattr(layer, 'project_weights'): layer.project_weights()

    def _project_critic_weights(self):
        for layer in self.critic.modules():
            if hasattr(layer, 'project_weights'): layer.project_weights()

    def load_pretrained_weights(self, path: str):
        """專門用來做 Transfer Learning / Curriculum Learning 的載入函數"""
        if not os.path.isfile(path):
            print(f"[Warning] Pretrained weights not found at '{path}'. Training from scratch.")
            return False
            
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(ckpt['actor'], strict=False)
        self.critic.load_state_dict(ckpt['critic'], strict=False)
        self.target_critic.load_state_dict(ckpt['target'], strict=False)
        self.log_temp.data = torch.tensor([ckpt['log_temp']], requires_grad=True, device=self.device)
        
        trk = ckpt.get('tracker', {})
        self.return_tracker.running_ret = trk.get('run_ret', self.return_tracker.running_ret)
        self.return_tracker.max_ret_seen = trk.get('max_ret', self.return_tracker.max_ret_seen)
        self.return_tracker.mu = trk.get('mu', self.return_tracker.mu)
        self.return_tracker.sigma2 = trk.get('var', self.return_tracker.sigma2)
        self.return_tracker.n_samples = trk.get('n', self.return_tracker.n_samples)
        
        print(f"[System] ✔️ Successfully loaded pretrained weights from '{path}'")
        return True

    def get_action(self, state: np.ndarray, context: torch.Tensor | None = None, evaluate: bool = False) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            single_state = np.asarray(state).ndim == 1
            s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if s_t.ndim == 1:
                s_t = s_t.unsqueeze(0)
            if self.use_rise and context is None:
                context = self.make_empty_context(s_t.shape[0])
            mu, sigma = self.actor.get_distribution_params(s_t, context, is_training=False)
            
            if evaluate:
                action = torch.tanh(mu)
            else:
                if self.repeats_left <= 0 or self.active_noise is None or self.active_noise.shape != mu.shape:
                    self.active_noise = torch.randn_like(mu)
                    self.repeats_left = np.random.choice(self.noise_choices, p=self.noise_probs)
                self.repeats_left -= 1
                action = torch.tanh(mu + sigma * self.active_noise)
                
        self.actor.train()
        action_np = action.cpu().numpy()
        return action_np.flatten() if single_state else action_np

    def update_parameters(self, buffer: TrajectoryBuffer, batch_size: int):
        self.optimization_steps += 1
        
        s, a, raw_r, ns, d, ctx, next_ctx = buffer.sample_batch(batch_size, self.device)
        scaled_r = self.return_tracker.scale(raw_r)

        # -----------------------------------------------------
        # 1. Compute Bellman Targets
        # -----------------------------------------------------
        with torch.no_grad():
            next_a, next_logp = self.actor(ns, next_ctx, is_training=False)
            temperature = self.log_temp.exp().detach()
            entropy_penalty = temperature * next_logp
            
            joint_obs = torch.cat([s, ns], dim=0)
            joint_act = torch.cat([a, next_a], dim=0)
            joint_ctx = torch.cat([ctx, next_ctx], dim=0)
            
            target_q_preds, target_logp_preds = self.target_critic(joint_obs, joint_act, joint_ctx, is_training=True)
            
            # Isolate next state predictions
            t_q_ns = target_q_preds[:, batch_size:]
            t_logp_ns = target_logp_preds[:, batch_size:]
            
            # Conservative Q-learning mapping
            min_indices = t_q_ns.argmin(dim=0).squeeze(-1)
            selected_logp = t_logp_ns[min_indices, torch.arange(batch_size)]
            
            target_gamma = self.cfg['gamma'] ** self.cfg.get('n_step', 1)
            target_z = scaled_r + target_gamma * (1 - d) * (self.support - entropy_penalty)
            target_z = torch.clamp(target_z, self.v_min, self.v_max)
            
            projection_indices = (target_z - self.v_min) / self.atom_step
            idx_lower = projection_indices.floor().long()
            idx_upper = torch.clamp(idx_lower + 1, 0, self.n_atoms - 1)
            
            offset = projection_indices - idx_lower.float()
            probs = selected_logp.exp()
            mass_lower = probs * (1.0 - offset)
            mass_upper = probs * offset
            
            projected_dist = torch.zeros_like(selected_logp)
            projected_dist.scatter_add_(1, idx_lower, mass_lower)
            projected_dist.scatter_add_(1, idx_upper, mass_upper)

        # -----------------------------------------------------
        # 2. Critic Optimization
        # -----------------------------------------------------
        self.opt_critic.zero_grad()
        
        _, critic_logp = self.critic(joint_obs, joint_act, joint_ctx, is_training=True)
        critic_loss = -(projected_dist.unsqueeze(0) * critic_logp[:, :batch_size]).sum(dim=-1).mean()

        critic_loss.backward()
        self.opt_critic.step()
        self._project_critic_weights()
        self.sch_critic.step()

        # Polyak Averaging
        with torch.no_grad():
            torch._foreach_lerp_(self.target_params, self.critic_params, self.cfg['tau'])

        # -----------------------------------------------------
        # 3. Actor & Temperature Optimization (Delayed)
        # -----------------------------------------------------
        act_loss_track, temp_loss_track = 0.0, 0.0
        if self.optimization_steps % 2 == 0:
            
            self.opt_actor.zero_grad()
            
            joint_s = torch.cat([s, ns], dim=0)
            joint_actor_ctx = torch.cat([ctx, next_ctx], dim=0)
            new_actions, new_logp = self.actor(joint_s, joint_actor_ctx, is_training=True)
            
            new_a_curr = new_actions[:batch_size]
            logp_curr = new_logp[:batch_size]

            # Protect CUDA Graph from overwriting
            self.critic.requires_grad_(False)
            q_vals, _ = self.critic(s, new_a_curr, ctx, is_training=False)
            q_min = torch.min(q_vals[0], q_vals[1]).unsqueeze(1)
            self.critic.requires_grad_(True)
            
            actor_loss = (self.log_temp.exp().detach() * logp_curr - q_min).mean()

            actor_loss.backward()
            self.opt_actor.step()

            self.opt_temp.zero_grad()
            temp_loss = self.log_temp.exp() * (-logp_curr.detach() - self.heuristic_entropy).mean()
                
            temp_loss.backward()
            self.opt_temp.step()

            self._project_actor_weights()
            self.sch_actor.step()
            self.sch_temp.step()
            
            act_loss_track = actor_loss.item()
            temp_loss_track = temp_loss.item()

        return critic_loss.item(), act_loss_track, temp_loss_track

    # ---------------------------------------------------------
    # Persistency
    # ---------------------------------------------------------
    def checkpoint(self, path, step, ep, b_score):
        state = {
            'step': step, 'ep': ep, 'best_score': b_score, 'opt_steps': self.optimization_steps,
            'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
            'target': self.target_critic.state_dict(), 'log_temp': self.log_temp.item(),
            'rise_context_dim': self.cfg.get('rise_context_dim', 0),
            'rise_context_len': self.cfg.get('rise_context_len', 0),
            'opt_a': self.opt_actor.state_dict(), 'opt_c': self.opt_critic.state_dict(), 'opt_t': self.opt_temp.state_dict(),
            'sch_a': self.sch_actor.state_dict(), 'sch_c': self.sch_critic.state_dict(), 'sch_t': self.sch_temp.state_dict(),
            'tracker': {
                'run_ret': self.return_tracker.running_ret, 'max_ret': self.return_tracker.max_ret_seen,
                'mu': self.return_tracker.mu, 'var': self.return_tracker.sigma2, 'n': self.return_tracker.n_samples
            }
        }
        torch.save(state, path)

    def restore(self, path):
        if not os.path.isfile(path): return 0, 0, -float('inf')
        
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if self.use_rise and (
            ckpt.get('rise_context_dim') != self.cfg.get('rise_context_dim')
            or ckpt.get('rise_context_len') != self.cfg.get('rise_context_len')
        ):
            print("[Warning] Checkpoint is incompatible with state-based RISE. Starting fresh.")
            return 0, 0, -float('inf')

        try:
            self.actor.load_state_dict(ckpt['actor'])
            self.critic.load_state_dict(ckpt['critic'])
            self.target_critic.load_state_dict(ckpt['target'])
            self.opt_actor.load_state_dict(ckpt['opt_a'])
            self.opt_critic.load_state_dict(ckpt['opt_c'])
            self.opt_temp.load_state_dict(ckpt['opt_t'])
        except RuntimeError as e:
            print(f"[Warning] Checkpoint restore failed ({e}). Starting fresh.")
            return 0, 0, -float('inf')
        self.log_temp.data = torch.tensor([ckpt['log_temp']], requires_grad=True, device=self.device)
        
        self.sch_actor.load_state_dict(ckpt.get('sch_a', self.sch_actor.state_dict()))
        self.sch_critic.load_state_dict(ckpt.get('sch_c', self.sch_critic.state_dict()))
        self.sch_temp.load_state_dict(ckpt.get('sch_t', self.sch_temp.state_dict()))
        
        self.optimization_steps = ckpt.get('opt_steps', 0)
        trk = ckpt.get('tracker', {})
        self.return_tracker.running_ret = trk.get('run_ret', self.return_tracker.running_ret)
        self.return_tracker.max_ret_seen = trk.get('max_ret', self.return_tracker.max_ret_seen)
        self.return_tracker.mu = trk.get('mu', self.return_tracker.mu)
        self.return_tracker.sigma2 = trk.get('var', self.return_tracker.sigma2)
        self.return_tracker.n_samples = trk.get('n', self.return_tracker.n_samples)
        
        print(f"[System] Resumed from step {ckpt['step']} (Record: {ckpt['best_score']:.2f})")
        return ckpt['step'], ckpt['ep'], ckpt['best_score']


# ---------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------
def render_metrics(csv_path: str, out_img: str):
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if data.ndim == 1: data = data.reshape(1, -1)
    if len(data) == 0: return

    steps, means, stds, scores = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, means, color='#1f77b4', linewidth=2, label='Mean Return')
    ax.fill_between(steps, means - stds, means + stds, color='#1f77b4', alpha=0.25)
    ax.plot(steps, scores, color='#d62728', linestyle='--', linewidth=2, label='Evaluation Score')
    ax.axhline(450, color='#2ca02c', linestyle=':', linewidth=2, label='Success Threshold')
    
    ax.set_title('Agent Learning Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_img, dpi=200)
    plt.close(fig)

class TensorboardScalarLogger:
    def __init__(self, log_dir: str, cfg: dict):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            self.writer = None
            print("[TensorBoard] tensorboard is not installed; scalar logging is disabled.")
            print("[TensorBoard] Install it in the active env to enable logs: pip install tensorboard")
            return

        self.writer = SummaryWriter(log_dir=log_dir)
        cfg_text = "\n".join(f"{key}: {value}" for key, value in sorted(cfg.items()))
        self.writer.add_text("config", f"```text\n{cfg_text}\n```", global_step=0)

    def log_scalars(self, step: int, scalars: dict[str, float | int]):
        if self.writer is None:
            return
        for key, value in scalars.items():
            if value is None:
                continue
            self.writer.add_scalar(key, float(value), global_step=step)
        self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()

def execute_evaluation(env, controller, n_eps=50):
    num_envs = env.num_envs
    rounds = max(math.ceil(n_eps / num_envs), 1)
    returns = []
    for _ in range(rounds):
        s, _ = env.reset(random_start_init=False)
        prev_action = np.zeros((num_envs, controller.cfg['a_dim']), dtype=np.float32)
        rise_context = controller.make_initial_context(s, prev_action)
        ep_ret = np.zeros(num_envs, dtype=np.float32)
        terminal = np.zeros(num_envs, dtype=bool)
        while not np.all(terminal):
            token = controller.make_context_token(s, prev_action)
            rise_context = controller.append_context_token(rise_context, token)
            a = controller.get_action(s, context=rise_context, evaluate=True)
            s, r, d, t, _ = env.step(a)
            prev_action = a.astype(np.float32)
            active = ~terminal
            ep_ret += r * active
            terminal = np.logical_or(terminal, np.logical_or(d, t))
        returns.extend(ep_ret.tolist())
    returns = returns[:n_eps]
    avg, dev = np.mean(returns), np.std(returns)
    return avg, dev, avg - dev

def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def parse_args():
    parser = argparse.ArgumentParser(description="Train STAM-FlashSAC or the FlashSAC baseline.")
    parser.add_argument("--env-id", type=str, default="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0")
    parser.add_argument("--sim-device", type=str, default=None, help='IsaacLab/PyTorch device, e.g. "cuda:2".')
    parser.add_argument("--variant-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=50_000_896)
    parser.add_argument("--num-train-envs", type=int, default=1024)
    parser.add_argument("--explore-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--updates-per-interaction-step", type=int, default=2)
    parser.add_argument("--eval-freq", type=int, default=5_000_000)
    parser.add_argument("--logging-freq", type=int, default=100_000)
    parser.add_argument("--num-eval-episodes", type=int, default=1024)
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--save-buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", dest="use_compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-rise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rise-context-len", type=int, default=8)
    parser.add_argument("--rise-hidden-dim", type=int, default=128)
    parser.add_argument("--rise-include-prev-action", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def infer_variant_name(use_rise: bool, context_len: int, include_prev_action: bool) -> str:
    if not use_rise:
        return "flashsac_baseline"
    suffix = "obs_action" if include_prev_action else "obs_only"
    return f"memory_K{context_len}_{suffix}"


def run_experiment():
    args = parse_args()
    seed = np.random.randint(100000) if args.seed is None else args.seed
    variant_name = args.variant_name or infer_variant_name(
        args.use_rise, args.rise_context_len, args.rise_include_prev_action
    )
    cfg = {
        'env_id': args.env_id,
        'sim_device': args.sim_device,
        'variant_name': variant_name,
        'pretrained_path': args.pretrained_path,
        'seed': seed,
        # FlashSAC official GPU-sim scale is 50_000_896 env steps, 1024 envs,
        # 100k warmup transitions, batch 2048, and UTD=2.
        'max_steps': args.max_steps,
        'num_train_envs': args.num_train_envs,
        'explore_steps': args.explore_steps,
        'batch_size': args.batch_size,
        'updates_per_interaction_step': args.updates_per_interaction_step,
        'eval_freq': args.eval_freq,
        'logging_freq': args.logging_freq,
        'num_eval_episodes': args.num_eval_episodes,
        'buffer_capacity': args.buffer_capacity,
        'save_buffer': args.save_buffer,
        'gamma': 0.99,
        'n_step': args.n_step,
        'tau': 0.01,
        'lr': 3e-4,
        'min_lr': 1.5e-4,
        'use_rise': args.use_rise,
        'rise_context_len': args.rise_context_len,
        'rise_hidden_dim': args.rise_hidden_dim,
        'rise_include_prev_action': args.rise_include_prev_action,
        'use_compile': args.use_compile
    }
    cfg['num_interaction_steps'] = math.ceil(cfg['max_steps'] / cfg['num_train_envs'])
    cfg['num_update_steps'] = cfg['num_interaction_steps'] * cfg['updates_per_interaction_step']

    output_root = args.output_root or os.path.join(repo_dir, "checkpoints")
    run_name = f"{sanitize_name(cfg['env_id'])}_{sanitize_name(variant_name)}_seed{cfg['seed']}"
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    sim_device = args.sim_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if sim_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(sim_device.split(":")[-1]) if ":" in sim_device else 0)
    device = torch.device(sim_device if torch.cuda.is_available() or sim_device == "cpu" else "cpu")
    print(f"--- Initialization [Device: {device}] ---")
    print(f"--- Run: {run_name} ---")
    
    train_env = make_isaaclab_env(
        env_name=cfg['env_id'],
        num_envs=cfg['num_train_envs'],
        seed=cfg['seed'],
        headless=True,
        device=sim_device,
    )
    test_env = train_env
    
    cfg['s_dim'] = train_env.single_observation_space.shape[0]
    cfg['a_dim'] = train_env.single_action_space.shape[0]
    cfg['rise_context_dim'] = cfg['s_dim'] + cfg['a_dim'] if cfg.get('rise_include_prev_action', True) else cfg['s_dim']
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    
    controller = ContinuousControlAgent(cfg, device)
    memory = TrajectoryBuffer(
        cfg['s_dim'],
        cfg['a_dim'],
        capacity=cfg['buffer_capacity'],
        rise_context_dim=cfg['rise_context_dim'] if cfg.get('use_rise', False) else 0,
        rise_context_len=cfg['rise_context_len'],
        n_step=cfg['n_step'],
        gamma=cfg['gamma'],
    )
    
    tb_log_dir = os.path.join(output_dir, "tensorboard")
    log_file = os.path.join(output_dir, "training_log.csv")
    best_ckpt = os.path.join(output_dir, "controller_best.pt")
    latest_ckpt = os.path.join(output_dir, "controller_latest.pt")
    best_buffer = os.path.join(output_dir, "buffer_best.pt")
    latest_buffer = os.path.join(output_dir, "buffer_latest.pt")
    tb_logger = TensorboardScalarLogger(tb_log_dir, cfg)
    print(f"[TensorBoard] Logging to: {tb_log_dir}")
    
    current_step, ep_idx, best_metric = controller.restore(latest_ckpt)
    using_pretrained = False
    
    if current_step == 0:
        with open(log_file, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestep", "AvgReturn", "StdDev", "Score"])
        print(f"--- Started Task: {cfg['env_id']} ({cfg['num_train_envs']} envs) ---")
        
        if cfg.get('pretrained_path'):
            using_pretrained = controller.load_pretrained_weights(cfg['pretrained_path'])
            if using_pretrained:
                # 若為 Fine-tuning，動態調降學習率，並確保有足夠的 explore 暖身期
                cfg['lr'] = 5e-5
                cfg['min_lr'] = 1e-5
    else:
        print(f"\n[INFO] Resuming Training from Step {current_step}...")
        if cfg['save_buffer'] and not memory.load(latest_buffer):
            current_step, ep_idx, best_metric = 0, 0, -float('inf')
            with open(log_file, 'w', newline='') as f:
                csv.writer(f).writerow(["Timestep", "AvgReturn", "StdDev", "Score"])
            print("[INFO] Restarting training because saved replay data is not usable with RISE.")
        
    obs, _ = train_env.reset()
    prev_action = np.zeros((cfg['num_train_envs'], cfg['a_dim']), dtype=np.float32)
    rise_context = controller.make_initial_context(obs, prev_action)
    ep_reward = np.zeros(cfg['num_train_envs'], dtype=np.float32)
    loss_c, loss_a, loss_t = 0.0, 0.0, 0.0
    grad_updates = 0
    max_interaction_steps = math.ceil(cfg['max_steps'] / cfg['num_train_envs'])
    start_interaction_step = current_step // cfg['num_train_envs']
    next_eval_step = max(((current_step // cfg['eval_freq']) + 1) * cfg['eval_freq'], cfg['eval_freq'])
    next_log_step = max(((current_step // cfg['logging_freq']) + 1) * cfg['logging_freq'], cfg['logging_freq'])
    
    for interaction_step in range(start_interaction_step, max_interaction_steps):
        env_step = (interaction_step + 1) * cfg['num_train_envs']
        current_token = controller.make_context_token(obs, prev_action)
        rise_context = controller.append_context_token(rise_context, current_token)
        stored_context = controller.context_to_numpy(rise_context)
        
        if memory.current_size < cfg['explore_steps']:
            if using_pretrained or current_step > 0:
                action = controller.get_action(obs, context=rise_context)
            else:
                action = train_env.action_space.sample()
        else:
            action = controller.get_action(obs, context=rise_context)
            
        next_obs, reward, done, trunc, infos = train_env.step(action)
        terminal = np.logical_or(done, trunc)
        next_buffer_obs = next_obs.copy()
        if isinstance(infos, dict) and "final_obs" in infos:
            next_buffer_obs[terminal] = infos["final_obs"][terminal]

        next_token = controller.make_context_token(next_obs, action.astype(np.float32))
        next_buffer_token = controller.make_context_token(next_buffer_obs, action.astype(np.float32))
        if cfg.get('use_rise', False):
            next_context = controller.append_context_token(rise_context.clone(), next_token)
            stored_next_context = controller.context_to_numpy(
                controller.append_context_token(rise_context.clone(), next_buffer_token)
            )
        else:
            next_context = None
            stored_next_context = None
        controller.return_tracker.process_batch(reward, terminal)
        memory.store_batch(
            obs, action, reward, next_buffer_obs, done.astype(np.float32),
            terminal=terminal.astype(np.float32),
            context=stored_context,
            next_context=stored_next_context
        )
        obs = next_obs
        prev_action = action.astype(np.float32)
        rise_context = next_context
        ep_reward += reward

        if memory.current_size >= cfg['explore_steps']:
            for _ in range(cfg['updates_per_interaction_step']):
                c_val, a_val, t_val = controller.update_parameters(memory, cfg['batch_size'])
                loss_c += c_val
                loss_a += a_val
                loss_t += t_val
                grad_updates += 1

        if np.any(terminal):
            controller.repeats_left = 0 
            finished_returns = ep_reward[terminal]
            ep_idx += int(np.sum(terminal))
            ep_reward[terminal] = 0.0
            prev_action[terminal] = 0.0
            if rise_context is not None:
                # IsaacLab vector envs usually auto-reset done sub-envs before returning next_obs.
                # Reset their history so a new episode does not inherit stale obs/action context.
                rise_context[terminal] = controller.make_initial_context(obs[terminal], prev_action[terminal])

            if grad_updates > 0 and (interaction_step + 1) % 25 == 0:
                act_iters = max(grad_updates // 2, 1)
                avg_critic_loss = loss_c / grad_updates
                avg_actor_loss = loss_a / act_iters
                avg_temp_loss = loss_t / act_iters
                print(f"Step: {env_step:09d} | Ep: {ep_idx:6d} | Ret: {np.mean(finished_returns):8.2f} | "
                      f"LC: {avg_critic_loss:7.3f} | LA: {avg_actor_loss:7.3f} | LT: {avg_temp_loss:7.3f}")
                tb_logger.log_scalars(env_step, {
                    "episode/return": float(np.mean(finished_returns)),
                    "episode/num_finished": int(np.sum(terminal)),
                    "train/critic_loss": avg_critic_loss,
                    "train/actor_loss": avg_actor_loss,
                    "train/temp_loss": avg_temp_loss,
                    "train/temperature": float(controller.log_temp.exp().detach().cpu().item()),
                    "train/buffer_size": memory.current_size,
                    "train/updates": controller.optimization_steps,
                    "train/actor_lr": controller.sch_actor.get_last_lr()[0],
                    "train/critic_lr": controller.sch_critic.get_last_lr()[0],
                })
                loss_c, loss_a, loss_t, grad_updates = 0.0, 0.0, 0.0, 0

        if env_step >= next_log_step:
            tb_logger.log_scalars(env_step, {
                "train/buffer_size": memory.current_size,
                "train/updates": controller.optimization_steps,
                "train/temperature": float(controller.log_temp.exp().detach().cpu().item()),
                "train/reward_scale_sigma": float(torch.sqrt(controller.return_tracker.sigma2 + 1e-8).detach().cpu().item()),
                "train/reward_scale_max_return": float(controller.return_tracker.max_ret_seen.detach().cpu().item()),
                "train/actor_lr": controller.sch_actor.get_last_lr()[0],
                "train/critic_lr": controller.sch_critic.get_last_lr()[0],
            })
            next_log_step += cfg['logging_freq']

        # Benchmarking & Saving
        if env_step >= next_eval_step:
            print(f"\n=> Running Validation Sequence (Step {env_step})")
            val_mean, val_std, val_score = execute_evaluation(test_env, controller, cfg['num_eval_episodes'])
            print(f"   Score: {val_score:8.2f} (µ: {val_mean:8.2f}, σ: {val_std:6.2f})")
            tb_logger.log_scalars(env_step, {
                "eval/avg_return": val_mean,
                "eval/std_return": val_std,
                "eval/score": val_score,
                "eval/best_score": max(best_metric, val_score),
            })
            
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([env_step, val_mean, val_std, val_score])
            render_metrics(log_file, os.path.join(output_dir, "performance_chart.png"))
            
            if val_score > best_metric:
                best_metric = val_score
                controller.checkpoint(best_ckpt, env_step, ep_idx, best_metric)
                if cfg['save_buffer']:
                    memory.save(best_buffer)
                print(f"   [+] Validation improved. Model & Buffer archived.")
            
            controller.checkpoint(latest_ckpt, env_step, ep_idx, best_metric)
            if cfg['save_buffer']:
                memory.save(latest_buffer)
            next_eval_step += cfg['eval_freq']
            print("-" * 60 + "\n")

    train_env.close()
    tb_logger.close()

if __name__ == "__main__":
    run_experiment()
