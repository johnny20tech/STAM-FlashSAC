import os
from collections import deque

import numpy as np
import torch


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
        stam_context_dim: int = 0,
        stam_context_len: int = 1,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        self.capacity, self.idx, self.current_size = capacity, 0, 0
        self.stam_context_dim = stam_context_dim
        self.stam_context_len = stam_context_len
        self.n_step = max(int(n_step), 1)
        self.gamma = float(gamma)
        self.n_step_transitions = deque(maxlen=self.n_step)
        self.states = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, act_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_states = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        self.contexts = torch.zeros((capacity, stam_context_len, stam_context_dim), dtype=torch.float32)
        self.next_contexts = torch.zeros((capacity, stam_context_len, stam_context_dim), dtype=torch.float32)

    def store(self, s, a, r, ns, d, terminal=None, context=None, next_context=None):
        self.store_batch(
            np.asarray(s, dtype=np.float32).reshape(1, -1),
            np.asarray(a, dtype=np.float32).reshape(1, -1),
            np.asarray(r, dtype=np.float32).reshape(1),
            np.asarray(ns, dtype=np.float32).reshape(1, -1),
            np.asarray(d, dtype=np.float32).reshape(1),
            None if terminal is None else np.asarray(terminal, dtype=np.float32).reshape(1),
            None if context is None else np.asarray(context, dtype=np.float32).reshape(1, self.stam_context_len, -1),
            None if next_context is None else np.asarray(next_context, dtype=np.float32).reshape(1, self.stam_context_len, -1),
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
        if self.stam_context_dim > 0:
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
            'stam_context_dim': self.stam_context_dim,
            'stam_context_len': self.stam_context_len,
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
        if self.stam_context_dim > 0 and 'contexts' not in state:
            print("[Warning] Replay buffer is incompatible with state-based STAM context. Regenerating buffer.")
            return False
        saved_context_dim = state.get('stam_context_dim', state.get('rise_context_dim', self.stam_context_dim))
        saved_context_len = state.get('stam_context_len', state.get('rise_context_len', self.stam_context_len))
        if saved_context_dim != self.stam_context_dim:
            print("[Warning] Replay buffer STAM context size mismatch. Regenerating buffer.")
            return False
        if saved_context_len != self.stam_context_len:
            print("[Warning] Replay buffer STAM context length mismatch. Regenerating buffer.")
            return False
        sz = state['current_size']
        self.idx = state['idx']
        self.current_size = sz
        self.states[:sz] = state['states']
        self.actions[:sz] = state['actions']
        self.rewards[:sz] = state['rewards']
        self.next_states[:sz] = state['next_states']
        self.dones[:sz] = state['dones']
        if self.stam_context_dim > 0:
            self.contexts[:sz] = state['contexts']
            self.next_contexts[:sz] = state['next_contexts']
        print(f"[System] ✔️ Replay buffer loaded with {sz} transitions.")
        return True

