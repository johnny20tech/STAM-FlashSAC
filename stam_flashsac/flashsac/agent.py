import copy
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stam_flashsac.flashsac.buffers import EmpiricalReturnTracker, TrajectoryBuffer
from stam_flashsac.flashsac.models import ActorNetwork, DistributionalCritic
from stam_flashsac.flashsac.utils import build_lr_scheduler, generate_zeta_distribution


class ContinuousControlAgent:
    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        
        self.v_min, self.v_max, self.n_atoms = -5.0, 5.0, 101
        self.atom_step = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms, device=device).view(1, -1)
        self.use_stam = cfg.get('use_stam', False)
        
        # Instantiate Architectures
        self.actor = ActorNetwork(cfg['s_dim'], cfg['a_dim'], stam_cfg=cfg).to(device)
        self.critic = DistributionalCritic(cfg['s_dim'] + cfg['a_dim'], n_atoms=self.n_atoms, v_range=(self.v_min, self.v_max), stam_cfg=cfg).to(device)
        
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
        if not self.use_stam:
            return None
        if self.cfg.get('stam_include_prev_action', True):
            if self.cfg.get('stam_action_mode', 'correct') == 'shuffled':
                prev_action = prev_action[np.random.permutation(prev_action.shape[0])]
            token = np.concatenate([state, prev_action], axis=-1)
        else:
            token = state
        return torch.as_tensor(token, dtype=torch.float32, device=self.device)

    def make_initial_context(self, state: np.ndarray, prev_action: np.ndarray) -> torch.Tensor | None:
        if not self.use_stam:
            return None
        token = self.make_context_token(state, prev_action)
        if token.ndim == 1:
            token = token.unsqueeze(0)
        return token.unsqueeze(1).repeat(1, self.cfg['stam_context_len'], 1)

    def append_context_token(self, context: torch.Tensor | None, token: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_stam:
            return None
        if context is None:
            raise ValueError("context must be initialized before appending STAM tokens")
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
        if not self.use_stam:
            return None
        return torch.zeros(
            (batch_size, self.cfg['stam_context_len'], self.cfg['stam_context_dim']),
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
            if self.use_stam and context is None:
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
            'stam_context_dim': self.cfg.get('stam_context_dim', 0),
            'stam_context_len': self.cfg.get('stam_context_len', 0),
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

        ckpt_context_dim = ckpt.get('stam_context_dim', ckpt.get('rise_context_dim'))
        ckpt_context_len = ckpt.get('stam_context_len', ckpt.get('rise_context_len'))
        if self.use_stam and (
            ckpt_context_dim != self.cfg.get('stam_context_dim')
            or ckpt_context_len != self.cfg.get('stam_context_len')
        ):
            print("[Warning] Checkpoint is incompatible with state-based STAM. Starting fresh.")
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
