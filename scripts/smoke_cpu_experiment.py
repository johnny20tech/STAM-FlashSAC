import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stam_flashsac.flashsac.agent import ContinuousControlAgent
from stam_flashsac.flashsac.buffers import TrajectoryBuffer


def make_cfg(use_stam: bool) -> dict:
    state_dim = 6
    action_dim = 3
    context_len = 4
    return {
        "s_dim": state_dim,
        "a_dim": action_dim,
        "gamma": 0.99,
        "n_step": 3,
        "tau": 0.01,
        "lr": 3e-4,
        "min_lr": 1.5e-4,
        "max_steps": 512,
        "explore_steps": 0,
        "num_interaction_steps": 16,
        "updates_per_interaction_step": 2,
        "num_update_steps": 32,
        "use_stam": use_stam,
        "stam_context_len": context_len,
        "stam_hidden_dim": 16,
        "stam_include_prev_action": True,
        "stam_context_dim": state_dim + action_dim,
        "use_compile": False,
    }


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains non-finite values")


def smoke_agent(use_stam: bool) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cpu")
    cfg = make_cfg(use_stam)
    agent = ContinuousControlAgent(cfg, device)

    batch_size = 8
    states = rng.normal(size=(batch_size, cfg["s_dim"])).astype(np.float32)
    actions = rng.uniform(-0.5, 0.5, size=(batch_size, cfg["a_dim"])).astype(np.float32)
    context = None
    if use_stam:
        context = torch.randn(batch_size, cfg["stam_context_len"], cfg["stam_context_dim"], device=device)

    sampled_action = agent.get_action(states, context=context, evaluate=False)
    if sampled_action.shape != (batch_size, cfg["a_dim"]):
        raise AssertionError(f"unexpected action shape: {sampled_action.shape}")

    state_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    action_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    actor_action, log_prob = agent.actor(state_t, context, is_training=True)
    critic_q, critic_log_probs = agent.critic(state_t, action_t, context, is_training=True)
    if actor_action.shape != (batch_size, cfg["a_dim"]):
        raise AssertionError(f"unexpected actor output shape: {actor_action.shape}")
    if log_prob.shape != (batch_size, 1):
        raise AssertionError(f"unexpected log_prob shape: {log_prob.shape}")
    if critic_q.shape != (2, batch_size):
        raise AssertionError(f"unexpected critic q shape: {critic_q.shape}")
    if critic_log_probs.shape != (2, batch_size, agent.n_atoms):
        raise AssertionError(f"unexpected critic distribution shape: {critic_log_probs.shape}")
    assert_finite("actor_action", actor_action)
    assert_finite("log_prob", log_prob)
    assert_finite("critic_q", critic_q)
    assert_finite("critic_log_probs", critic_log_probs)

    buffer = TrajectoryBuffer(
        cfg["s_dim"],
        cfg["a_dim"],
        capacity=64,
        stam_context_dim=cfg["stam_context_dim"] if use_stam else 0,
        stam_context_len=cfg["stam_context_len"],
        n_step=cfg["n_step"],
        gamma=cfg["gamma"],
    )
    for _ in range(12):
        s = rng.normal(size=(batch_size, cfg["s_dim"])).astype(np.float32)
        a = rng.uniform(-0.5, 0.5, size=(batch_size, cfg["a_dim"])).astype(np.float32)
        r = rng.normal(size=(batch_size,)).astype(np.float32)
        ns = rng.normal(size=(batch_size, cfg["s_dim"])).astype(np.float32)
        done = rng.random(batch_size) < 0.1
        ctx = None
        next_ctx = None
        if use_stam:
            ctx = rng.normal(size=(batch_size, cfg["stam_context_len"], cfg["stam_context_dim"])).astype(np.float32)
            next_ctx = rng.normal(size=(batch_size, cfg["stam_context_len"], cfg["stam_context_dim"])).astype(np.float32)
        agent.return_tracker.process_batch(r, done)
        buffer.store_batch(s, a, r, ns, done.astype(np.float32), terminal=done.astype(np.float32), context=ctx, next_context=next_ctx)

    if buffer.current_size < batch_size:
        raise AssertionError(f"buffer did not collect enough transitions: {buffer.current_size}")

    losses = [agent.update_parameters(buffer, batch_size) for _ in range(2)]
    for loss_tuple in losses:
        for value in loss_tuple:
            if not math.isfinite(float(value)):
                raise AssertionError(f"non-finite loss: {loss_tuple}")


def main() -> int:
    smoke_agent(use_stam=False)
    smoke_agent(use_stam=True)
    print("CPU architecture smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
