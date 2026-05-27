import csv
import math

import matplotlib.pyplot as plt
import numpy as np


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
        stam_context = controller.make_initial_context(s, prev_action)
        ep_ret = np.zeros(num_envs, dtype=np.float32)
        terminal = np.zeros(num_envs, dtype=bool)
        while not np.all(terminal):
            token = controller.make_context_token(s, prev_action)
            stam_context = controller.append_context_token(stam_context, token)
            a = controller.get_action(s, context=stam_context, evaluate=True)
            s, r, d, t, _ = env.step(a)
            prev_action = a.astype(np.float32)
            active = ~terminal
            ep_ret += r * active
            terminal = np.logical_or(terminal, np.logical_or(d, t))
        returns.extend(ep_ret.tolist())
    returns = returns[:n_eps]
    avg, dev = np.mean(returns), np.std(returns)
    return avg, dev, avg - dev

