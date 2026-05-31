import argparse
import csv
import json
import math
import os

import numpy as np
import torch

from stam_flashsac.flashsac.runtime import configure_runtime, configure_torch_runtime


repo_dir, runtime_dir = configure_runtime()
configure_torch_runtime(torch)


from stam_flashsac.flashsac.agent import ContinuousControlAgent
from stam_flashsac.flashsac.buffers import TrajectoryBuffer
from stam_flashsac.flashsac.metrics import TensorboardScalarLogger, execute_evaluation, render_metrics
from stam_flashsac.flashsac.utils import sanitize_name
from stam_flashsac.vendor.flash_rl.envs.isaaclab import make_isaaclab_env


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
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="STAM-FlashSAC")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--save-buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", dest="use_compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-stam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-rise", dest="use_stam", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-use-rise", dest="use_stam", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--stam-context-len", "--rise-context-len", dest="stam_context_len", type=int, default=8)
    parser.add_argument("--stam-hidden-dim", "--rise-hidden-dim", dest="stam_hidden_dim", type=int, default=128)
    parser.add_argument("--stam-include-prev-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rise-include-prev-action", dest="stam_include_prev_action", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-rise-include-prev-action", dest="stam_include_prev_action", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--stam-fusion", choices=["gate", "concat"], default="gate")
    parser.add_argument("--stam-encoder", choices=["lstm", "gru", "mlp"], default="lstm")
    parser.add_argument("--stam-action-mode", choices=["correct", "shuffled"], default="correct")
    parser.add_argument("--stam-apply-to", choices=["both", "actor", "critic"], default="both")
    return parser.parse_args()


def infer_variant_name(use_stam: bool, context_len: int, include_prev_action: bool) -> str:
    if not use_stam:
        return "flashsac_baseline"
    suffix = "obs_action" if include_prev_action else "obs_only"
    return f"memory_K{context_len}_{suffix}"


def reset_training_rollout_state(train_env, controller, cfg):
    obs, _ = train_env.reset()
    prev_action = np.zeros((cfg['num_train_envs'], cfg['a_dim']), dtype=np.float32)
    stam_context = controller.make_initial_context(obs, prev_action)
    ep_reward = np.zeros(cfg['num_train_envs'], dtype=np.float32)
    controller.repeats_left = 0
    controller.active_noise = None
    return obs, prev_action, stam_context, ep_reward


def run_experiment():
    args = parse_args()
    seed = np.random.randint(100000) if args.seed is None else args.seed
    variant_name = args.variant_name or infer_variant_name(
        args.use_stam, args.stam_context_len, args.stam_include_prev_action
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
        'use_stam': args.use_stam,
        'stam_context_len': args.stam_context_len,
        'stam_hidden_dim': args.stam_hidden_dim,
        'stam_include_prev_action': args.stam_include_prev_action,
        'stam_fusion': args.stam_fusion,
        'stam_encoder': args.stam_encoder,
        'stam_action_mode': args.stam_action_mode,
        'stam_apply_to': args.stam_apply_to,
        'wandb': args.wandb,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'wandb_mode': args.wandb_mode,
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
    cfg['stam_context_dim'] = cfg['s_dim'] + cfg['a_dim'] if cfg.get('stam_include_prev_action', True) else cfg['s_dim']
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    
    controller = ContinuousControlAgent(cfg, device)
    memory = TrajectoryBuffer(
        cfg['s_dim'],
        cfg['a_dim'],
        capacity=cfg['buffer_capacity'],
        stam_context_dim=cfg['stam_context_dim'] if cfg.get('use_stam', False) else 0,
        stam_context_len=cfg['stam_context_len'],
        n_step=cfg['n_step'],
        gamma=cfg['gamma'],
    )
    
    tb_log_dir = os.path.join(output_dir, "tensorboard")
    log_file = os.path.join(output_dir, "training_log.csv")
    best_ckpt = os.path.join(output_dir, "controller_best.pt")
    latest_ckpt = os.path.join(output_dir, "controller_latest.pt")
    best_buffer = os.path.join(output_dir, "buffer_best.pt")
    latest_buffer = os.path.join(output_dir, "buffer_latest.pt")
    tb_logger = TensorboardScalarLogger(
        tb_log_dir,
        cfg,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        run_name=run_name,
        output_dir=output_dir,
    )
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
            print("[INFO] Restarting training because saved replay data is not usable with STAM.")
        
    obs, prev_action, stam_context, ep_reward = reset_training_rollout_state(train_env, controller, cfg)
    loss_c, loss_a, loss_t = 0.0, 0.0, 0.0
    grad_updates = 0
    log_return_sum = 0.0
    log_finished_episodes = 0
    max_interaction_steps = math.ceil(cfg['max_steps'] / cfg['num_train_envs'])
    start_interaction_step = current_step // cfg['num_train_envs']
    next_eval_step = max(((current_step // cfg['eval_freq']) + 1) * cfg['eval_freq'], cfg['eval_freq'])
    next_log_step = max(((current_step // cfg['logging_freq']) + 1) * cfg['logging_freq'], cfg['logging_freq'])
    
    for interaction_step in range(start_interaction_step, max_interaction_steps):
        env_step = (interaction_step + 1) * cfg['num_train_envs']
        current_token = controller.make_context_token(obs, prev_action)
        stam_context = controller.append_context_token(stam_context, current_token)
        stored_context = controller.context_to_numpy(stam_context)
        
        if memory.current_size < cfg['explore_steps']:
            if using_pretrained or current_step > 0:
                action = controller.get_action(obs, context=stam_context)
            else:
                action = train_env.action_space.sample()
        else:
            action = controller.get_action(obs, context=stam_context)
            
        next_obs, reward, done, trunc, infos = train_env.step(action)
        terminal = np.logical_or(done, trunc)
        next_buffer_obs = next_obs.copy()
        if isinstance(infos, dict) and "final_obs" in infos:
            next_buffer_obs[terminal] = infos["final_obs"][terminal]

        next_token = controller.make_context_token(next_obs, action.astype(np.float32))
        next_buffer_token = controller.make_context_token(next_buffer_obs, action.astype(np.float32))
        if cfg.get('use_stam', False):
            next_context = controller.append_context_token(stam_context.clone(), next_token)
            stored_next_context = controller.context_to_numpy(
                controller.append_context_token(stam_context.clone(), next_buffer_token)
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
        stam_context = next_context
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
            log_return_sum += float(np.sum(finished_returns))
            log_finished_episodes += int(np.sum(terminal))
            ep_idx += int(np.sum(terminal))
            ep_reward[terminal] = 0.0
            prev_action[terminal] = 0.0
            if stam_context is not None:
                # IsaacLab vector envs usually auto-reset done sub-envs before returning next_obs.
                # Reset their history so a new episode does not inherit stale obs/action context.
                stam_context[terminal] = controller.make_initial_context(obs[terminal], prev_action[terminal])

        if env_step >= next_log_step:
            avg_return = log_return_sum / log_finished_episodes if log_finished_episodes > 0 else float("nan")
            avg_critic_loss = loss_c / grad_updates if grad_updates > 0 else float("nan")
            act_iters = max(grad_updates // 2, 1)
            avg_actor_loss = loss_a / act_iters if grad_updates > 0 else float("nan")
            avg_temp_loss = loss_t / act_iters if grad_updates > 0 else float("nan")
            print(f"Step: {env_step:09d} | Ep: {ep_idx:6d} | Ret: {avg_return:8.2f} | "
                  f"LC: {avg_critic_loss:7.3f} | LA: {avg_actor_loss:7.3f} | LT: {avg_temp_loss:7.3f}")

            scalars = {
                "train/buffer_size": memory.current_size,
                "train/updates": controller.optimization_steps,
                "train/temperature": float(controller.log_temp.exp().detach().cpu().item()),
                "train/reward_scale_sigma": float(torch.sqrt(controller.return_tracker.sigma2 + 1e-8).detach().cpu().item()),
                "train/reward_scale_max_return": float(controller.return_tracker.max_ret_seen.detach().cpu().item()),
                "train/actor_lr": controller.sch_actor.get_last_lr()[0],
                "train/critic_lr": controller.sch_critic.get_last_lr()[0],
            }
            if log_finished_episodes > 0:
                scalars.update({
                    "episode/return": avg_return,
                    "episode/num_finished": log_finished_episodes,
                })
            if grad_updates > 0:
                scalars.update({
                    "train/critic_loss": avg_critic_loss,
                    "train/actor_loss": avg_actor_loss,
                    "train/temp_loss": avg_temp_loss,
                })
            tb_logger.log_scalars(env_step, scalars)
            loss_c, loss_a, loss_t, grad_updates = 0.0, 0.0, 0.0, 0
            log_return_sum, log_finished_episodes = 0.0, 0
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
            obs, prev_action, stam_context, ep_reward = reset_training_rollout_state(train_env, controller, cfg)
            print("[INFO] Training environment reset after validation.")
            next_eval_step += cfg['eval_freq']
            print("-" * 60 + "\n")

    train_env.close()
    tb_logger.close()

if __name__ == "__main__":
    run_experiment()
