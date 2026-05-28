import argparse
import json
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Record an MP4 rollout from a trained STAM-FlashSAC checkpoint.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing config.json and checkpoints.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help='Checkpoint to load: "best", "latest", or an explicit .pt path.',
    )
    parser.add_argument("--out", type=Path, default=None, help="Output MP4 path. Defaults to <run-dir>/rollout.mp4.")
    parser.add_argument("--device", type=str, default=None, help='IsaacLab/PyTorch device, e.g. "cuda:0".')
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs to instantiate.")
    parser.add_argument("--steps", type=int, default=250, help="Number of rollout control steps.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
    parser.add_argument("--frame-stride", type=int, default=2, help="Record one frame every N env steps.")
    parser.add_argument(
        "--action-scale",
        type=float,
        default=None,
        help="Override the environment action scale. Defaults to the training wrapper scale for the task.",
    )
    parser.add_argument("--width", type=int, default=960, help="Rendered frame width.")
    parser.add_argument("--height", type=int, default=540, help="Rendered frame height.")
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        default=(2.0, 1.6, 1.2),
        metavar=("X", "Y", "Z"),
        help="Viewport camera position. Defaults to a close table view.",
    )
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        default=(0.5, 0.0, 0.15),
        metavar=("X", "Y", "Z"),
        help="Viewport camera target. Defaults to the lift table/object area.",
    )
    parser.add_argument(
        "--rendering-mode",
        choices=("performance", "balanced", "quality"),
        default="balanced",
        help="IsaacLab rendering preset.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the run config seed.")
    return parser.parse_args()


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "best":
        return run_dir / "controller_best.pt"
    if checkpoint == "latest":
        return run_dir / "controller_latest.pt"
    return Path(checkpoint).expanduser().resolve()


def strip_compile_prefix(state_dict):
    prefix = "_orig_mod."
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def extract_policy_obs(obs_dict, env):
    import torch

    obs = obs_dict["policy"]
    if "critic" in env.unwrapped.single_observation_space:
        obs = torch.cat((obs, obs_dict["critic"]), dim=-1)
    return obs.detach().cpu().numpy()


def append_video_frame(writer, frame):
    import numpy as np

    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[-1] > 3:
        frame = frame[..., :3]
    writer.append_data(np.ascontiguousarray(frame))


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing run config: {cfg_path}")

    with cfg_path.open() as f:
        cfg = json.load(f)

    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    out_path = (args.out or (run_dir / "rollout.mp4")).expanduser().resolve()
    if out_path.suffix.lower() != ".mp4":
        mp4_path = out_path.with_suffix(".mp4")
        print(f"[RecordMP4] Changing output extension from {out_path.name} to {mp4_path.name}")
        out_path = mp4_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device or cfg.get("sim_device") or "cuda:0"
    seed = cfg["seed"] if args.seed is None else args.seed

    from stam_flashsac.flashsac.runtime import configure_runtime, configure_torch_runtime

    configure_runtime()

    from isaaclab.app import AppLauncher

    print("[RecordMP4] Launching IsaacLab with offscreen video rendering...", flush=True)
    app_launcher = AppLauncher(
        headless=True,
        device=device,
        enable_cameras=True,
        video=True,
        rendering_mode=args.rendering_mode,
    )
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import imageio.v2 as imageio
        import numpy as np
        import torch

        configure_torch_runtime(torch)

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        from stam_flashsac.flashsac.agent import ContinuousControlAgent
        from stam_flashsac.vendor.flash_rl.envs.isaaclab import ACTION_BOUNDS

        print("[RecordMP4] Creating environment...", flush=True)
        env_cfg = parse_env_cfg(cfg["env_id"], device=device, num_envs=args.num_envs)
        env_cfg.seed = seed
        env_cfg.viewer.resolution = (args.width, args.height)
        env_cfg.viewer.eye = tuple(args.camera_eye)
        env_cfg.viewer.lookat = tuple(args.camera_lookat)
        env = gym.make(cfg["env_id"], cfg=env_cfg, render_mode="rgb_array")

        torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        agent_cfg = dict(cfg)
        agent_cfg["sim_device"] = device
        agent_cfg["num_train_envs"] = args.num_envs
        agent_cfg["use_compile"] = False
        controller = ContinuousControlAgent(agent_cfg, torch_device)

        ckpt = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
        controller.actor.load_state_dict(strip_compile_prefix(ckpt["actor"]), strict=True)
        controller.actor.eval()
        print(
            f"[RecordMP4] Loaded {checkpoint_path} "
            f"(step={ckpt.get('step', 'unknown')}, best_score={ckpt.get('best_score', 'unknown')})"
        )

        print("[RecordMP4] Resetting environment...", flush=True)
        obs_dict, _ = env.reset()
        print("[RecordMP4] Environment reset complete.", flush=True)
        obs = extract_policy_obs(obs_dict, env)
        prev_action = np.zeros((args.num_envs, agent_cfg["a_dim"]), dtype=np.float32)
        stam_context = controller.make_initial_context(obs, prev_action)
        action_bounds = ACTION_BOUNDS.get(cfg["env_id"], 1.0) if args.action_scale is None else args.action_scale
        print(f"[RecordMP4] Using action scale: {action_bounds}", flush=True)

        writer = imageio.get_writer(
            out_path,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        )
        frames_written = 0
        print("[RecordMP4] Rendering first frame...", flush=True)
        first_frame = env.render()
        if first_frame is not None and np.any(first_frame):
            append_video_frame(writer, first_frame)
            frames_written += 1
        print(f"[RecordMP4] First render complete; frames_written={frames_written}.", flush=True)

        try:
            with torch.inference_mode():
                for step in range(args.steps):
                    token = controller.make_context_token(obs, prev_action)
                    stam_context = controller.append_context_token(stam_context, token)
                    action = controller.get_action(obs, context=stam_context, evaluate=True)
                    scaled_action = np.clip(action, -1.0, 1.0) * action_bounds
                    torch_action = torch.as_tensor(scaled_action, dtype=torch.float32, device=env.unwrapped.device)

                    next_obs_dict, _, terminated, truncated, _ = env.step(torch_action)
                    obs = extract_policy_obs(next_obs_dict, env)
                    prev_action = action.astype(np.float32)

                    if (step + 1) % args.frame_stride == 0:
                        frame = env.render()
                        if frame is not None:
                            append_video_frame(writer, frame)
                            frames_written += 1
                            print(f"[RecordMP4] Captured frame {frames_written} at step {step + 1}.", flush=True)

                    terminal = torch.logical_or(terminated, truncated).detach().cpu().numpy()
                    if np.any(terminal) and stam_context is not None:
                        prev_action[terminal] = 0.0
                        stam_context[terminal] = controller.make_initial_context(obs[terminal], prev_action[terminal])
        finally:
            writer.close()

        if frames_written == 0:
            raise RuntimeError("No frames were captured; check that IsaacLab rendering is enabled.")

        print(f"[RecordMP4] Wrote {frames_written} frames to {out_path}")
        env.close()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
