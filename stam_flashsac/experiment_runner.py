import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from stam_flashsac.results import export_run_result, is_complete_run, variant_with_nstep


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
DEFAULT_PYTHON = REPO_DIR / ".venv" / "bin" / "python"
DEFAULT_ENV_ID = "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0"


@dataclass(frozen=True)
class Variant:
    name: str
    use_stam: bool
    context_len: int = 0
    include_prev_action: bool = False


VARIANTS = {
    "baseline": Variant("baseline", use_stam=False),
    "memory_k4_action": Variant("memory_k4_action", use_stam=True, context_len=4, include_prev_action=True),
    "memory_k8_action": Variant("memory_k8_action", use_stam=True, context_len=8, include_prev_action=True),
    "memory_k16_action": Variant("memory_k16_action", use_stam=True, context_len=16, include_prev_action=True),
    "memory_k8_obs": Variant("memory_k8_obs", use_stam=True, context_len=8, include_prev_action=False),
    "memory_k16_obs": Variant("memory_k16_obs", use_stam=True, context_len=16, include_prev_action=False),
}

SUITES = {
    "main": ["baseline", "memory_k16_action"],
    "ablations": ["memory_k4_action", "memory_k8_action", "memory_k16_action", "memory_k8_obs", "memory_k16_obs"],
    "all": ["baseline", "memory_k16_action", "memory_k4_action", "memory_k8_action", "memory_k8_obs", "memory_k16_obs"],
}


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def setup_logging(output_root: Path) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stam-flashsac-runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_root / "experiment_runner.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def build_run_name(env_id: str, variant: Variant, seed: int) -> str:
    safe_env = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in env_id)
    return f"{safe_env}_{variant.name}_seed{seed}"


def selected_variants(args: argparse.Namespace) -> list[Variant]:
    names = [name.strip() for name in args.variants.split(",")] if args.variants else list(SUITES[args.suite])
    names = [name for name in names if name]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}")
    return [VARIANTS[name] for name in names]


def build_command(args: argparse.Namespace, variant: Variant, seed: int, output_root: Path) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "stam_flashsac.training",
        "--env-id",
        args.env_id,
        "--sim-device",
        f"cuda:{args.current_gpu}",
        "--variant-name",
        variant.name,
        "--seed",
        str(seed),
        "--max-steps",
        str(args.max_steps),
        "--num-train-envs",
        str(args.num_train_envs),
        "--explore-steps",
        str(args.explore_steps),
        "--batch-size",
        str(args.batch_size),
        "--updates-per-interaction-step",
        str(args.updates_per_interaction_step),
        "--eval-freq",
        str(args.eval_freq),
        "--logging-freq",
        str(args.logging_freq),
        "--num-eval-episodes",
        str(args.num_eval_episodes),
        "--buffer-capacity",
        str(args.buffer_capacity),
        "--n-step",
        str(args.n_step),
        "--output-root",
        str(output_root),
    ]
    cmd.append("--use-rise" if variant.use_stam else "--no-use-rise")
    if variant.use_stam:
        cmd.extend(["--rise-context-len", str(variant.context_len)])
        cmd.append("--rise-include-prev-action" if variant.include_prev_action else "--no-rise-include-prev-action")
    cmd.append("--compile" if args.use_compile else "--no-compile")
    cmd.append("--save-buffer" if args.save_buffer else "--no-save-buffer")
    return cmd


def launch_job(
    args: argparse.Namespace,
    logger: logging.Logger,
    variant: Variant,
    seed: int,
    gpu: int,
    output_root: Path,
) -> tuple[subprocess.Popen, object, str, Variant, int]:
    run_name = build_run_name(args.env_id, variant, seed)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "process.log"
    log_file = open(log_path, "a", buffering=1)

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.mask_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["NVIDIA_VISIBLE_DEVICES"] = str(gpu)
        args.current_gpu = 0
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("NVIDIA_VISIBLE_DEVICES", None)
        args.current_gpu = gpu
    env["PYTHONUNBUFFERED"] = "1"

    cmd = build_command(args, variant, seed, output_root)
    logger.info("launch %s on physical GPU %s", run_name, gpu)
    logger.info("log %s", log_path)
    logger.info("cmd CUDA_VISIBLE_DEVICES=%s %s", env.get("CUDA_VISIBLE_DEVICES", "<unset>"), " ".join(cmd))
    process = subprocess.Popen(cmd, cwd=str(REPO_DIR), env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    return process, log_file, run_name, variant, seed


def export_finished_run(args: argparse.Namespace, logger: logging.Logger, output_root: Path, variant: Variant, seed: int) -> None:
    run_dir = output_root / build_run_name(args.env_id, variant, seed)
    if not is_complete_run(run_dir, args.max_steps):
        logger.warning("not exporting incomplete run %s", run_dir.name)
        return
    dest = export_run_result(
        run_dir=run_dir,
        results_dir=Path(args.results_dir).resolve(),
        env_name=args.env_id,
        variant=variant_with_nstep(variant.name, args.n_step),
        seed=seed,
        result_mode=args.result_mode,
        result_set=args.result_set,
    )
    logger.info("exported lightweight result to %s", dest)


def run_queue(args: argparse.Namespace, logger: logging.Logger, variants: list[Variant], output_root: Path) -> int:
    seeds = parse_csv_ints(args.seeds)
    gpus = parse_csv_ints(args.gpus)
    queue = [(variant, seed) for seed in seeds for variant in variants]
    if args.skip_completed:
        queue = [
            (variant, seed)
            for variant, seed in queue
            if not is_complete_run(output_root / build_run_name(args.env_id, variant, seed), args.max_steps)
        ]
        logger.info("queued %d incomplete/missing run(s)", len(queue))

    active: list[dict] = []
    failures = 0
    while queue or active:
        while queue and len(active) < len(gpus) * args.processes_per_gpu:
            counts = {gpu: 0 for gpu in gpus}
            for job in active:
                counts[job["gpu"]] += 1
            free_gpus = [gpu for gpu in gpus if counts[gpu] < args.processes_per_gpu]
            if not free_gpus:
                break
            variant, seed = queue.pop(0)
            gpu = free_gpus[0]
            process, log_file, run_name, variant, seed = launch_job(args, logger, variant, seed, gpu, output_root)
            active.append(
                {"process": process, "log_file": log_file, "run_name": run_name, "gpu": gpu, "variant": variant, "seed": seed}
            )

        time.sleep(args.poll_seconds)
        still_active = []
        for job in active:
            ret = job["process"].poll()
            if ret is None:
                still_active.append(job)
                continue
            job["log_file"].close()
            if ret == 0:
                logger.info("finished %s on GPU %s", job["run_name"], job["gpu"])
                export_finished_run(args, logger, output_root, job["variant"], job["seed"])
            else:
                failures += 1
                logger.error("failed %s on GPU %s with exit code %s", job["run_name"], job["gpu"], ret)
        active = still_active
    return failures


def run_plotter(args: argparse.Namespace, logger: logging.Logger) -> int:
    cmd = [
        args.python,
        "-m",
        "stam_flashsac.plotting",
        "--results-dir",
        str(Path(args.results_dir).resolve()),
    ]
    logger.info("plot cmd %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_DIR))


def resolve_result_set(args: argparse.Namespace) -> str:
    if args.result_set:
        return args.result_set
    if args.result_mode == "overwrite":
        return "latest"
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STAM-FlashSAC experiments.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="main")
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--processes-per-gpu", type=int, default=1)
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=str(REPO_DIR / "results"))
    parser.add_argument("--result-mode", choices=["append", "overwrite"], default="append")
    parser.add_argument("--result-set", type=str, default=None)
    parser.add_argument("--python", type=str, default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)))
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
    parser.add_argument("--compile", dest="use_compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mask-cuda-visible-devices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    args.result_set = resolve_result_set(args)
    return args


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root or (REPO_DIR / "checkpoints" / args.result_set)).resolve()
    logger = setup_logging(output_root)
    variants = selected_variants(args)
    logger.info(
        "suite=%s variants=%s seeds=%s gpus=%s result_mode=%s result_set=%s",
        args.suite,
        ",".join(v.name for v in variants),
        args.seeds,
        args.gpus,
        args.result_mode,
        args.result_set,
    )

    if args.dry_run:
        gpus = parse_csv_ints(args.gpus)
        slots = [gpu for gpu in gpus for _ in range(args.processes_per_gpu)]
        queue = [(variant, seed) for seed in parse_csv_ints(args.seeds) for variant in variants]
        for idx, (variant, seed) in enumerate(queue):
            args.current_gpu = 0 if args.mask_cuda_visible_devices else slots[idx % len(slots)]
            logger.info("dry-run physical_gpu=%s %s", slots[idx % len(slots)], " ".join(build_command(args, variant, seed, output_root)))
        return 0

    failures = run_queue(args, logger, variants, output_root)
    if failures:
        logger.error("completed with %d failed run(s); skipping plots", failures)
        return 1
    if not args.skip_plot:
        return run_plotter(args, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
