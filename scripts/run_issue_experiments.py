import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MPL_DIR = Path(__file__).resolve().parents[1] / "runtime" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stam_flashsac.experiment_runner import DEFAULT_ENV_ID, VARIANTS
from stam_flashsac.results import export_run_result, is_complete_run, sanitize_name, variant_with_nstep


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = REPO_DIR / ".venv" / "bin" / "python"
DEFAULT_FINAL_EXP5 = REPO_DIR.parent / "final-exp5" / "checkpoints"
LIGHTWEIGHT_REFERENCE_FILES = ("config.json", "metadata.json", "training_log.csv", "performance_chart.png", "process.log")
PLOT_COLORS = (
    "#111827",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#009E73",
    "#E69F00",
    "#56B4E9",
    "#7F3C8D",
    "#11A579",
    "#3969AC",
)
DISPLAY_NAMES = {
    "baseline": "FlashSAC",
    "memory_k16_action": "STAM K=16 + action",
    "memory_k16_obs": "STAM K=16 w.o. action",
    "memory_k32_action": "STAM K=32 + action",
    "memory_k16_action_concat": "STAM K=16 concat fusion",
    "memory_k16_action_gru": "STAM K=16 GRU",
    "memory_k16_action_mlp": "STAM K=16 MLP",
    "memory_k16_action_shuffled_action": "STAM K=16 shuffled action",
    "memory_k16_action_actor_only": "STAM K=16 actor only",
    "memory_k16_action_critic_only": "STAM K=16 critic only",
    "memory_k16_action_hidden64": "STAM K=16 hidden 64",
    "memory_k16_action_hidden256": "STAM K=16 hidden 256",
}


@dataclass(frozen=True)
class ReferenceSpec:
    variant: str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class IssueSpec:
    number: int
    run_variants: tuple[str, ...]
    references: tuple[ReferenceSpec, ...]


@dataclass(frozen=True)
class Job:
    issue: int
    variant: str
    seed: int


ISSUES = {
    8: IssueSpec(8, ("memory_k32_action",), (ReferenceSpec("memory_k16_action", (0, 1, 2)),)),
    9: IssueSpec(9, ("memory_k16_action_concat",), (ReferenceSpec("memory_k16_action", (0, 1, 2)),)),
    10: IssueSpec(
        10,
        ("memory_k16_action_gru", "memory_k16_action_mlp"),
        (ReferenceSpec("memory_k16_action", (0, 1, 2)),),
    ),
    11: IssueSpec(
        11,
        ("memory_k16_action_shuffled_action",),
        (ReferenceSpec("memory_k16_action", (0, 1, 2)), ReferenceSpec("memory_k16_obs", (0, 1, 2))),
    ),
    12: IssueSpec(
        12,
        ("memory_k16_action_actor_only", "memory_k16_action_critic_only"),
        (ReferenceSpec("memory_k16_action", (0, 1, 2)), ReferenceSpec("baseline", (0, 1, 2))),
    ),
    13: IssueSpec(
        13,
        ("memory_k16_action_hidden64", "memory_k16_action_hidden256"),
        (ReferenceSpec("memory_k16_action", (0, 1, 2)),),
    ),
}


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def issue_name(issue_number: int) -> str:
    return f"issue_{issue_number:02d}"


def run_name(env_id: str, variant: str, seed: int) -> str:
    return f"{sanitize_name(env_id)}_{variant}_seed{seed}"


def issue_checkpoint_root(args: argparse.Namespace, issue_number: int) -> Path:
    return Path(args.checkpoints_dir) / issue_name(issue_number)


def issue_results_root(args: argparse.Namespace, issue_number: int) -> Path:
    return Path(args.results_dir) / "issues" / issue_name(issue_number)


def result_run_dir(results_dir: Path, env_id: str, variant: str, seed: int, n_step: int) -> Path:
    return results_dir / "experiment" / sanitize_name(env_id) / variant_with_nstep(variant, n_step) / f"seed{seed}"


def final_exp5_run_dir(final_exp5_dir: Path, env_id: str, variant: str, seed: int) -> Path:
    return final_exp5_dir / run_name(env_id, variant, seed)


def checkpoint_run_dir(args: argparse.Namespace, job: Job) -> Path:
    return issue_checkpoint_root(args, job.issue) / run_name(args.env_id, job.variant, job.seed)


def issue_result_run_dir(args: argparse.Namespace, job: Job) -> Path:
    return result_run_dir(issue_results_root(args, job.issue), args.env_id, job.variant, job.seed, args.n_step)


def find_complete_reference(variant: str, seed: int, args: argparse.Namespace) -> tuple[Path | None, str, bool]:
    primary = result_run_dir(Path(args.results_dir), args.env_id, variant, seed, args.n_step)
    if is_complete_run(primary, args.max_steps):
        return primary, "results", True

    historical = final_exp5_run_dir(Path(args.final_exp5_dir), args.env_id, variant, seed)
    if is_complete_run(historical, args.max_steps):
        return historical, "final_exp5", True

    if primary.exists():
        return primary, "results", False
    if historical.exists():
        return historical, "final_exp5", False
    return None, "missing", False


def copy_reference(src: Path, dest: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dest.mkdir(parents=True, exist_ok=True)
    for name in LIGHTWEIGHT_REFERENCE_FILES:
        source_file = src / name
        if source_file.exists():
            shutil.copy2(source_file, dest / name)
    tensorboard_dir = src / "tensorboard"
    if tensorboard_dir.exists():
        for event_file in tensorboard_dir.glob("events.out.tfevents*"):
            event_dest = dest / "tensorboard" / event_file.name
            event_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(event_file, event_dest)


def write_manifest(issue_dir: Path, rows: list[dict]) -> None:
    issue_dir.mkdir(parents=True, exist_ok=True)
    (issue_dir / "manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with (issue_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_nstep_variant(variant: str) -> tuple[str, str | None]:
    marker = "_nstep"
    if marker not in variant:
        return variant, None
    base, n_step = variant.rsplit(marker, 1)
    return base, n_step


def display_name(variant: str) -> str:
    base, n_step = split_nstep_variant(variant)
    label = DISPLAY_NAMES.get(base, base)
    if n_step is None:
        return label
    return label


def variant_sort_key(variant: str) -> tuple[int, int, str]:
    base, _ = split_nstep_variant(variant)
    order = [
        "baseline",
        "memory_k16_obs",
        "memory_k16_action",
        "memory_k32_action",
        "memory_k16_action_concat",
        "memory_k16_action_gru",
        "memory_k16_action_mlp",
        "memory_k16_action_shuffled_action",
        "memory_k16_action_actor_only",
        "memory_k16_action_critic_only",
        "memory_k16_action_hidden64",
        "memory_k16_action_hidden256",
    ]
    rank = order.index(base) if base in order else len(order)
    k = 0
    if "_k" in base:
        try:
            k = int(base.split("_k", 1)[1].split("_", 1)[0])
        except ValueError:
            k = 0
    return rank, k, variant


def configure_plot_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#94A3B8"
    plt.rcParams["axes.labelcolor"] = "#334155"
    plt.rcParams["xtick.color"] = "#475569"
    plt.rcParams["ytick.color"] = "#475569"


def normalize_training_log(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.rename(columns={"Timestep": "step", "AvgReturn": "avg_return", "StdDev": "std_dev", "Score": "score"})


def infer_experiment_metadata(csv_path: Path, issue_root: Path) -> tuple[str, str, int | None, str]:
    rel = csv_path.relative_to(issue_root)
    parts = rel.parts
    env_name = "unknown"
    variant = "unknown"
    seed = None
    kind = "experiment"
    if len(parts) >= 5 and parts[0] == "experiment":
        env_name = parts[1]
        variant = parts[2]
        if parts[3].startswith("seed"):
            try:
                seed = int(parts[3].replace("seed", ""))
            except ValueError:
                seed = None
    metadata_path = csv_path.parent / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
        env_name = metadata.get("env_name", env_name)
        variant = metadata.get("variant", variant)
        seed = metadata.get("seed", seed)
    config_path = csv_path.parent / "config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)
        env_name = config.get("env_id", env_name)
        seed = config.get("seed", seed)
    return env_name, variant, seed, kind


def infer_reference_metadata(csv_path: Path, issue_root: Path, n_step: int, env_id: str) -> tuple[str, str, int | None, str]:
    rel = csv_path.relative_to(issue_root)
    parts = rel.parts
    variant = parts[1] if len(parts) >= 3 else "unknown"
    seed = None
    if len(parts) >= 3 and parts[2].startswith("seed"):
        try:
            seed = int(parts[2].replace("seed", ""))
        except ValueError:
            seed = None
    config_path = csv_path.parent / "config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)
        variant = config.get("variant_name", variant)
        seed = config.get("seed", seed)
        env_id = config.get("env_id", env_id)
    return env_id, variant_with_nstep(variant, n_step), seed, "reference"


def collect_issue_results(issue_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for csv_path in sorted((issue_root / "experiment").glob("*/*/seed*/**/training_log.csv")):
        df = normalize_training_log(csv_path)
        if df.empty:
            continue
        env_name, variant, seed, kind = infer_experiment_metadata(csv_path, issue_root)
        df["issue"] = issue_root.name
        df["kind"] = kind
        df["env_name"] = env_name
        df["variant"] = variant
        df["seed"] = seed
        df["run_path"] = str(csv_path.parent.relative_to(issue_root))
        frames.append(df)
    for csv_path in sorted((issue_root / "references").glob("*/*/training_log.csv")):
        df = normalize_training_log(csv_path)
        if df.empty:
            continue
        env_name, variant, seed, kind = infer_reference_metadata(csv_path, issue_root, args.n_step, args.env_id)
        df["issue"] = issue_root.name
        df["kind"] = kind
        df["env_name"] = env_name
        df["variant"] = variant
        df["seed"] = seed
        df["run_path"] = str(csv_path.parent.relative_to(issue_root))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def auc(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return float(y.iloc[-1]) if len(y) else float("nan")
    order = np.argsort(x.to_numpy())
    x_sorted = x.to_numpy()[order]
    y_sorted = y.to_numpy()[order]
    return float(np.trapz(y_sorted, x_sorted) / max(x_sorted[-1] - x_sorted[0], 1.0))


def build_issue_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (issue, kind, env_name, variant, seed, run_path), group in df.groupby(
        ["issue", "kind", "env_name", "variant", "seed", "run_path"],
        dropna=False,
    ):
        group = group.sort_values("step")
        last = group.iloc[-1]
        rows.append(
            {
                "issue": issue,
                "kind": kind,
                "env_name": env_name,
                "variant": variant,
                "label": display_name(str(variant)),
                "seed": int(seed) if pd.notna(seed) else "",
                "final_step": int(last["step"]),
                "final_avg_return": float(last["avg_return"]),
                "final_score": float(last["score"]) if "score" in group else float("nan"),
                "best_avg_return": float(group["avg_return"].max()),
                "auc_avg_return": auc(group["step"], group["avg_return"]),
                "run_path": run_path,
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "seed", "kind"])


def build_issue_by_variant(summary: pd.DataFrame) -> pd.DataFrame:
    stats = (
        summary.groupby(["variant", "label"], dropna=False)["final_avg_return"]
        .agg(["count", "mean", "std", "sem", "min", "max"])
        .reset_index()
    )
    return stats.sort_values("variant", key=lambda values: values.map(variant_sort_key))


def plot_issue_final_bars(summary: pd.DataFrame, out_path: Path) -> None:
    stats = build_issue_by_variant(summary)
    if stats.empty:
        return
    configure_plot_style()
    colors = {variant: PLOT_COLORS[idx % len(PLOT_COLORS)] for idx, variant in enumerate(stats["variant"])}
    fig, ax = plt.subplots(figsize=(max(8.0, len(stats) * 1.1), 5.4), dpi=240)
    x = np.arange(len(stats))
    yerr = stats["sem"].fillna(0.0)
    bars = ax.bar(x, stats["mean"], yerr=yerr, capsize=4, color=[colors[v] for v in stats["variant"]], alpha=0.9)
    for idx, variant in enumerate(stats["variant"]):
        vals = summary.loc[summary["variant"] == variant, "final_avg_return"].to_numpy()
        if len(vals):
            offsets = np.linspace(-0.14, 0.14, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(np.full(len(vals), x[idx]) + offsets, vals, s=22, color="#FFFFFF", edgecolor=colors[variant], linewidth=1.1, zorder=3)
    for bar, value, err in zip(bars, stats["mean"], yerr):
        ax.annotate(
            f"{float(value):,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, float(value + err)),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#1E293B",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(stats["label"], rotation=18, ha="right")
    ax.set_ylabel("Final Average Return")
    ax.grid(True, axis="y", linestyle="--", alpha=0.42, color="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.16)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def plot_issue_learning_curves(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    configure_plot_style()
    variants = sorted(df["variant"].dropna().unique(), key=variant_sort_key)
    colors = {variant: PLOT_COLORS[idx % len(PLOT_COLORS)] for idx, variant in enumerate(variants)}
    fig, ax = plt.subplots(figsize=(10.2, 6.0), dpi=240)
    final_labels = []
    for variant in variants:
        cur = df[df["variant"] == variant]
        stats = cur.groupby("step")["avg_return"].agg(["mean", "sem"]).reset_index()
        if stats.empty:
            continue
        x = stats["step"] / 1e6
        y = stats["mean"]
        err = stats["sem"].fillna(0.0)
        color = colors[variant]
        ax.plot(x, y, label=display_name(str(variant)), color=color, linewidth=2.2)
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.14, linewidth=0)
        final_labels.append((float(x.iloc[-1]), float(y.iloc[-1]), color))
    for idx, (x_val, y_val, color) in enumerate(sorted(final_labels, key=lambda item: item[1])):
        offset = -10 if idx % 2 == 0 else 10
        ax.annotate(
            f"{y_val:,.0f}",
            xy=(x_val, y_val),
            xytext=(10, offset),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
    ax.set_xlabel("Steps (Millions)")
    ax.set_ylabel("Average Evaluation Return")
    ax.grid(True, linestyle="--", alpha=0.42, color="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", framealpha=0.95)
    ax.margins(x=0.08, y=0.12)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def issue_ready_for_finalize(rows: list[dict]) -> bool:
    scheduled = [row for row in rows if row["kind"] == "scheduled_run"]
    return bool(scheduled) and all(row["status"] == "complete" for row in scheduled)


def finalize_issue_outputs(args: argparse.Namespace, issue_number: int, rows: list[dict], finalized: set[int]) -> bool:
    if issue_number in finalized or args.dry_run or not issue_ready_for_finalize(rows):
        return False
    issue_root = issue_results_root(args, issue_number)
    df = collect_issue_results(issue_root, args)
    if df.empty:
        return False
    summaries_dir = issue_root / "summaries"
    figures_dir = issue_root / "figures"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = build_issue_summary(df)
    summary.to_csv(summaries_dir / "final_returns.csv", index=False)
    build_issue_by_variant(summary).to_csv(summaries_dir / "final_return_by_variant.csv", index=False)
    plot_issue_final_bars(summary, figures_dir / "final_return_bars.png")
    plot_issue_learning_curves(df, figures_dir / "learning_curves.png")
    finalized.add(issue_number)
    print(f"[issue {issue_number}] wrote aggregate tables to {summaries_dir}")
    print(f"[issue {issue_number}] wrote aggregate figures to {figures_dir}")
    return True


def build_training_command(args: argparse.Namespace, job: Job, local_gpu: int) -> list[str]:
    variant = VARIANTS[job.variant]
    cmd = [
        args.python,
        "-m",
        "stam_flashsac.training",
        "--env-id",
        args.env_id,
        "--sim-device",
        f"cuda:{local_gpu}",
        "--variant-name",
        variant.name,
        "--seed",
        str(job.seed),
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
        str(issue_checkpoint_root(args, job.issue)),
    ]
    cmd.append("--use-stam" if variant.use_stam else "--no-use-stam")
    if variant.use_stam:
        cmd.extend(["--stam-context-len", str(variant.context_len)])
        cmd.extend(["--stam-hidden-dim", str(variant.hidden_dim)])
        cmd.extend(["--stam-fusion", variant.fusion])
        cmd.extend(["--stam-encoder", variant.encoder])
        cmd.extend(["--stam-action-mode", variant.action_mode])
        cmd.extend(["--stam-apply-to", variant.apply_to])
        cmd.append("--stam-include-prev-action" if variant.include_prev_action else "--no-stam-include-prev-action")
    cmd.append("--compile" if args.use_compile else "--no-compile")
    cmd.append("--save-buffer" if args.save_buffer else "--no-save-buffer")
    cmd.append("--wandb" if args.wandb else "--no-wandb")
    cmd.extend(["--wandb-project", args.wandb_project])
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])
    cmd.extend(["--wandb-mode", args.wandb_mode])
    return cmd


def collect_jobs_and_manifests(args: argparse.Namespace) -> tuple[list[Job], dict[int, list[dict]]]:
    seeds = parse_csv_ints(args.seeds)
    jobs: list[Job] = []
    manifests: dict[int, list[dict]] = {}

    for number in parse_csv_ints(args.issues):
        spec = ISSUES[number]
        issue_root = issue_results_root(args, number)
        rows: list[dict] = []
        scheduled: set[tuple[str, int]] = {(variant, seed) for variant in spec.run_variants for seed in seeds}

        for reference in spec.references:
            for seed in reference.seeds:
                src, source_kind, complete = find_complete_reference(reference.variant, seed, args)
                dest = issue_root / "references" / reference.variant / f"seed{seed}"
                rows.append(
                    {
                        "issue": number,
                        "kind": "reference",
                        "variant": reference.variant,
                        "seed": seed,
                        "status": "complete" if complete else "missing_or_incomplete",
                        "source_kind": source_kind,
                        "source_path": "" if src is None else str(src),
                        "dest_path": str(dest),
                        "command": "",
                    }
                )
                if complete and src is not None:
                    copy_reference(src, dest, args.dry_run)
                else:
                    scheduled.add((reference.variant, seed))

        for variant, seed in sorted(scheduled):
            if variant not in VARIANTS:
                raise ValueError(f"Issue {number} requested unknown variant: {variant}")
            job = Job(number, variant, seed)
            run_dir = checkpoint_run_dir(args, job)
            result_dir = issue_result_run_dir(args, job)
            status = "complete" if is_complete_run(run_dir, args.max_steps) or is_complete_run(result_dir, args.max_steps) else "pending"
            rows.append(
                {
                    "issue": number,
                    "kind": "scheduled_run",
                    "variant": variant,
                    "seed": seed,
                    "status": status,
                    "source_kind": "runner",
                    "source_path": str(run_dir),
                    "dest_path": str(result_dir),
                    "command": "",
                }
            )
            if status != "complete":
                jobs.append(job)
        manifests[number] = rows
    return jobs, manifests


def export_completed_job(args: argparse.Namespace, job: Job) -> Path | None:
    run_dir = checkpoint_run_dir(args, job)
    if not is_complete_run(run_dir, args.max_steps):
        return None
    return export_run_result(
        run_dir=run_dir,
        results_dir=issue_results_root(args, job.issue),
        env_name=args.env_id,
        variant=variant_with_nstep(job.variant, args.n_step),
        seed=job.seed,
        result_mode=args.result_mode,
        result_set=issue_name(job.issue),
    )


def update_manifest_rows(args: argparse.Namespace, manifests: dict[int, list[dict]], jobs: list[Job], status: str) -> None:
    job_keys = {(job.issue, job.variant, job.seed) for job in jobs}
    for number, rows in manifests.items():
        for row in rows:
            key = (int(row["issue"]), row["variant"], int(row["seed"]))
            if row["kind"] == "scheduled_run" and key in job_keys:
                row["status"] = status
                if status == "complete":
                    row["dest_path"] = str(issue_result_run_dir(args, Job(*key)))
        write_manifest(issue_results_root(args, number), rows)


def run_global_queue(args: argparse.Namespace, jobs: list[Job], manifests: dict[int, list[dict]], finalized: set[int]) -> int:
    gpus = parse_csv_ints(args.gpus)
    slots = [gpu for gpu in gpus for _ in range(args.processes_per_gpu)]
    queue = list(jobs)
    active: list[dict] = []
    failures = 0

    while queue or active:
        while queue and len(active) < len(slots):
            counts = {gpu: 0 for gpu in gpus}
            for item in active:
                counts[item["gpu"]] += 1
            free_gpus = [gpu for gpu in gpus if counts[gpu] < args.processes_per_gpu]
            if not free_gpus:
                break
            gpu = free_gpus[0]
            job = queue.pop(0)
            run_dir = checkpoint_run_dir(args, job)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(run_dir / "process.log", "a", buffering=1)
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["NVIDIA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            cmd = build_training_command(args, job, local_gpu=0)
            command_text = f"CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}"
            print(f"[issue {job.issue}] launch {job.variant} seed{job.seed} on physical GPU {gpu}")
            print(f"[issue {job.issue}] cmd {command_text}")
            for row in manifests[job.issue]:
                if row["kind"] == "scheduled_run" and row["variant"] == job.variant and int(row["seed"]) == job.seed:
                    row["status"] = "running"
                    row["command"] = command_text
                    row["source_path"] = str(run_dir)
            write_manifest(issue_results_root(args, job.issue), manifests[job.issue])
            process = subprocess.Popen(cmd, cwd=str(REPO_DIR), env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            active.append({"process": process, "log_file": log_file, "job": job, "gpu": gpu})

        time.sleep(args.poll_seconds)
        still_active = []
        for item in active:
            ret = item["process"].poll()
            if ret is None:
                still_active.append(item)
                continue
            item["log_file"].close()
            job = item["job"]
            exported = export_completed_job(args, job)
            status = "complete" if ret == 0 and exported is not None else "failed"
            if status == "failed":
                failures += 1
            print(f"[issue {job.issue}] {status} {job.variant} seed{job.seed} on GPU {item['gpu']} exit={ret}")
            for row in manifests[job.issue]:
                if row["kind"] == "scheduled_run" and row["variant"] == job.variant and int(row["seed"]) == job.seed:
                    row["status"] = status
                    row["dest_path"] = str(exported or issue_result_run_dir(args, job))
            write_manifest(issue_results_root(args, job.issue), manifests[job.issue])
            finalize_issue_outputs(args, job.issue, manifests[job.issue], finalized)
        active = still_active
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STAM-FlashSAC reviewer issue experiments with one global queue.")
    parser.add_argument("--issues", type=str, default="8")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--gpus", type=str, default="5")
    parser.add_argument("--processes-per-gpu", type=int, default=1)
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--checkpoints-dir", type=str, default=str(REPO_DIR / "checkpoints" / "issues"))
    parser.add_argument("--results-dir", type=str, default=str(REPO_DIR / "results"))
    parser.add_argument("--final-exp5-dir", type=str, default=str(DEFAULT_FINAL_EXP5))
    parser.add_argument("--result-mode", choices=["append", "overwrite"], default="append")
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
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="STAM-FlashSAC")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    unknown_issues = [issue for issue in parse_csv_ints(args.issues) if issue not in ISSUES]
    if unknown_issues:
        raise ValueError(f"Unknown issue(s): {unknown_issues}")
    return args


def main() -> int:
    args = parse_args()
    jobs, manifests = collect_jobs_and_manifests(args)
    finalized: set[int] = set()
    for number, rows in manifests.items():
        write_manifest(issue_results_root(args, number), rows)
        finalize_issue_outputs(args, number, rows, finalized)
    if args.dry_run:
        for job in jobs:
            cmd = build_training_command(args, job, local_gpu=0)
            print(f"[issue {job.issue}] dry-run CUDA_VISIBLE_DEVICES=<slot> {' '.join(cmd)}")
        update_manifest_rows(args, manifests, jobs, "dry_run")
        print(f"dry-run queued {len(jobs)} job(s) across {len(parse_csv_ints(args.gpus)) * args.processes_per_gpu} slot(s)")
        return 0
    failures = run_global_queue(args, jobs, manifests, finalized)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
