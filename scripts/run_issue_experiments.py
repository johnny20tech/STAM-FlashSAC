import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stam_flashsac.experiment_runner import DEFAULT_ENV_ID, VARIANTS
from stam_flashsac.results import is_complete_run, sanitize_name, variant_with_nstep


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = REPO_DIR / ".venv" / "bin" / "python"
DEFAULT_FINAL_EXP5 = REPO_DIR.parent / "final-exp5" / "checkpoints"
LIGHTWEIGHT_REFERENCE_FILES = ("config.json", "metadata.json", "training_log.csv", "performance_chart.png", "process.log")


@dataclass(frozen=True)
class ReferenceSpec:
    variant: str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class IssueSpec:
    number: int
    run_variants: tuple[str, ...]
    references: tuple[ReferenceSpec, ...]
    forced_runs: tuple[tuple[str, tuple[int, ...]], ...] = ()


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


def run_name(env_id: str, variant: str, seed: int) -> str:
    return f"{sanitize_name(env_id)}_{variant}_seed{seed}"


def result_run_dir(results_dir: Path, env_id: str, variant: str, seed: int, n_step: int) -> Path:
    return results_dir / "experiment" / sanitize_name(env_id) / variant_with_nstep(variant, n_step) / f"seed{seed}"


def final_exp5_run_dir(final_exp5_dir: Path, env_id: str, variant: str, seed: int) -> Path:
    return final_exp5_dir / run_name(env_id, variant, seed)


def find_complete_reference(
    variant: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Path | None, str, bool]:
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


def write_manifest(issue_dir: Path, rows: list[dict], dry_run: bool) -> None:
    issue_dir.mkdir(parents=True, exist_ok=True)
    (issue_dir / "manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with (issue_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def issue_runner_command(
    args: argparse.Namespace,
    issue_number: int,
    variants: list[str],
    seeds: list[int],
) -> list[str]:
    issue_name = f"issue_{issue_number:02d}"
    python = args.python
    return [
        python,
        "scripts/run_experiments.py",
        "--variants",
        ",".join(variants),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--gpus",
        args.gpus,
        "--processes-per-gpu",
        str(args.processes_per_gpu),
        "--env-id",
        args.env_id,
        "--output-root",
        str(Path(args.checkpoints_dir) / issue_name),
        "--results-dir",
        str(Path(args.results_dir) / "issues" / issue_name),
        "--result-mode",
        args.result_mode,
        "--result-set",
        issue_name,
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
        "--python",
        python,
        "--wandb-mode",
        args.wandb_mode,
        "--poll-seconds",
        str(args.poll_seconds),
        "--skip-plot",
    ]


def run_issue(args: argparse.Namespace, issue_number: int) -> int:
    spec = ISSUES[issue_number]
    seeds = parse_csv_ints(args.seeds)
    issue_name = f"issue_{issue_number:02d}"
    issue_results_dir = Path(args.results_dir) / "issues" / issue_name
    issue_manifest_dir = issue_results_dir
    manifest_rows = []

    scheduled: dict[str, set[int]] = {variant: set(seeds) for variant in spec.run_variants}
    for variant, forced_seeds in spec.forced_runs:
        scheduled.setdefault(variant, set()).update(forced_seeds)

    for reference in spec.references:
        for seed in reference.seeds:
            src, source_kind, complete = find_complete_reference(reference.variant, seed, args)
            dest = issue_results_dir / "references" / reference.variant / f"seed{seed}"
            row = {
                "issue": issue_number,
                "kind": "reference",
                "variant": reference.variant,
                "seed": seed,
                "status": "complete" if complete else "missing_or_incomplete",
                "source_kind": source_kind,
                "source_path": "" if src is None else str(src),
                "dest_path": str(dest),
            }
            manifest_rows.append(row)
            if complete and src is not None:
                copy_reference(src, dest, args.dry_run)
            else:
                scheduled.setdefault(reference.variant, set()).add(seed)

    for variant in list(scheduled):
        unknown = variant not in VARIANTS
        for seed in sorted(scheduled[variant]):
            manifest_rows.append(
                {
                    "issue": issue_number,
                    "kind": "scheduled_run",
                    "variant": variant,
                    "seed": seed,
                    "status": "unknown_variant" if unknown else "pending",
                    "source_kind": "runner",
                    "source_path": "",
                    "dest_path": str(issue_results_dir),
                    "command": "",
                }
            )
        if unknown:
            raise ValueError(f"Issue {issue_number} requested unknown variant: {variant}")

    write_manifest(issue_manifest_dir, manifest_rows, args.dry_run)
    runnable = {variant: sorted(seed_set) for variant, seed_set in scheduled.items() if seed_set}
    if not runnable:
        print(f"[issue {issue_number}] nothing to run")
        return 0

    failures = 0
    for seed_group in sorted({tuple(value) for value in runnable.values()}):
        variants = [variant for variant, variant_seeds in runnable.items() if tuple(variant_seeds) == seed_group]
        cmd = issue_runner_command(args, issue_number, variants, list(seed_group))
        command_text = " ".join(cmd)
        for row in manifest_rows:
            if row.get("kind") == "scheduled_run" and row["variant"] in variants and int(row["seed"]) in seed_group:
                row["command"] = command_text
        if args.dry_run:
            print(f"[issue {issue_number}] dry-run {command_text}")
            continue
        print(f"[issue {issue_number}] launch {command_text}")
        failures += subprocess.call(cmd, cwd=str(REPO_DIR))
    for row in manifest_rows:
        if row.get("kind") != "scheduled_run":
            continue
        if args.dry_run:
            row["status"] = "dry_run"
            row["command"] = row.get("command") or "not launched"
            continue
        variant = row["variant"]
        seed = int(row["seed"])
        result_dir = result_run_dir(issue_results_dir, args.env_id, variant, seed, args.n_step)
        row["status"] = "complete" if is_complete_run(result_dir, args.max_steps) else "missing_or_incomplete"
        row["dest_path"] = str(result_dir)
    write_manifest(issue_manifest_dir, manifest_rows, args.dry_run)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STAM-FlashSAC reviewer issue experiments.")
    parser.add_argument("--issues", type=str, default="8,9,10,11,12,13")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--gpus", type=str, default="2,3")
    parser.add_argument("--processes-per-gpu", type=int, default=2)
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
    failures = 0
    for issue_number in parse_csv_ints(args.issues):
        failures += run_issue(args, issue_number)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
