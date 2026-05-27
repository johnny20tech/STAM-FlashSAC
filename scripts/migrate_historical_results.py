import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stam_flashsac.results import export_run_result, infer_variant_and_seed, is_complete_run, variant_with_nstep


DEFAULT_MAX_STEPS = 50_000_896


def migrate_checkpoint_runs(
    source_name: str,
    checkpoint_root: Path,
    results_dir: Path,
    max_steps: int,
    n_step: int,
) -> list[dict]:
    rows = []
    for run_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
        if not is_complete_run(run_dir, max_steps):
            continue
        variant, seed = infer_variant_and_seed(run_dir)
        if variant is None or seed is None:
            continue
        result_variant = variant_with_nstep(variant, n_step)
        dest = results_dir / "experiment" / result_variant / f"seed{seed}" / source_name
        if dest.exists():
            shutil.rmtree(dest)
        export_run_result(
            run_dir=run_dir,
            results_dir=results_dir,
            variant=result_variant,
            seed=seed,
            result_mode="append",
            result_set=source_name,
            source=source_name,
        )
        # export_run_result appends under seed/source only when seed already has contents.
        direct_dest = results_dir / "experiment" / result_variant / f"seed{seed}"
        if direct_dest.exists() and (direct_dest / "training_log.csv").exists():
            dest.mkdir(parents=True, exist_ok=True)
            for path in list(direct_dest.iterdir()):
                if path.name == source_name:
                    continue
                shutil.move(str(path), dest / path.name)
        rows.append(
            {
                "source": source_name,
                "variant": result_variant,
                "seed": seed,
                "relative_path": str(dest.relative_to(results_dir)),
            }
        )
    return rows


def copy_existing_summaries(source_name: str, source_results: Path, results_dir: Path) -> None:
    summary_dest = results_dir / "summaries"
    summary_dest.mkdir(parents=True, exist_ok=True)
    for filename in ("summary.csv", "paired_main.csv"):
        src = source_results / filename
        if src.exists():
            shutil.copy2(src, summary_dest / f"{source_name}_{filename}")


def write_manifest(rows: list[dict], results_dir: Path) -> None:
    manifest = results_dir / "summaries" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "variant", "seed", "relative_path"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parents[1]
    workspace = repo_dir.parent
    parser = argparse.ArgumentParser(description="Migrate final-exp3/final-exp5 lightweight results.")
    parser.add_argument("--results-dir", type=str, default=str(repo_dir / "results"))
    parser.add_argument("--final-exp3", type=str, default=str(workspace / "final-exp3"))
    parser.add_argument("--final-exp5", type=str, default=str(workspace / "final-exp5"))
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    final_exp3 = Path(args.final_exp3)
    final_exp5 = Path(args.final_exp5)
    rows = []
    rows += migrate_checkpoint_runs("final_exp3", final_exp3 / "checkpoints", results_dir, args.max_steps, n_step=1)
    rows += migrate_checkpoint_runs("final_exp5_completed", final_exp5 / "checkpoints", results_dir, args.max_steps, n_step=3)
    copy_existing_summaries("final_exp3", final_exp3 / "results", results_dir)
    copy_existing_summaries("final_exp5_completed", final_exp5 / "results_completed", results_dir)
    write_manifest(rows, results_dir)
    print(f"Migrated {len(rows)} completed runs into {results_dir / 'experiment'}")
    print(f"Wrote {results_dir / 'summaries' / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
