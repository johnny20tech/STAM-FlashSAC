import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stam_flashsac.results import export_run_result, infer_variant_and_seed, is_complete_run, variant_with_nstep


DEFAULT_MAX_STEPS = 50_000_896
PAPER_MAIN_DATASET = "paper_main"
NSTEP1_ARCHIVE_DATASET = "nstep1_archive"


def migrate_checkpoint_runs(
    dataset: str,
    checkpoint_root: Path,
    results_dir: Path,
    max_steps: int,
    n_step: int,
) -> list[dict]:
    rows = []
    if not checkpoint_root.exists():
        return rows
    for run_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
        if not is_complete_run(run_dir, max_steps):
            continue
        variant, seed = infer_variant_and_seed(run_dir)
        if variant is None or seed is None:
            continue
        result_variant = variant_with_nstep(variant, n_step)
        dest = export_run_result(
            run_dir=run_dir,
            results_dir=results_dir,
            variant=result_variant,
            seed=seed,
            result_mode="overwrite",
            result_set=dataset,
            dataset=dataset,
            use_result_set_subdir=False,
            original_run_name=run_dir.name,
        )
        rows.append(
            {
                "dataset": dataset,
                "variant": result_variant,
                "seed": seed,
                "relative_path": str(dest.relative_to(results_dir)),
            }
        )
    return rows


def write_manifest(rows: list[dict], results_dir: Path) -> None:
    manifest = results_dir / "summaries" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "relative_path"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Migrate lightweight n-step archive and paper-main results.")
    parser.add_argument("--results-dir", type=str, default=str(repo_dir / "results"))
    parser.add_argument("--nstep1-archive-dir", type=str, default=None)
    parser.add_argument("--paper-main-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = []
    if args.nstep1_archive_dir:
        rows += migrate_checkpoint_runs(
            NSTEP1_ARCHIVE_DATASET,
            Path(args.nstep1_archive_dir) / "checkpoints",
            results_dir,
            args.max_steps,
            n_step=1,
        )
    if args.paper_main_dir:
        rows += migrate_checkpoint_runs(
            PAPER_MAIN_DATASET,
            Path(args.paper_main_dir) / "checkpoints",
            results_dir,
            args.max_steps,
            n_step=3,
        )
    write_manifest(rows, results_dir)
    print(f"Migrated {len(rows)} completed runs into {results_dir / 'experiment'}")
    print(f"Wrote {results_dir / 'summaries' / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
