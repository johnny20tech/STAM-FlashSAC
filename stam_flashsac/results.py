import csv
import json
import shutil
from pathlib import Path


LIGHTWEIGHT_RUN_FILES = ("config.json", "training_log.csv", "performance_chart.png")


def variant_with_nstep(variant: str, n_step: int | None) -> str:
    if n_step is None:
        return variant
    suffix = f"_nstep{n_step}"
    return variant if variant.endswith(suffix) else f"{variant}{suffix}"


def last_training_step(csv_path: Path) -> int | None:
    if not csv_path.exists():
        return None
    last_step = None
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                last_step = int(float(row[0]))
            except ValueError:
                continue
    return last_step


def is_complete_run(run_dir: Path, max_steps: int) -> bool:
    last_step = last_training_step(run_dir / "training_log.csv")
    return last_step is not None and last_step >= max_steps


def result_destination(
    results_dir: Path,
    variant: str,
    seed: int,
    result_mode: str,
    result_set: str,
) -> Path:
    base = results_dir / "experiment" / variant / f"seed{seed}"
    if result_mode == "append" and base.exists() and any(base.iterdir()):
        return base / result_set
    return base


def export_run_result(
    run_dir: Path,
    results_dir: Path,
    variant: str,
    seed: int,
    result_mode: str,
    result_set: str,
    source: str | None = None,
) -> Path:
    dest = result_destination(results_dir, variant, seed, result_mode, result_set)
    if result_mode == "overwrite" and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for name in LIGHTWEIGHT_RUN_FILES:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)

    metadata = {
        "variant": variant,
        "seed": seed,
        "source": source or result_set,
        "run_dir": str(run_dir),
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return dest


def infer_variant_and_seed(run_dir: Path) -> tuple[str | None, int | None]:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            cfg = json.load(f)
        return cfg.get("variant_name"), cfg.get("seed")

    name = run_dir.name
    seed = None
    variant = None
    if "_seed" in name:
        prefix, raw_seed = name.rsplit("_seed", 1)
        try:
            seed = int(raw_seed)
        except ValueError:
            seed = None
        for candidate in (
            "memory_k16_action",
            "memory_k16_obs",
            "memory_k8_action",
            "memory_k8_obs",
            "memory_k4_action",
            "baseline",
        ):
            if prefix.endswith(candidate):
                variant = candidate
                break
    return variant, seed


def copy_tree_files(src_dir: Path, dest_dir: Path, patterns: tuple[str, ...]) -> int:
    copied = 0
    for pattern in patterns:
        for src in src_dir.glob(pattern):
            if not src.is_file():
                continue
            rel = src.relative_to(src_dir)
            dest = dest_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
    return copied
