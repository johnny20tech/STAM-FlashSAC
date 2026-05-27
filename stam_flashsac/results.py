import csv
import json
import shutil
from pathlib import Path


LIGHTWEIGHT_RUN_FILES = ("config.json", "training_log.csv")


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


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
    env_name: str,
    variant: str,
    seed: int,
    result_mode: str,
    result_set: str,
    *,
    use_result_set_subdir: bool = True,
) -> Path:
    base = results_dir / "experiment" / sanitize_name(env_name) / variant / f"seed{seed}"
    if use_result_set_subdir and result_mode == "append" and base.exists() and any(base.iterdir()):
        return base / result_set
    return base


def export_run_result(
    run_dir: Path,
    results_dir: Path,
    env_name: str,
    variant: str,
    seed: int,
    result_mode: str,
    result_set: str,
    dataset: str | None = None,
    use_result_set_subdir: bool = True,
    original_run_name: str | None = None,
) -> Path:
    dest = result_destination(
        results_dir,
        env_name,
        variant,
        seed,
        result_mode,
        result_set,
        use_result_set_subdir=use_result_set_subdir,
    )
    if result_mode == "overwrite" and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for name in LIGHTWEIGHT_RUN_FILES:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)

    metadata = {
        "env_name": env_name,
        "variant": variant,
        "seed": seed,
        "dataset": dataset or result_set,
    }
    if original_run_name:
        metadata["original_run_name"] = original_run_name
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return dest


def infer_variant_seed_and_env(run_dir: Path) -> tuple[str | None, int | None, str | None]:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            cfg = json.load(f)
        return cfg.get("variant_name"), cfg.get("seed"), cfg.get("env_id")

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
    return variant, seed, None


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
