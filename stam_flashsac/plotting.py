import argparse
import json
import os
import shutil
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
MPL_DIR = REPO_DIR / "runtime" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAIN_VARIANTS = ["baseline", "memory_k16_action"]
VARIANT_ORDER = [
    "baseline",
    "memory_k4_action",
    "memory_k8_action",
    "memory_k8_obs",
    "memory_k16_action",
    "memory_k16_obs",
]
DISPLAY_NAMES = {
    "baseline": "FlashSAC Baseline",
    "memory_k4_action": "Our Method (K=4)",
    "memory_k8_action": "Our Method (K=8)",
    "memory_k16_action": "Our Method (K=16)",
    "memory_k8_obs": "Our Method (K=8, w.o. action)",
    "memory_k16_obs": "Our Method (K=16, w.o. action)",
}
COLORS = {
    "baseline": "#111827",
    "memory_k4_action": "#0072B2",
    "memory_k8_action": "#D55E00",
    "memory_k16_action": "#CC79A7",
    "memory_k8_obs": "#009E73",
    "memory_k16_obs": "#E69F00",
}


def infer_env_variant_seed_dataset(csv_path: Path, results_dir: Path) -> tuple[str, str, int | None, str]:
    rel = csv_path.relative_to(results_dir)
    parts = rel.parts
    env_name = "unknown"
    variant = "unknown"
    seed = None
    dataset = "unknown"
    if len(parts) >= 5 and parts[0] == "experiment":
        env_name = parts[1]
        variant = parts[2]
        if parts[3].startswith("seed"):
            try:
                seed = int(parts[3].replace("seed", ""))
            except ValueError:
                seed = None
    cfg_path = csv_path.parent / "config.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        env_name = cfg.get("env_id", env_name)
        variant = cfg.get("variant_name", variant)
        seed = cfg.get("seed", seed)
    meta_path = csv_path.parent / "metadata.json"
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
        env_name = meta.get("env_name", env_name)
        variant = meta.get("variant", variant)
        dataset = meta.get("dataset", dataset)
    return env_name, variant, seed, dataset


def split_nstep_variant(variant: str) -> tuple[str, str | None]:
    marker = "_nstep"
    if marker not in variant:
        return variant, None
    base, n_step = variant.rsplit(marker, 1)
    return base, n_step


def display_name(variant: str, default_n_step: int = 3) -> str:
    base, n_step = split_nstep_variant(variant)
    name = DISPLAY_NAMES.get(base, variant)
    if n_step is None or n_step == str(default_n_step):
        return name
    return f"{name}, n-step={n_step}"


def color_for_variant(variant: str) -> str:
    base, _ = split_nstep_variant(variant)
    return COLORS.get(base, "#64748B")


def variant_sort_key(variant: str) -> tuple[int, int, str]:
    base, n_step = split_nstep_variant(variant)
    base_rank = VARIANT_ORDER.index(base) if base in VARIANT_ORDER else len(VARIANT_ORDER)
    n_rank = int(n_step) if n_step and n_step.isdigit() else 0
    return base_rank, n_rank, variant


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted((results_dir / "experiment").glob("*/*/*/**/training_log.csv")):
        env_name, variant, seed, dataset = infer_env_variant_seed_dataset(csv_path, results_dir)
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df = df.rename(columns={"Timestep": "step", "AvgReturn": "avg_return", "StdDev": "std_dev", "Score": "score"})
        df["env_name"] = env_name
        df["variant"] = variant
        df["seed"] = seed
        df["dataset"] = dataset
        df["run_path"] = str(csv_path.parent.relative_to(results_dir))
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def auc(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return float(y.iloc[-1]) if len(y) else float("nan")
    order = np.argsort(x.to_numpy())
    x_sorted = x.to_numpy()[order]
    y_sorted = y.to_numpy()[order]
    return float(np.trapz(y_sorted, x_sorted) / max(x_sorted[-1] - x_sorted[0], 1.0))


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, env_name, variant, seed), group in df.groupby(["dataset", "env_name", "variant", "seed"], dropna=False):
        group = group.sort_values("step")
        last = group.iloc[-1]
        rows.append(
            {
                "dataset": dataset,
                "env_name": env_name,
                "variant": variant,
                "seed": seed,
                "final_step": int(last["step"]),
                "final_avg_return": float(last["avg_return"]),
                "final_score": float(last["score"]),
                "best_avg_return": float(group["avg_return"].max()),
                "best_score": float(group["score"].max()),
                "auc_avg_return": auc(group["step"], group["avg_return"]),
                "auc_score": auc(group["step"], group["score"]),
                "run_path": last["run_path"],
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "env_name", "variant", "seed"])


def build_by_variant(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("variant", dropna=False)["final_avg_return"]
        .agg(["count", "mean", "std", "sem", "min", "max"])
        .reset_index()
        .sort_values("variant")
    )


def build_by_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("dataset", dropna=False)["final_avg_return"]
        .agg(["count", "mean", "std", "sem", "min", "max"])
        .reset_index()
        .sort_values("dataset")
    )


def build_by_env(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("env_name", dropna=False)["final_avg_return"]
        .agg(["count", "mean", "std", "sem", "min", "max"])
        .reset_index()
        .sort_values("env_name")
    )


def configure_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#94A3B8"
    plt.rcParams["axes.labelcolor"] = "#334155"
    plt.rcParams["xtick.color"] = "#475569"
    plt.rcParams["ytick.color"] = "#475569"


def format_return(value: float) -> str:
    return f"{value:,.0f}"


def plot_learning_curves(df: pd.DataFrame, out_path: Path, variants: list[str], default_n_step: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=240)
    ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    final_points = []
    for variant in variants:
        cur = df[df["variant"] == variant]
        if cur.empty:
            continue
        stats = cur.groupby("step")["avg_return"].agg(["mean", "sem"]).reset_index()
        x = stats["step"] / 1e6
        y = stats["mean"]
        err = stats["sem"].fillna(0.0)
        color = color_for_variant(variant)
        ax.plot(x, y, label=display_name(variant, default_n_step), color=color, linewidth=2.4)
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.16, linewidth=0)
        final_points.append((float(x.iloc[-1]), float(y.iloc[-1]), color))
    for idx, (x_val, y_val, color) in enumerate(sorted(final_points, key=lambda item: item[1])):
        ax.annotate(
            format_return(y_val),
            xy=(x_val, y_val),
            xytext=(10, -6 if idx % 2 == 0 else 8),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "alpha": 0.88},
        )
    ax.set_xlabel("Steps (Millions)", fontsize=11, labelpad=10)
    ax.set_ylabel("Average Evaluation Return", fontsize=11, labelpad=10)
    ax.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", framealpha=0.95)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def plot_selected_final_bars(summary: pd.DataFrame, out_path: Path, variants: list[str], default_n_step: int) -> None:
    order = [variant for variant in variants if variant in set(summary["variant"])]
    stats = summary.groupby("variant")["final_avg_return"].agg(["mean", "sem"]).reindex(order).dropna(how="all")
    if stats.empty:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=240)
    x = np.arange(len(stats))
    yerr = stats["sem"].fillna(0.0)
    bars = ax.bar(x, stats["mean"], yerr=yerr, color=[color_for_variant(v) for v in stats.index], alpha=0.92, capsize=4)
    for bar, value, err in zip(bars, stats["mean"], yerr):
        ax.annotate(
            format_return(float(value)),
            xy=(bar.get_x() + bar.get_width() / 2, float(value + err)),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#1E293B",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(v, default_n_step) for v in stats.index], rotation=12, ha="right")
    ax.set_ylabel("Final Average Return")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot STAM-FlashSAC results.")
    parser.add_argument("--results-dir", type=str, default=str(REPO_DIR / "results"))
    parser.add_argument("--figure-dataset", type=str, default="auto")
    parser.add_argument("--figure-env", type=str, default="auto")
    parser.add_argument("--default-n-step", type=int, default=3)
    return parser.parse_args()


def select_figure_dataset(summary: pd.DataFrame, requested: str) -> str:
    datasets = sorted(str(dataset) for dataset in summary["dataset"].dropna().unique())
    if requested != "auto":
        if requested not in datasets:
            raise ValueError(f"Unknown figure dataset '{requested}'. Available datasets: {', '.join(datasets)}")
        return requested
    if "paper_main" in datasets:
        return "paper_main"
    return datasets[-1]


def select_figure_env(summary: pd.DataFrame, dataset: str, requested: str) -> str:
    envs = sorted(str(env_name) for env_name in summary.loc[summary["dataset"] == dataset, "env_name"].dropna().unique())
    if requested != "auto":
        if requested not in envs:
            raise ValueError(f"Unknown figure environment '{requested}'. Available environments: {', '.join(envs)}")
        return requested
    return envs[-1]


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    summaries_dir = results_dir / "summaries"
    figures_dir = results_dir / "figures"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    df = load_results(results_dir)
    if df.empty:
        print(f"No training logs found under {results_dir / 'experiment'}")
        return 1
    summary = build_summary(df)
    summary.to_csv(summaries_dir / "summary.csv", index=False)
    build_by_variant(summary).to_csv(summaries_dir / "by_variant.csv", index=False)
    build_by_dataset(summary).to_csv(summaries_dir / "by_dataset.csv", index=False)
    build_by_env(summary).to_csv(summaries_dir / "by_env.csv", index=False)

    figure_dataset = select_figure_dataset(summary, args.figure_dataset)
    figure_env = select_figure_env(summary, figure_dataset, args.figure_env)
    dataset_df = df[(df["dataset"] == figure_dataset) & (df["env_name"] == figure_env)]
    dataset_summary = summary[(summary["dataset"] == figure_dataset) & (summary["env_name"] == figure_env)]
    variants = sorted(set(dataset_df["variant"]), key=variant_sort_key)
    desired_obs_order = ["memory_k8_obs", "memory_k8_action", "memory_k16_obs", "memory_k16_action"]
    plot_learning_curves(dataset_df, figures_dir / "all_learning_curves.png", variants, args.default_n_step)
    plot_selected_final_bars(
        dataset_summary,
        figures_dir / "50M_final_average_return_baseline_vs_our_method_K4_K8_K16.png",
        [
            variant for variant in variants
            if split_nstep_variant(variant)[0] in {"baseline", "memory_k4_action", "memory_k8_action", "memory_k16_action"}
        ],
        args.default_n_step,
    )
    plot_selected_final_bars(
        dataset_summary,
        figures_dir / "50M_final_average_return_action_vs_obs_only_K8_K16.png",
        sorted(
            [
                variant for variant in variants
                if split_nstep_variant(variant)[0] in set(desired_obs_order)
            ],
            key=lambda variant: (
                desired_obs_order.index(split_nstep_variant(variant)[0]),
                int(split_nstep_variant(variant)[1] or 0),
            ),
        ),
        args.default_n_step,
    )

    print(f"Wrote {summaries_dir / 'summary.csv'}")
    print(f"Wrote {figure_dataset} figures for {figure_env} under {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
