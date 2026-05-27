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


def infer_variant_seed(csv_path: Path, results_dir: Path) -> tuple[str, int | None, str]:
    rel = csv_path.relative_to(results_dir)
    parts = rel.parts
    variant = "unknown"
    seed = None
    source = "unknown"
    if len(parts) >= 4 and parts[0] == "experiment":
        variant = parts[1]
        if parts[2].startswith("seed"):
            try:
                seed = int(parts[2].replace("seed", ""))
            except ValueError:
                seed = None
        if len(parts) >= 5:
            source = parts[3]
    cfg_path = csv_path.parent / "config.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        variant = cfg.get("variant_name", variant)
        seed = cfg.get("seed", seed)
    meta_path = csv_path.parent / "metadata.json"
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
        variant = meta.get("variant", variant)
        source = meta.get("source", source)
    return variant, seed, source


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
    for csv_path in sorted((results_dir / "experiment").glob("*/*/**/training_log.csv")):
        variant, seed, source = infer_variant_seed(csv_path, results_dir)
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df = df.rename(columns={"Timestep": "step", "AvgReturn": "avg_return", "StdDev": "std_dev", "Score": "score"})
        df["variant"] = variant
        df["seed"] = seed
        df["source"] = source
        df["run_dir"] = str(csv_path.parent)
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
    for (source, variant, seed), group in df.groupby(["source", "variant", "seed"], dropna=False):
        group = group.sort_values("step")
        last = group.iloc[-1]
        rows.append(
            {
                "source": source,
                "variant": variant,
                "seed": seed,
                "final_step": int(last["step"]),
                "final_avg_return": float(last["avg_return"]),
                "final_score": float(last["score"]),
                "best_avg_return": float(group["avg_return"].max()),
                "best_score": float(group["score"].max()),
                "auc_avg_return": auc(group["step"], group["avg_return"]),
                "auc_score": auc(group["step"], group["score"]),
                "run_dir": last["run_dir"],
            }
        )
    return pd.DataFrame(rows).sort_values(["source", "variant", "seed"])


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
    ax.set_xlabel("Steps (Millions)", fontsize=11, labelpad=10)
    ax.set_ylabel("Average Evaluation Return", fontsize=11, labelpad=10)
    ax.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", framealpha=0.95)
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
    parser.add_argument("--figure-source", type=str, default="auto")
    parser.add_argument("--default-n-step", type=int, default=3)
    return parser.parse_args()


def select_figure_source(summary: pd.DataFrame, requested: str) -> str:
    sources = sorted(str(source) for source in summary["source"].dropna().unique())
    if requested != "auto":
        if requested not in sources:
            raise ValueError(f"Unknown figure source '{requested}'. Available sources: {', '.join(sources)}")
        return requested
    if "final_exp5_completed" in sources:
        return "final_exp5_completed"
    return sources[-1]


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

    figure_source = select_figure_source(summary, args.figure_source)
    source_df = df[df["source"] == figure_source]
    source_summary = summary[summary["source"] == figure_source]
    variants = sorted(set(source_df["variant"]), key=variant_sort_key)
    desired_obs_order = ["memory_k8_obs", "memory_k8_action", "memory_k16_obs", "memory_k16_action"]
    plot_learning_curves(source_df, figures_dir / "all_learning_curves.png", variants, args.default_n_step)
    plot_selected_final_bars(
        source_summary,
        figures_dir / "50M_final_average_return_baseline_vs_our_method_K4_K8_K16.png",
        [
            variant for variant in variants
            if split_nstep_variant(variant)[0] in {"baseline", "memory_k4_action", "memory_k8_action", "memory_k16_action"}
        ],
        args.default_n_step,
    )
    plot_selected_final_bars(
        source_summary,
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
    print(f"Wrote {figure_source} figures under {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
