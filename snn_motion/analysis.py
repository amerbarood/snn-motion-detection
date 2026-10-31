"""Cross-sequence analysis of a results CSV: summary table, class balance and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

DIRECTIONS = ["up", "down", "left", "right"]
METRICS = ["accuracy", "overall_precision", "overall_recall", "overall_f1"]
OUTPUT_DIR = Path("outputs/analysis")


def add_balance_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Normalised standard deviation (0 = perfectly balanced) and Shannon entropy
    (2 bits = perfectly balanced) of the per-direction label counts."""
    df = df.copy()
    counts = df[[f"count_{d}" for d in DIRECTIONS]].to_numpy(float)
    mu = df["total_frames"].to_numpy(float)[:, None] / 4
    df["nsd"] = np.sqrt(((counts - mu) ** 2).mean(1)) / mu[:, 0]
    p = counts / df["total_frames"].to_numpy(float)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["entropy"] = np.where(p > 0, -p * np.log2(p), 0).sum(1)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    s = df[METRICS].agg(["mean", "std", "median", "max"]).T
    s.index = ["Accuracy", "Precision", "Recall", "F1 score"]
    return s.round(1)


def _save(fig, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def _boxplot(ax, data: pd.DataFrame, labels: list[str], palette="viridis"):
    melted = data.melt(var_name="metric", value_name="value")
    sns.boxplot(data=melted, x="metric", y="value", hue="metric", palette=palette, width=0.5, legend=False,
                medianprops={"color": "red", "linewidth": 2}, ax=ax)
    sns.stripplot(data=melted, x="metric", y="value", size=4, color="black", alpha=0.3, jitter=True, ax=ax)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(-5, 105)
    for y in (25, 50, 75):
        ax.axhline(y, color="gray", linestyle="--", alpha=0.5)
    for i, col in enumerate(data.columns):
        ax.text(i, -3, f"Median: {data[col].median():.1f}", ha="center", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Value (%)")


def plot_all(df: pd.DataFrame) -> list[Path]:
    paths = []
    fig, ax = plt.subplots(figsize=(11, 7))
    _boxplot(ax, df[METRICS], ["Accuracy", "Precision", "Recall", "F1 score"])
    ax.set_title(f"Overall performance across {len(df)} sequences")
    paths.append(_save(fig, "overall_metrics"))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        _boxplot(ax, df[[f"{metric}_{d}" for d in DIRECTIONS]], [d.capitalize() for d in DIRECTIONS], "Set2")
        ax.set_title(f"{metric.capitalize()} by direction")
    paths.append(_save(fig, "metrics_by_direction"))

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    ordered = df.sort_values("accuracy", ascending=False)
    axes[0].bar(ordered["sequence_name"], ordered["accuracy"], color="skyblue")
    axes[0].set(title="Accuracy by sequence", ylabel="Accuracy (%)")
    axes[1].bar(ordered["sequence_name"], ordered["overall_f1"], color="lightgreen")
    axes[1].set(title="F1 score by sequence", ylabel="F1 (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=90)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    paths.append(_save(fig, "per_sequence"))

    df = add_balance_measures(df)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.regplot(data=df, x="entropy", y="overall_f1", ax=axes[0], scatter_kws={"s": 60, "alpha": 0.7})
    axes[0].axvline(2.0, color="red", linestyle="--", alpha=0.5)
    axes[0].set(title="Class balance (entropy) vs F1", xlabel="Entropy (bits)", ylabel="Overall F1 (%)")
    for ax, y in zip(axes[1:], ["nsd", "entropy"]):
        sc = ax.scatter(df["total_frames"], df[y], c=df["overall_f1"], cmap="cividis", s=80, edgecolors="black")
        ax.set(title=f"Number of frames vs {y.upper() if y == 'nsd' else y} (colour = F1)", xlabel="Frames", ylabel=y)
        fig.colorbar(sc, ax=ax, label="Overall F1 (%)")
    paths.append(_save(fig, "balance_vs_performance"))
    return paths


def run(csv: Path) -> None:
    df = pd.read_csv(csv)
    print(f"{len(df)} sequences in {csv}\n")
    print(summary(df).to_string())
    plot_all(df)
