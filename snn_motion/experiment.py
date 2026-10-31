"""Run the full experiment: one SNN per sequence, results collected in a CSV."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import preprocessing, vot
from .train import OUTPUT_DIR, TrainConfig, evaluate, plot_sequence_results, print_metrics, train

RESULTS_DIR = Path("results")
DIRECTIONS = ["up", "down", "left", "right"]


def run_sequence(seq_dir: Path, cfg: TrainConfig, binary: bool = False, plot: bool = True) -> dict:
    name = Path(seq_dir).name
    videos, labels, n_frames = preprocessing.prepare_sequence(seq_dir, binary)
    splits = preprocessing.split(videos, labels)
    print(f"Split: {len(splits.y_train)} train, {len(splits.y_val)} val, {len(splits.y_test)} test")
    model, history = train(splits, cfg, name)
    m = evaluate(model, splits.x_test, splits.y_test, cfg.batch_size)
    print_metrics(m)
    if plot:
        plot_sequence_results(name, history, m)

    counts = np.bincount(labels, minlength=4)
    row = {"sequence_name": name, "total_frames": n_frames}
    row.update({f"count_{d}": int(c) for d, c in zip(DIRECTIONS, counts)})
    row["accuracy"] = m["accuracy"]
    for metric in ("precision", "recall", "f1"):
        row.update({f"{metric}_{d}": float(v) for d, v in zip(DIRECTIONS, m[metric])})
    row.update({
        "overall_precision": float(m["precision"].mean()),
        "overall_recall": float(m["recall"].mean()),
        "overall_f1": float(m["f1"].mean()),
    })
    return row


def run_all(names: list[str] | None, cfg: TrainConfig, download: bool = True, binary: bool = False,
            sequences_dir: Path = vot.SEQUENCES_DIR) -> Path:
    """Train/evaluate every sequence (downloading as needed); results saved after each one."""
    if download:
        dirs = vot.download(names, sequences_dir)
    else:
        dirs = [sequences_dir / n for n in names] if names else sorted(p for p in sequences_dir.iterdir() if p.is_dir())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"snn_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    rows = []
    for i, seq_dir in enumerate(dirs, 1):
        print(f"\n[{i}/{len(dirs)}] {seq_dir.name}")
        try:
            rows.append(run_sequence(seq_dir, cfg, binary))
        except Exception as e:  # keep going: some sequences are too short or too imbalanced
            print(f"Skipping {seq_dir.name}: {e}")
            continue
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nResults for {len(rows)} sequences saved to {out}")
    if rows:
        print(pd.DataFrame(rows)[["accuracy", "overall_precision", "overall_recall", "overall_f1"]].describe().round(1))
    return out
