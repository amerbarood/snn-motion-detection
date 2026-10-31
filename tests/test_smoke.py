"""Checks on a synthetic sequence (a bright square moving around); no downloads."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from snn_motion import analysis, preprocessing
from snn_motion.model import SNNModel
from snn_motion.train import TrainConfig, evaluate, train


def make_sequence(folder: Path, n_frames: int = 80) -> Path:
    """Square moves right, then down, then left, then up; writes frames + groundtruth.txt."""
    folder.mkdir(parents=True, exist_ok=True)
    x, y, rows = 40, 40, []
    moves = [(3, 0), (0, 3), (-3, 0), (0, -3)]
    for i in range(n_frames):
        dx, dy = moves[(i * 4) // n_frames]
        x, y = x + dx, y + dy
        img = np.zeros((160, 160), np.uint8)
        img[y : y + 20, x : x + 20] = 255
        cv2.imwrite(str(folder / f"{i:08d}.jpg"), img)
        rows.append(f"{x},{y},{x + 20},{y},{x + 20},{y + 20},{x},{y + 20}")
    (folder / "groundtruth.txt").write_text("\n".join(rows))
    return folder


@pytest.fixture(scope="module")
def sequence(tmp_path_factory):
    return make_sequence(tmp_path_factory.mktemp("seq") / "square")


def test_direction_labels():
    assert preprocessing.direction_label(0, -5) == 0  # up
    assert preprocessing.direction_label(0, 5) == 1  # down
    assert preprocessing.direction_label(-5, 1) == 2  # left
    assert preprocessing.direction_label(5, 1) == 3  # right


def test_preprocessing_shapes_and_labels(sequence):
    videos, labels, n_frames = preprocessing.prepare_sequence(sequence)
    assert n_frames == 80
    assert videos.shape[1:] == (5, 64, 64)
    assert set(labels) == {0, 1, 2, 3}
    assert labels[0] == 3  # first movement is to the right
    binary = preprocessing.encode_motion(preprocessing.load_frames(sequence)[:3], binary=True)
    assert set(np.unique(binary)) <= {0.0, 1.0}


def test_model_forward():
    out, mem = SNNModel()(torch.rand(3, 5, 64, 64))
    assert out.shape == (3, 4) and mem.shape == (3, 4)


def test_train_and_evaluate(sequence, tmp_path, monkeypatch):
    import snn_motion.train as t

    monkeypatch.setattr(t, "MODELS_DIR", tmp_path)
    videos, labels, _ = preprocessing.prepare_sequence(sequence)
    splits = preprocessing.split(videos, labels)
    model, history = train(splits, TrainConfig(epochs=3, print_every=1), "square")
    metrics = evaluate(model, splits.x_test, splits.y_test)
    assert 0 <= metrics["accuracy"] <= 100 and metrics["confusion_matrix"].shape == (4, 4)
    assert (tmp_path / "square_snn_best.pt").exists()


def test_balance_measures():
    df = pd.DataFrame({"total_frames": [100, 100], "count_up": [25, 100], "count_down": [25, 0],
                       "count_left": [25, 0], "count_right": [25, 0]})
    out = analysis.add_balance_measures(df)
    assert out["entropy"].tolist() == pytest.approx([2.0, 0.0])
    assert out["nsd"][0] == pytest.approx(0.0)
