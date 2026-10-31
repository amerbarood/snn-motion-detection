"""From video frames to labelled 5-frame mini-videos.

frames -> grayscale 128x128 -> |frame difference| -> Haar DWT ->
0.4*LL + 0.3*LH + 0.3*HL (each scaled to max 1) -> 64x64 input per time step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pywt
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["Up", "Down", "Left", "Right"]
IMAGE_SIZE = (128, 128)
MINI_VIDEO_SIZE = 5


def load_frames(seq_dir: Path, size=IMAGE_SIZE) -> np.ndarray:
    files = sorted(p for p in Path(seq_dir).iterdir() if p.suffix.lower() == ".jpg")
    if not files:
        raise FileNotFoundError(f"No .jpg frames in {seq_dir}")
    return np.stack([cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), size) for f in files])


def load_ground_truth(seq_dir: Path) -> np.ndarray:
    """Rotated boxes, one row per frame: x1,y1,x2,y2,x3,y3,x4,y4."""
    files = sorted(Path(seq_dir).glob("*groundtruth.txt"))
    if not files:
        raise FileNotFoundError(f"No groundtruth file in {seq_dir}")
    return np.loadtxt(files[0], delimiter=",", ndmin=2)


def _scaled(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x / m if m > 0 else x


def encode_motion(frames: np.ndarray, binary: bool = False, percentile: float = 75) -> np.ndarray:
    """Encode consecutive frame pairs; returns len(frames)-1 maps of size 64x64.

    By default the continuous DWT combination is returned, which is what the
    original experiments fed to the network. With binary=True the top 25% of
    values become 1 (spikes) and the rest 0.
    """
    out = []
    for prev, curr in zip(frames[:-1], frames[1:]):
        diff = np.abs(curr.astype(np.float32) - prev.astype(np.float32))
        LL, (LH, HL, _HH) = pywt.dwt2(diff, "haar")
        combined = 0.4 * _scaled(LL) + 0.3 * _scaled(LH) + 0.3 * _scaled(HL)
        if binary:
            combined = (combined > np.percentile(combined, percentile)).astype(np.float32)
        out.append(combined)
    return np.array(out, dtype=np.float32)


def box_centres(ground_truth: np.ndarray) -> np.ndarray:
    return np.stack([ground_truth[:, [0, 2, 4, 6]].mean(1), ground_truth[:, [1, 3, 5, 7]].mean(1)], axis=1)


def direction_label(dx: float, dy: float) -> int:
    """0 Up, 1 Down, 2 Left, 3 Right (image y grows downwards)."""
    if abs(dy) > abs(dx):
        return 0 if dy < 0 else 1
    return 2 if dx < 0 else 3


def make_mini_videos(encoded: np.ndarray, ground_truth: np.ndarray, size: int = MINI_VIDEO_SIZE):
    """Sliding window of `size` encoded frames; the label compares the box centre
    in the first and last frame of the window."""
    centres = box_centres(ground_truth)
    videos, labels = [], []
    for i in range(len(encoded) - size):
        videos.append(encoded[i : i + size])
        dx, dy = centres[i + size - 1] - centres[i]
        labels.append(direction_label(dx, dy))
    return np.array(videos, dtype=np.float32), np.array(labels, dtype=np.int64)


@dataclass
class Splits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def split(videos: np.ndarray, labels: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15,
          seed: int = 42) -> Splits:
    """Stratified 70/15/15 split (stratification is dropped if a class is too rare for it)."""
    def strat(y):
        _, counts = np.unique(y, return_counts=True)
        return y if counts.min() >= 2 else None

    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        videos, labels, test_size=1 - train_ratio, random_state=seed, stratify=strat(labels))
    val_size = val_ratio / (1 - train_ratio)
    x_val, x_te, y_val, y_te = train_test_split(
        x_tmp, y_tmp, test_size=1 - val_size, random_state=seed, stratify=strat(y_tmp))
    return Splits(x_tr, y_tr, x_val, y_val, x_te, y_te)


def prepare_sequence(seq_dir: Path, binary: bool = False):
    """Load a downloaded sequence and return (mini-videos, labels, number of frames)."""
    frames = load_frames(seq_dir)
    gt = load_ground_truth(seq_dir)
    videos, labels = make_mini_videos(encode_motion(frames, binary), gt)
    counts = np.bincount(labels, minlength=4)
    dist = ", ".join(f"{n}: {c}" for n, c in zip(CLASS_NAMES, counts))
    print(f"{Path(seq_dir).name}: {len(frames)} frames -> {len(videos)} mini-videos ({dist})")
    return videos, labels, len(frames)
