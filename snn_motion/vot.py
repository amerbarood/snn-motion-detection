"""Downloading VOT2015 sequences (images + rotated bounding-box ground truth)."""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DESCRIPTION_URL = "https://data.votchallenge.net/vot2015/dataset/description.json"
ANNOTATIONS_BASE = "https://data.votchallenge.net/vot2015/dataset/"
IMAGES_BASE = "https://data.votchallenge.net/sequences/"
SEQUENCES_DIR = Path("data/sequences")


def fetch_description(url: str = DESCRIPTION_URL) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def sequence_names(description: dict | None = None) -> list[str]:
    description = description or fetch_description()
    return [s["name"] for s in description["sequences"]]


def download_sequence(seq: dict, out_dir: Path = SEQUENCES_DIR) -> Path:
    """Download one sequence into out_dir/<name>/ as groundtruth.txt plus numbered .jpg frames."""
    name = seq["name"]
    seq_dir = Path(out_dir) / name
    if (seq_dir / "groundtruth.txt").exists() and any(seq_dir.glob("*.jpg")):
        print(f"{name}: already downloaded")
        return seq_dir
    seq_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print(f"{name}: downloading annotations...")
        ann_zip = tmp / "annotations.zip"
        urllib.request.urlretrieve(ANNOTATIONS_BASE + seq["annotations"]["url"], ann_zip)
        with zipfile.ZipFile(ann_zip) as zf:
            zf.extract("groundtruth.txt", seq_dir)

        print(f"{name}: downloading images...")
        img_zip = tmp / "images.zip"
        urllib.request.urlretrieve(IMAGES_BASE + seq["channels"]["color"]["uid"] + ".zip", img_zip)
        with zipfile.ZipFile(img_zip) as zf:
            for info in zf.infolist():
                if info.filename.endswith(".jpg"):
                    with zf.open(info) as src, open(seq_dir / Path(info.filename).name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    print(f"{name}: saved to {seq_dir}")
    return seq_dir


def download(names: list[str] | None = None, out_dir: Path = SEQUENCES_DIR) -> list[Path]:
    """Download the named sequences (all 60 when names is None)."""
    description = fetch_description()
    by_name = {s["name"]: s for s in description["sequences"]}
    names = names or list(by_name)
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"Unknown sequence(s): {unknown}. Available: {sorted(by_name)}")
    return [download_sequence(by_name[n], out_dir) for n in names]
