"""Command-line entry point.

    python main.py list                              # names of the 60 VOT2015 sequences
    python main.py download bag ball1                # download sequences into data/sequences/
    python main.py train bag                         # train + evaluate one sequence
    python main.py run-all                           # every sequence -> results/snn_results_<time>.csv
    python main.py run-all --sequences bag ball1 --epochs 100
    python main.py analyze results/snn_results_<time>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def add_train_args(p: argparse.ArgumentParser) -> None:
    from snn_motion.train import TrainConfig

    d = TrainConfig()
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--beta", type=float, default=d.beta, help="membrane decay rate of the LIF neurons")
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--patience", type=int, default=d.patience, help="early-stopping patience")
    p.add_argument("--binary-spikes", action="store_true",
                   help="threshold the DWT maps into binary spikes (top 25%%) instead of continuous input")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Motion direction detection with spiking neural networks")
    parser.add_argument("--seed", type=int, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list VOT2015 sequence names")

    p = sub.add_parser("download", help="download VOT2015 sequences")
    p.add_argument("names", nargs="*", help="sequence names (default: all)")

    p = sub.add_parser("train", help="train and evaluate the SNN on one sequence")
    p.add_argument("sequence", help="sequence name (downloaded if needed) or path to a sequence folder")
    add_train_args(p)

    p = sub.add_parser("run-all", help="train and evaluate on many sequences, collecting a results CSV")
    p.add_argument("--sequences", nargs="*", help="sequence names (default: all 60)")
    p.add_argument("--no-download", action="store_true", help="use sequences already in data/sequences/")
    add_train_args(p)

    p = sub.add_parser("analyze", help="summary table and plots from a results CSV")
    p.add_argument("csv", type=Path)
    return parser


def train_config(args):
    from snn_motion.train import TrainConfig

    return TrainConfig(beta=args.beta, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                       patience=args.patience)


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.command == "list":
        from snn_motion import vot

        print("\n".join(vot.sequence_names()))

    elif args.command == "download":
        from snn_motion import vot

        vot.download(args.names or None)

    elif args.command == "train":
        from snn_motion import experiment, vot

        seq_dir = Path(args.sequence)
        if not seq_dir.is_dir():
            seq_dir = vot.download([args.sequence])[0]
        experiment.run_sequence(seq_dir, train_config(args), args.binary_spikes)

    elif args.command == "run-all":
        from snn_motion import experiment

        experiment.run_all(args.sequences or None, train_config(args), download=not args.no_download,
                           binary=args.binary_spikes)

    elif args.command == "analyze":
        from snn_motion import analysis

        analysis.run(args.csv)


if __name__ == "__main__":
    main()
