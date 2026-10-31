"""Training (SGD + ReduceLROnPlateau + early stopping), evaluation and per-sequence figures."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from .model import SNNModel  # noqa: E402
from .preprocessing import CLASS_NAMES, Splits  # noqa: E402

MODELS_DIR = Path("models")
OUTPUT_DIR = Path("outputs")


@dataclass
class TrainConfig:
    beta: float = 0.69
    lr: float = 0.07
    momentum: float = 0.9
    epochs: int = 500
    batch_size: int = 16
    patience: int = 100  # early stopping
    label_smoothing: float = 0.2
    lr_factor: float = 0.75
    lr_patience: int = 15
    min_lr: float = 1e-6
    print_every: int = 5


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_epoch: int = 0


def class_weights(labels: np.ndarray, n_classes: int = 4) -> torch.Tensor:
    """Normalised inverse-frequency weights."""
    counts = torch.bincount(torch.as_tensor(labels), minlength=n_classes).float()
    w = counts.sum() / (counts + 1e-6)
    return w / w.sum()


def _loader(x, y, batch_size, shuffle=False, drop_last=False):
    ds = TensorDataset(torch.as_tensor(x), torch.as_tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    if n == 0:
        return float("inf"), 0.0
    return total_loss / n, 100.0 * correct / n


def train(splits: Splits, cfg: TrainConfig, name: str = "sequence", device=None) -> tuple[SNNModel, History]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNNModel(input_size=splits.x_train[0, 0].size, beta=cfg.beta).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing, weight=class_weights(splits.y_train).to(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)

    # drop_last avoids a size-1 batch, which BatchNorm cannot train on.
    train_loader = _loader(splits.x_train, splits.y_train, cfg.batch_size, shuffle=True,
                           drop_last=len(splits.y_train) > cfg.batch_size)
    val_loader = _loader(splits.x_val, splits.y_val, cfg.batch_size)

    history = History()
    best_loss, best_state, since_best = float("inf"), None, 0
    for epoch in range(cfg.epochs):
        tr_loss, tr_acc = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, loss_fn, device)
        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.val_loss.append(va_loss)
        history.val_acc.append(va_acc)
        scheduler.step(va_loss)

        if (epoch + 1) % cfg.print_every == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{cfg.epochs} - train loss {tr_loss:.4f}, acc {tr_acc:.1f}% | "
                  f"val loss {va_loss:.4f}, acc {va_acc:.1f}% | lr {optimizer.param_groups[0]['lr']:.6f}")

        if va_loss < best_loss:
            best_loss, best_state, since_best = va_loss, copy.deepcopy(model.state_dict()), 0
            history.best_epoch = epoch + 1
        else:
            since_best += 1
            if since_best >= cfg.patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {cfg.patience} epochs)")
                break

    model.load_state_dict(best_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}_snn_best.pt"
    torch.save({"config": cfg.__dict__, "state_dict": best_state, "best_epoch": history.best_epoch,
                "val_loss": best_loss}, path)
    print(f"Best model: epoch {history.best_epoch}, val loss {best_loss:.4f} -> {path}")
    return model, history


@torch.no_grad()
def evaluate(model: SNNModel, x: np.ndarray, y: np.ndarray, batch_size: int = 16, device=None) -> dict:
    device = device or next(model.parameters()).device
    model.eval()
    preds = []
    for xb, _ in _loader(x, y, batch_size):
        preds.append(model(xb.to(device))[0].argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, labels=labels, zero_division=0)
    return {
        "accuracy": 100.0 * float((preds == y).mean()),
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "confusion_matrix": confusion_matrix(y, preds, labels=labels),
    }


def print_metrics(m: dict) -> None:
    print(f"Test accuracy {m['accuracy']:.2f}% | precision {m['precision'].mean():.2f}% | "
          f"recall {m['recall'].mean():.2f}% | F1 {m['f1'].mean():.2f}%")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<5} precision {m['precision'][i]:6.2f}%  recall {m['recall'][i]:6.2f}%  F1 {m['f1'][i]:6.2f}%")


def plot_sequence_results(name: str, history: History, m: dict) -> Path:
    cm = m["confusion_matrix"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"Movement tracking results for sequence: {name}", fontsize=16)

    axes[0, 0].plot(history.train_loss, "b-", label="Training")
    axes[0, 0].plot(history.val_loss, "r-", label="Validation")
    axes[0, 0].set(title="Learning curves", xlabel="Epoch", ylabel="Loss")
    axes[0, 1].plot(history.train_acc, "b-", label="Training")
    axes[0, 1].plot(history.val_acc, "r-", label="Validation")
    axes[0, 1].set(title="Accuracy curves", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes[0]:
        ax.legend()
        ax.grid(alpha=0.3)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1, 0])
    axes[1, 0].set(title=f"Confusion matrix (acc {m['accuracy']:.1f}%, F1 {m['f1'].mean():.1f}%)",
                   xlabel="Predicted", ylabel="True")

    per_class_acc = np.divide(100.0 * np.diag(cm), cm.sum(1), out=np.zeros(len(cm)), where=cm.sum(1) > 0)
    x, w = np.arange(len(CLASS_NAMES)), 0.2
    ax = axes[1, 1]
    for k, (label, values) in enumerate(
        [("Accuracy", per_class_acc), ("Precision", m["precision"]), ("Recall", m["recall"]), ("F1", m["f1"])]
    ):
        ax.bar(x + (k - 1.5) * w, values, w, label=label)
    ax.set(title="Metrics by class", ylabel="Score (%)", ylim=(0, 100))
    ax.set_xticks(x, CLASS_NAMES)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    ax.grid(alpha=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}_results.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path
