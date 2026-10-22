"""Train, tune and test the MLP baseline and the GCN on Facebook Page-Page.

Usage:
    python train.py gcn      # train one model with default settings, report validation F1

Test
labels are hidden (set to -1) during training and tuning, so any attempt to
use them in the loss raises an error; only the final runs (to come) may use them.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data

from dataset import augment, get_data
from modules import NodeClassifier, build_model

HIDDEN = -1  # label value for nodes whose labels the model must not see


@dataclass(frozen=True)
class Config:
    """One training setting. Fixed values follow Kipf & Welling (2017) and Shchur et al. (2018)."""

    model: str
    num_layers: int = 2
    dropout: float = 0.5
    p_edge: float = 0.0
    lr: float = 0.01
    hidden_dim: int = 64  # S: best GCN hidden size
    weight_decay: float = 5e-4  # K
    epochs: int = 200  # K: maximum; the best-validation-loss checkpoint is kept


def set_seed(seed: int) -> None:
    """Seed PyTorch so that weights, dropout and DropEdge repeat exactly."""
    torch.manual_seed(seed)


def hide_test_labels(data: Data) -> Data:
    """Return a copy of the graph whose test labels are replaced by ``HIDDEN``.

    ``F.cross_entropy`` raises an error on a label of -1, so a bug that put test
    nodes into the loss would stop the run instead of leaking silently.

    Args:
        data: Graph with ``y`` and the three split masks.

    Returns:
        A copy with ``y[test_mask] == HIDDEN``.
    """
    safe = data.clone()
    safe.y = data.y.clone()
    safe.y[data.test_mask] = HIDDEN
    return safe


@torch.no_grad()
def evaluate(model: NodeClassifier, data: Data, mask: torch.Tensor) -> tuple[float, float]:
    """Return the loss and macro-F1 on the nodes in ``mask``.

    Args:
        model: Trained model.
        data: Graph (the full graph is used for message passing).
        mask: Nodes to score.

    Returns:
        ``(cross-entropy loss, macro-F1)``.

    Raises:
        ValueError: If ``mask`` contains a node with a hidden label.
    """
    if (data.y[mask] == HIDDEN).any():
        raise ValueError("attempted to score nodes whose labels are hidden")
    model.eval()
    logits = model(data.x, data.edge_index)[mask]
    loss = F.cross_entropy(logits, data.y[mask]).item()
    f1 = f1_score(data.y[mask].numpy(), logits.argmax(1).numpy(), average="macro")
    return loss, float(f1)


def train_one(data: Data, cfg: Config, seed: int) -> tuple[NodeClassifier, dict]:
    """Train one model and keep the weights from its lowest validation loss.

    Args:
        data: Graph with split masks; its test labels must already be hidden.
        cfg: Training setting.
        seed: Model seed (start weights, dropout and DropEdge patterns).

    Returns:
        The model with its best weights loaded, and a history dict with the
        per-epoch curves, the best epoch, its validation scores and the
        training time.
    """
    set_seed(seed)
    model = build_model(
        cfg.model,
        data.num_features,
        int(data.y.max()) + 1,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_loss, best_state, best_epoch = float("inf"), None, 0
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        edges = augment(data.edge_index, cfg.p_edge, training=True)
        out = model(data.x, edges)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_loss, val_f1 = evaluate(model, data, data.val_mask)
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    history["train_seconds"] = time.perf_counter() - start
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_loss
    history["best_val_f1"] = history["val_f1"][best_epoch - 1]
    model.load_state_dict(best_state)
    return model, history


def main() -> None:
    """Parse the command and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=["mlp", "gcn"])
    args = parser.parse_args()
    _, history = train_one(hide_test_labels(get_data()), Config(args.model), seed=0)
    print(f"{args.model}: best epoch {history['best_epoch']}, val F1 {history['best_val_f1']:.4f}")


if __name__ == "__main__":
    main()
