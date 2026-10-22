"""Train, tune and test the MLP baseline and the GCN on Facebook Page-Page.

Usage:
    python train.py tune     # grid search, model seeds 0-2, scored on validation

Every value in the grid comes from a published source (see ``GRIDS``). Test
labels are hidden (set to -1) during training and tuning, so any attempt to
use them in the loss raises an error; only the final runs (to come) may use them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data

from dataset import augment, get_data
from modules import NodeClassifier, build_model

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

TUNE_SEEDS = (0, 1, 2)
HIDDEN = -1  # label value for nodes whose labels the model must not see

# Sources: Kipf & Welling 2017 (K), Shchur et al. 2018 (S), Rong et al. 2020 (R).
GRIDS = {
    "gcn": {
        "num_layers": [2, 3, 4],  # K App. B: 2-3 best; R: best GCN depth 4
        "dropout": [0.5, 0.8],  # K: 0.5; S: best GCN 0.8
        "p_edge": [0.0, 0.3],  # K: none; R: best GCN on Cora keeps 70 % of edges
        "lr": [0.01, 0.005],  # K, S: GCN 0.01; S: MLP 0.005
    },
    "mlp": {
        "num_layers": [2],
        "dropout": [0.5, 0.8],  # same tuning for every model (S)
        "p_edge": [0.0],
        "lr": [0.01, 0.005],
    },
}


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


def grid(model: str) -> list[Config]:
    """Return every setting in the grid for one model.

    Args:
        model: ``"mlp"`` or ``"gcn"``.

    Returns:
        One ``Config`` per combination of grid values.
    """
    keys, values = zip(*GRIDS[model].items(), strict=True)
    return [
        Config(model, **dict(zip(keys, combo, strict=True))) for combo in itertools.product(*values)
    ]


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


def tune(data: Data) -> dict[str, Config]:
    """Grid-search both models on validation and save the results.

    Each setting is trained with seeds 0-2. The setting with the highest mean
    validation macro-F1 wins (Shchur et al. pick by mean validation score).

    Args:
        data: Graph with hidden test labels.

    Returns:
        The best ``Config`` for each model.
    """
    rows, best = [], {}
    for model in GRIDS:
        scored = []
        for cfg in grid(model):
            runs = [train_one(data, cfg, s)[1] for s in TUNE_SEEDS]
            f1s = torch.tensor([r["best_val_f1"] for r in runs])
            row = {
                **asdict(cfg),
                "val_f1_mean": f1s.mean().item(),
                "val_f1_std": f1s.std().item(),
                "best_epochs": [r["best_epoch"] for r in runs],
                "seconds_mean": sum(r["train_seconds"] for r in runs) / len(runs),
            }
            rows.append(row)
            scored.append((row["val_f1_mean"], cfg))
            print(
                f"{model} L={cfg.num_layers} drop={cfg.dropout} edge={cfg.p_edge} lr={cfg.lr}: "
                f"val F1 {row['val_f1_mean']:.4f} ± {row['val_f1_std']:.4f}  "
                f"best epochs {row['best_epochs']}"
            )
        best[model] = max(scored, key=lambda t: t[0])[1]
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "tuning.json"), "w") as f:
        json.dump({"rows": rows, "best": {m: asdict(c) for m, c in best.items()}}, f, indent=2)
    return best


def main() -> None:
    """Parse the command and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["tune"])
    parser.parse_args()
    tune(hide_test_labels(get_data()))


if __name__ == "__main__":
    main()
