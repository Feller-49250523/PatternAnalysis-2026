"""Evaluate the saved checkpoints: test metrics, calibration, reject option and failures.

Usage:
    python predict.py                    # metrics, figures and failure candidates
    python predict.py --nodes 12 345     # also draw the 2-hop neighbourhood of these pages

For each model and each final seed, the temperature and the reject threshold are
fitted on validation logits only, frozen, and then applied to the test logits
(see ``calibrate.py``). Results are written to ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import textwrap
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import degree

from calibrate import (
    ECE_BINS,
    TARGET_ACCURACY,
    choose_threshold,
    ece,
    fit_temperature,
    reject_report,
)
from dataset import CLASS_NAMES, get_data, load_page_names
from modules import NodeClassifier, build_model
from train import CHECKPOINTS, RESULTS

ECE_LIMIT = 0.03  # AC2
COVERAGE_TARGET = 0.80  # AC3
LATENCY_REPEATS = 20
CONFIDENT = 0.9  # a wrong page at or above this confidence is "confidently wrong"
LOW_DEGREE = 2  # a page with this many links or fewer has little graph signal


def load_model(path: str, data: Data) -> NodeClassifier:
    """Rebuild a model from a checkpoint written by ``train.py final``.

    Args:
        path: Checkpoint path.
        data: Graph, used for the input and output sizes.

    Returns:
        The model in evaluation mode.
    """
    ckpt = torch.load(path, weights_only=True)
    cfg = ckpt["config"]
    model = build_model(
        cfg["model"],
        data.num_features,
        len(CLASS_NAMES),
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@torch.no_grad()
def forward(model: NodeClassifier, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the logits and the hidden embeddings of every node."""
    h = model.embed(data.x, data.edge_index)
    return model.classify(h, data.edge_index), h


def score_seed(logits: torch.Tensor, data: Data) -> tuple[dict, torch.Tensor]:
    """Fit calibration on validation, then score the test split once.

    Args:
        logits: Logits of every node from one trained model.
        data: Graph with real labels and split masks.

    Returns:
        The test metrics, and the calibrated probabilities of every node.
    """
    val, test = data.val_mask, data.test_mask
    temperature = fit_temperature(logits[val], data.y[val])
    probs_raw = logits.softmax(1)
    probs = (logits / temperature).softmax(1)
    threshold = choose_threshold(probs[val], data.y[val])

    y = data.y[test].numpy()
    pred = probs[test].argmax(1).numpy()
    precision, recall, _, _ = precision_recall_fscore_support(y, pred, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "auroc_macro_ovr": roc_auc_score(y, probs[test].numpy(), multi_class="ovr"),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "temperature": temperature,
        "ece_before": ece(probs_raw[test], data.y[test]),
        "ece_after": ece(probs[test], data.y[test]),
    }
    if threshold is None:
        metrics["reject"] = None
    else:
        metrics["reject"] = reject_report(probs[test], data.y[test], threshold, len(CLASS_NAMES))
    return metrics, probs


@torch.no_grad()
def latency(model: NodeClassifier, data: Data, repeats: int = LATENCY_REPEATS) -> float:
    """Return the median wall-clock time of one full-graph forward pass, in seconds."""
    for _ in range(3):  # warm-up
        model(data.x, data.edge_index)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(data.x, data.edge_index)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def summarise(runs: list[dict]) -> dict:
    """Return the mean and standard deviation over seeds of each scalar metric."""
    out = {}
    scalars = ("accuracy", "macro_f1", "auroc_macro_ovr", "temperature", "ece_before", "ece_after")
    for key in scalars:
        values = np.array([r[key] for r in runs])
        out[key] = {"mean": values.mean(), "std": values.std(ddof=1)}
    for key in ("precision", "recall"):
        values = np.array([r[key] for r in runs])
        out[key] = {"mean": values.mean(0).tolist(), "std": values.std(0, ddof=1).tolist()}
    rejects = [r["reject"] for r in runs if r["reject"] is not None]
    if rejects:
        for key in ("threshold", "coverage", "accuracy"):
            values = np.array([r[key] for r in rejects])
            out[f"reject_{key}"] = {"mean": values.mean(), "std": values.std(ddof=1)}
        out["reject_coverage_per_class"] = np.mean(
            [r["coverage_per_class"] for r in rejects], axis=0
        ).tolist()
    out["seeds_without_threshold"] = len(runs) - len(rejects)
    return out


def plot_reliability(before: torch.Tensor, after: torch.Tensor, y: torch.Tensor, path: str) -> None:
    """Draw reliability diagrams (accuracy against confidence) before and after scaling."""
    edges = np.linspace(0, 1, ECE_BINS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    titles = ("Before scaling", "After scaling")
    for ax, probs, title in zip(axes, (before, after), titles, strict=True):
        conf, pred = probs.max(1)
        conf, correct = conf.numpy(), (pred == y).numpy()
        accs = [
            correct[(conf > lo) & (conf <= hi)].mean() if ((conf > lo) & (conf <= hi)).any() else 0
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
        ax.bar(edges[:-1], accs, width=1 / ECE_BINS, align="edge", edgecolor="k", alpha=0.7)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.set(xlabel="confidence", title=f"{title} (ECE {ece(probs, y):.3f})", xlim=(0, 1))
        ax.legend(loc="upper left")
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tsne(embeddings: dict[str, torch.Tensor], y: torch.Tensor, path: str) -> None:
    """Draw a 2-D t-SNE of each model's hidden embeddings, coloured by true class."""
    fig, axes = plt.subplots(1, len(embeddings), figsize=(5 * len(embeddings), 4.5))
    for ax, (name, h) in zip(axes, embeddings.items(), strict=True):
        xy = TSNE(init="pca", random_state=0).fit_transform(h.numpy())
        for c, cls in enumerate(CLASS_NAMES):
            mask = (y == c).numpy()
            ax.scatter(xy[mask, 0], xy[mask, 1], s=3, alpha=0.6, label=cls)
        ax.set(title=f"{name.upper()} hidden layer (test pages)", xticks=[], yticks=[])
    axes[-1].legend(markerscale=4, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def failure_candidates(data: Data, probs: list[torch.Tensor], path: str) -> int:
    """List the test pages the GCN gets wrong for every seed, with a failure type.

    A page wrong for all five seeds is a stable failure, not bad luck. Each page
    is tagged with the rules below; the README discusses 3-5 of them.

    * ``wrong-neighbours``: fewer than half of its links go to its own class.
    * ``few-links``: degree ``LOW_DEGREE`` or less.
    * ``confident``: mean confidence ``CONFIDENT`` or more.

    Args:
        data: Graph with real labels and split masks.
        probs: Calibrated GCN probabilities of every node, one tensor per seed.
        path: Output CSV path.

    Returns:
        The number of pages listed.
    """
    names = load_page_names() or [""] * data.num_nodes
    stacked = torch.stack(probs)  # (seeds, nodes, classes)
    preds = stacked.argmax(2)
    always_wrong = (preds != data.y).all(0) & data.test_mask
    src, dst = data.edge_index
    deg = degree(src, data.num_nodes).long()
    rows = []
    for node in always_wrong.nonzero().flatten().tolist():
        neighbours = dst[src == node]
        counts = torch.bincount(data.y[neighbours], minlength=len(CLASS_NAMES))
        true, pred = int(data.y[node]), int(torch.mode(preds[:, node]).values)
        own_share = counts[true].item() / max(int(deg[node]), 1)
        conf = stacked[:, node].max(1).values.mean().item()
        tags = [
            tag
            for tag, hit in (
                ("wrong-neighbours", own_share < 0.5),
                ("few-links", deg[node] <= LOW_DEGREE),
                ("confident", conf >= CONFIDENT),
            )
            if hit
        ]
        rows.append(
            {
                "node": node,
                "page_name": names[node],
                "true": CLASS_NAMES[true],
                "predicted": CLASS_NAMES[pred],
                "mean_confidence": round(conf, 3),
                "degree": int(deg[node]),
                "own_class_share": round(own_share, 3),
                **{f"n_{c}": int(counts[i]) for i, c in enumerate(CLASS_NAMES)},
                "type": "+".join(tags) or "other",
            }
        )
    rows.sort(key=lambda r: -r["mean_confidence"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["node"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def plot_neighbourhood(data: Data, node: int, pred: int, path: str, cap: int = 40) -> None:
    """Draw a page's 2-hop neighbourhood, coloured by true class.

    Large neighbourhoods are cut to ``cap`` first-hop pages and 3 second-hop
    pages each (first in index order), so the picture stays readable.

    Args:
        data: Graph with real labels.
        node: Page to draw (shown large, with a black outline).
        pred: The class the model predicted for ``node``.
        path: Output PNG path.
        cap: Maximum number of first-hop pages drawn.
    """
    names = load_page_names() or [str(i) for i in range(data.num_nodes)]
    src, dst = data.edge_index
    hop1 = dst[src == node].tolist()[:cap]
    graph = nx.Graph()
    graph.add_node(node)
    for n1 in hop1:
        graph.add_edge(node, n1)
        for n2 in [n for n in dst[src == n1].tolist() if n != node][:3]:
            graph.add_edge(n1, n2)
    colours = [f"C{int(data.y[n])}" for n in graph.nodes]
    sizes = [400 if n == node else 120 if n in hop1 else 40 for n in graph.nodes]
    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.spring_layout(graph, seed=0)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colours, node_size=sizes)
    nx.draw_networkx_nodes(
        graph, pos, nodelist=[node], ax=ax, node_color=colours[0], node_size=400, edgecolors="k"
    )
    for c, cls in enumerate(CLASS_NAMES):
        ax.scatter([], [], color=f"C{c}", label=cls)
    ax.legend(loc="lower left")
    ax.set_title(
        f"{textwrap.fill(names[node], 50)}\n"
        f"true: {CLASS_NAMES[int(data.y[node])]}, predicted: {CLASS_NAMES[pred]}\n"
        f"degree {len(dst[src == node])} (showing {len(hop1)})"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Score every checkpoint and write metrics, figures and failure candidates."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nodes", type=int, nargs="*", default=[], help="pages to draw")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    data = get_data()
    test = data.test_mask
    report, gcn_probs, embeddings = {}, [], {}
    for model_name in ("gcn", "mlp"):
        paths = sorted(glob.glob(os.path.join(CHECKPOINTS, f"{model_name}_seed*.pt")))
        runs = []
        for i, path in enumerate(paths):
            model = load_model(path, data)
            logits, h = forward(model, data)
            metrics, probs = score_seed(logits, data)
            runs.append(metrics)
            if model_name == "gcn":
                gcn_probs.append(probs)
            if i == 0:  # figures and latency use the first final seed
                embeddings[model_name] = h[test]
                report[f"{model_name}_latency_seconds"] = latency(model, data)
                if model_name == "gcn":
                    plot_reliability(
                        logits[test].softmax(1),
                        probs[test],
                        data.y[test],
                        os.path.join(RESULTS, "reliability.png"),
                    )
            print(f"{model_name} {os.path.basename(path)}: F1 {metrics['macro_f1']:.4f}")
        report[model_name] = {"per_seed": runs, "summary": summarise(runs)}

    s = report["gcn"]["summary"]
    report["acceptance"] = {
        "AC2_ece": bool(s["ece_after"]["mean"] <= ECE_LIMIT),
        "AC3_reject": bool(
            s["seeds_without_threshold"] == 0
            and s["reject_coverage"]["mean"] >= COVERAGE_TARGET
            and s["reject_accuracy"]["mean"] >= TARGET_ACCURACY
        ),
        "AC5_latency": bool(report["gcn_latency_seconds"] <= 1.0),
    }
    with open(os.path.join(RESULTS, "predict.json"), "w") as f:
        json.dump(report, f, indent=2)

    plot_tsne(embeddings, data.y[test], os.path.join(RESULTS, "tsne.png"))
    n = failure_candidates(data, gcn_probs, os.path.join(RESULTS, "failures.csv"))
    mean_pred = torch.stack(gcn_probs).mean(0).argmax(1)
    for node in args.nodes:
        out = os.path.join(RESULTS, f"neighbourhood_{node}.png")
        plot_neighbourhood(data, node, int(mean_pred[node]), out)

    for model_name in ("gcn", "mlp"):
        m = report[model_name]["summary"]
        line = f"{model_name}: F1 {m['macro_f1']['mean']:.4f}, ECE {m['ece_before']['mean']:.4f}"
        line += f" -> {m['ece_after']['mean']:.4f}"
        if "reject_coverage" in m:
            line += f", coverage {m['reject_coverage']['mean']:.3f}"
            line += f" at accuracy {m['reject_accuracy']['mean']:.3f}"
        print(line)
    print(f"latency: gcn {report['gcn_latency_seconds'] * 1000:.1f} ms")
    print(f"acceptance: {report['acceptance']}")
    print(f"failure candidates (GCN wrong for every seed): {n}")


if __name__ == "__main__":
    main()
