"""Data loading, preprocessing and leakage-free splitting for Facebook Page-Page.

Source: the course-specified release github.com/gayanku/facebook_page-page_dataset
(``facebook.npz``), built from MUSAE (Rozemberczki et al., 2021).

Setting: transductive semi-supervised node classification. The model sees the
features and edges of every node during message passing, but only the labels of
training nodes. Validation labels are used for early stopping, temperature
scaling and threshold selection; test labels are used only for the final report.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected

URL = "https://raw.githubusercontent.com/gayanku/facebook_page-page_dataset/main/facebook.npz"
SHA256 = "7a1b8a0687515e9859d1ca649dc276825f5db44568894dfbb6b55ebc1cf6a9c1"
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# The dataset README lists a different order, but that order is wrong: the labels
# are alphabetical (checked against page_type in the MUSAE facebook_target.csv
# for all 22,470 pages).
CLASS_NAMES = ["company", "government", "politician", "tvshow"]

# Fixed before any final result was seen. The feasibility smoke test used seed 42
# and its test scores informed the design, so the final study uses a fresh split
# whose test nodes have never been evaluated.
SPLIT_SEED = 3710

SPLITS = ("train", "val", "test")


def download(root: str = DEFAULT_ROOT) -> str:
    """Download ``facebook.npz`` if it is missing, then verify its hash.

    Args:
        root: Directory that holds the dataset file.

    Returns:
        Path to the verified ``facebook.npz``.

    Raises:
        ValueError: If the file does not match the expected SHA-256 hash.
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "facebook.npz")
    if not os.path.exists(path):
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path}: SHA-256 mismatch (got {digest}). Delete it and retry.")
    return path


def load_facebook(root: str = DEFAULT_ROOT) -> Data:
    """Load and clean the graph.

    Preprocessing:
      * ``allow_pickle=False``: only numeric arrays are read. The ``page_name``
        array is a pickled object array, which can run code when loaded, so it
        is skipped (see ``load_page_names``).
      * Self-loops are removed (179 pages linked to themselves, each listed
        twice = 358 entries). GCNConv adds its own self-loops when it runs.
      * Edges are made undirected and duplicate pairs are merged
        (``to_undirected`` also coalesces), giving 170,823 undirected edges.
      * Features are the 128-dim SVD vectors as given. They are already dense and
        centred, so no further scaling is applied.

    Args:
        root: Directory that holds the dataset file.

    Returns:
        Graph with ``x`` (22470 x 128), ``edge_index`` (2 x 341646) and ``y``.

    Raises:
        ValueError: If the arrays have inconsistent sizes or contain NaNs.
    """
    arrays = np.load(download(root), allow_pickle=False)
    edge_index = torch.from_numpy(np.stack([arrays["edges_x"], arrays["edges_y"]])).long()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)
    x = torch.from_numpy(arrays["X"]).float()
    y = torch.from_numpy(arrays["y"].astype(np.int64))
    if x.size(0) != y.size(0):
        raise ValueError(f"{x.size(0)} feature rows but {y.size(0)} labels")
    if torch.isnan(x).any():
        raise ValueError("node features contain NaNs")
    return Data(x=x, edge_index=edge_index, y=y)


def make_split(
    y: torch.Tensor,
    split_seed: int = SPLIT_SEED,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Make a stratified, seeded train/val/test split of the nodes.

    Stratified, so each split keeps the class ratios (TV show is only 15 %).
    The split seed is separate from the model seed: keep it fixed so that all
    models and all training seeds are compared on the same test nodes.

    Args:
        y: Node labels, shape ``(num_nodes,)``.
        split_seed: Seed for the shuffle inside the split.
        val_frac: Fraction of nodes for validation.
        test_frac: Fraction of nodes for testing.

    Returns:
        Boolean masks keyed by ``"train"``, ``"val"`` and ``"test"``.
    """
    n = y.size(0)
    idx = np.arange(n)
    labels = y.numpy()
    train, rest = train_test_split(
        idx, test_size=val_frac + test_frac, stratify=labels, random_state=split_seed
    )
    val, test = train_test_split(
        rest,
        test_size=test_frac / (val_frac + test_frac),
        stratify=labels[rest],
        random_state=split_seed,
    )
    masks = {}
    for name, ids in zip(SPLITS, (train, val, test), strict=True):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[torch.from_numpy(ids)] = True
        masks[name] = mask
    check_split(masks)
    return masks


def check_split(masks: dict[str, torch.Tensor]) -> None:
    """Check that the splits are disjoint and cover every node.

    Args:
        masks: Boolean masks keyed by ``"train"``, ``"val"`` and ``"test"``.

    Raises:
        ValueError: If two splits share a node or a node is in no split.
    """
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            shared = int((masks[a] & masks[b]).sum())
            if shared:
                raise ValueError(f"{a} and {b} share {shared} nodes")
    if not (masks["train"] | masks["val"] | masks["test"]).all():
        raise ValueError("some nodes are in no split")


def get_data(root: str = DEFAULT_ROOT, split_seed: int = SPLIT_SEED) -> Data:
    """Load the graph and attach ``train_mask``, ``val_mask`` and ``test_mask``.

    Args:
        root: Directory that holds the dataset file.
        split_seed: Seed for the train/val/test split.

    Returns:
        Graph with the three split masks attached.
    """
    data = load_facebook(root)
    for name, mask in make_split(data.y, split_seed=split_seed).items():
        data[f"{name}_mask"] = mask
    return data


if __name__ == "__main__":
    data = get_data()
    print(data)
    print({s: int(data[f"{s}_mask"].sum()) for s in SPLITS})
