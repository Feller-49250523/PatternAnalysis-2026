# Multi-layer GCN for Facebook Page-Page Node Classification

**Student number:** 49250523 · **Project:** 2.1 Classification · **Model:** multi-layer GNN (GCN) · **Difficulty:** Normal

> **Status (22 Oct 2026):** data pipeline, models and training script are complete. Tuning and the final test runs are complete (Sections 5.4, 7.1). Calibration, the reject option and the failure analysis (`predict.py`) are next. Sections marked **TODO** are filled in after the final runs.

## Contents

1. [Problem and algorithm](#1-problem-and-algorithm)
2. [Feasibility review](#2-feasibility-review)
3. [Data, preprocessing and split](#3-data-preprocessing-and-split)
4. [Models](#4-models)
5. [Training and hyperparameter tuning](#5-training-and-hyperparameter-tuning)
6. [Leakage controls](#6-leakage-controls)
7. [Results](#7-results)
8. [Usage](#8-usage)
9. [Dependencies](#9-dependencies)
10. [Code structure and style](#10-code-structure-and-style)
11. [Project log](#11-project-log)
12. [Limitations](#12-limitations)
13. [References](#13-references)
14. [Artificial Intelligence Usage Disclosure](#14-artificial-intelligence-usage-disclosure)

---

## 1. Problem and algorithm

**Problem.** The Facebook Large Page-Page network has 22,470 verified Facebook pages. Two pages are linked when they like each other. Each page has a 128-dimensional feature vector built from its description, and one of four categories: company, government, politician or TV show. The task is semi-supervised node classification: predict the category of pages whose labels are withheld. The engineering question is whether the model's **confidence** can be trusted, so that a triage system can auto-label the pages it is sure about and send the rest to a human.

**How a GCN works.** A graph convolutional network (Kipf & Welling, 2017) updates every page at once. In each layer, each page takes a degree-weighted average of its own features and its neighbours' features, then multiplies the result by a weight matrix that is shared by all pages:

$$H^{(l+1)} = \mathrm{ReLU}\left(\tilde D^{-1/2}\tilde A\,\tilde D^{-1/2} H^{(l)} W^{(l)}\right), \qquad \tilde A = A + I$$

The weight for the edge between pages $i$ and $j$ is $1/\sqrt{\tilde d_i \tilde d_j}$, so a hub page with many links has a smaller say in each neighbour's average. With $L$ layers, each page receives information from pages up to $L$ hops away. The last layer outputs 4 scores per page, and softmax turns them into class probabilities. Linked pages usually share a category (edge homophily 0.885), so averaging over neighbours removes noise from each page's own features.

**Figure.** TODO: a 2-hop neighbourhood of one page coloured by class, and a t-SNE plot of the learned embeddings.

---

## 2. Feasibility review

This review was checked off by the tutor on 22 Oct 2026. The version below has the same content, with the corrections listed under [Revisions since the check-off](#revisions-since-the-check-off).

### 2.1 User need, scope and acceptance criteria

**User:** a page-categorisation team that must classify pages whose features and links are known but whose category is withheld. **Purpose:** a triage assistant that auto-labels pages it is confident about and sends ambiguous pages to a human reviewer. The open question is whether confidence tracks correctness, so the criteria test calibration and the reject option, not only accuracy.

| # | Acceptance criterion (on the held-out test split) |
|---|---|
| AC1 | Macro-F1 ≥ 0.90 and ≥ 0.05 above the MLP baseline on the same split. |
| AC2 | Expected calibration error (top-label ECE, 15 equal-width bins) ≤ 0.03 after temperature scaling fitted on validation only. |
| AC3 | Reject rule: choose the threshold with the highest validation coverage subject to ≥ 97 % validation accuracy, then freeze it. On test: auto-label ≥ 80 % of pages at ≥ 97 % accuracy. |
| AC4 | No leakage: train/validation/test masks are disjoint; only training labels enter the loss; every model, calibration and threshold choice uses validation results. |
| AC5 | Resources: one full training ≤ 10 min and peak memory ≤ 4 GB on a laptop CPU; full-graph inference ≤ 1 s. |

### 2.2 Model choice and course concepts

**Data:** the course-specified release (`gayanku/facebook_page-page_dataset`, `facebook.npz`): 22,470 pages, 170,823 undirected mutual-like edges after removing 179 self-loops, 4 classes, 128-dimensional SVD features. **Setting:** transductive semi-supervised classification on a stratified, seeded 70/15/15 node split. All features and edges are visible during message passing; only training labels are used for learning. **Model:** a multi-layer GCN with dropout and residual connections. **Baseline:** a 2-layer MLP on the same features with no graph. **Course link:** GCN layers are a first-order approximation of spectral graph filters, i.e. the Fourier and convolution ideas from the course applied to a graph Laplacian. Connected pages share a label on 88.5 % of edges, which supports neighbour averaging. Calibration uses temperature scaling (Guo et al., 2017).

### 2.3 Preliminary evidence (smoke test, 21 Oct 2026, seed 42, Apple M4 CPU)

No isolated nodes and no NaNs. Median degree 7, maximum 709. Class counts 6,495 / 6,880 / 5,768 / 3,327 (TV show is the minority, 15 %). The smoke test compared the MLP with three 2-layer GNNs (200 epochs, no tuning). ECE and confidence are **before** calibration.

| Model | Test acc. | Macro-F1 | AUROC (macro OvR) | ECE (uncalibrated) | Mean confidence right / wrong | Coverage at confidence ≥ 0.9 → accuracy | Params | Train time |
|---|---|---|---|---|---|---|---|---|
| MLP | 0.797 | 0.775 | 0.947 | 0.041 | 0.83 / 0.57 | 45 % → 0.964 | 8.5 k | 1.5 s |
| GCN | 0.928 | 0.924 | 0.988 | 0.035 | 0.91 / 0.68 | 68 % → 0.979 | 8.5 k | 12 s |
| GAT | 0.936 | 0.932 | 0.990 | 0.021 | 0.93 / 0.72 | 76 % → 0.981 | 8.7 k | 23 s |
| GIN | 0.939 | 0.935 | 0.991 | 0.018 | 0.95 / 0.74 | 83 % → 0.979 | 16.8 k | 15 s |

"Confidence" is the highest predicted class probability; "coverage" is the fraction of pages auto-labelled. Peak memory was 0.88 GB. **Takeaways:** the graph adds 14.9–16.0 macro-F1 points over the MLP. GAT and GIN beat the GCN by only 0.8 and 1.1 points, from one seed, at up to 1.9× the training time. These test scores are exploratory: they informed the design, so the final study uses a fresh split (Section 3.3).

### 2.4 Risks, budget and fallback

- **Data quirks:** the dataset README gives the wrong label order. The labels are alphabetical (company, government, politician, tvshow), verified for all 22,470 pages against the MUSAE target file. The file holds 179 self-loops (358 entries), which are removed. The pickled `page_name` array is skipped for safety; names come from the MUSAE CSV.
- **Over-smoothing** in deeper GCNs can blur class boundaries; depth is tuned and residual connections are used.
- **Small gaps:** results are reported as mean ± standard deviation over 5 training seeds. TV show has the lowest F1 for every model; the failure analysis studies 3–5 such pages.
- **Budget:** each run takes seconds to minutes on a laptop CPU, so no cluster is needed.

### Revisions since the check-off

| Change | Reason |
|---|---|
| Scope reduced to **GCN vs MLP** (GAT and GIN dropped) | The GCN passes AC1. GAT and GIN gained ≤ 1.1 points from one seed at up to 1.9× the cost, so the GCN is the cheapest model that meets the need. Their smoke-test rows stay as evidence for this choice. |
| "358 self-loops" → **179 self-loops (358 entries)** | Each self-loop appears twice in the edge list. The edge count (170,823) was already correct. |
| AC2, AC3, AC4 reworded | AC3 now states the exact threshold rule; AC4 states what can be checked in code. (Suggested by an AI review, see Section 14.) |
| User need reworded | "Newly verified pages" implied an inductive setting; the study is transductive. |
| Table labels | ECE and confidence are marked as uncalibrated; confidence and coverage are defined. |
| Homophily claim narrowed | "Message passing assumes homophily" overstated the theory; the measured 88.5 % is now given as support. |

---

## 3. Data, preprocessing and split

### 3.1 Source

`facebook.npz` from [gayanku/facebook_page-page_dataset](https://github.com/gayanku/facebook_page-page_dataset), built from MUSAE (Rozemberczki et al., 2021). `dataset.py` downloads it once and checks its SHA-256 hash. It is not committed.

| Array | Shape | Content |
|---|---|---|
| `X` | 22,470 × 128 | SVD features of page descriptions |
| `edges_x`, `edges_y` | 342,004 | directed pairs (both directions of each link) |
| `y` | 22,470 | labels 0–3 |
| `page_name` | 22,470 | **skipped**: pickled object array |

### 3.2 Preprocessing

| Step | Why |
|---|---|
| `np.load(..., allow_pickle=False)` | A pickled array can run code when loaded. Only numeric arrays are read. |
| Remove 179 self-loops (358 entries) | A page liking itself is not a relation. GCNConv adds its own self-loop in a controlled way. |
| `to_undirected` (also merges duplicates) | Links are mutual likes; both directions must exist. Result: 170,823 undirected edges. |
| No feature scaling | The SVD features are already dense and centred. |
| Check for NaNs and size mismatches | Raises `ValueError` instead of training on bad data. |

**Labels:** the dataset README's label order matches page types for only 31 % of pages. Alphabetical order (0 company, 1 government, 2 politician, 3 tvshow) matches 100 %.

### 3.3 Split

**Setting: transductive.** All 22,470 pages and all edges are present during training, as in a real platform where every page exists but only some are labelled. Only training labels enter the loss.

| Split | Pages | Use |
|---|---|---|
| Train (70 %) | 15,729 | loss |
| Validation (15 %) | 3,370 | early stopping, hyperparameter choice, temperature, reject threshold |
| Test (15 %) | 3,371 | final report only, once per final model |

- **Stratified**, so every split has the same class ratios (TV show 14.8 % in each). A random split without stratification would vary by about ±21 TV-show pages in the test set by chance.
- **Shuffled with a fixed split seed**, so the split does not depend on the file order and is the same in every script and every run. `train.py` and `predict.py` each rebuild the split; a fixed seed makes sure they agree.
- **Fresh split seed (3710).** The smoke test used seed 42 and its test scores informed the design, so the final study uses a new split, fixed before any final result was seen. By chance, 491 of the 3,371 new test pages were also in the old test set; only aggregate scores were seen for those.
- **Split seed ≠ model seed.** The split seed decides *which pages* are in each split and stays fixed. Model seeds decide the start weights, dropout and DropEdge patterns, and change between runs.

### 3.4 Augmentation

`augment()` implements DropEdge (Rong et al., 2020): on each training step it removes a random fraction of edges, both directions together. It is used on training passes only and never touches labels. Whether it helps is a tuned hyperparameter (Section 5).

---

## 4. Models

Both models implement one interface (`NodeClassifier`): `forward(x, edge_index) = classify(embed(x, edge_index))`. `embed` returns the 64-dimensional hidden representation used for t-SNE.

| Model | Architecture | Params |
|---|---|---|
| MLP (baseline) | 128 → 64 → 4, ReLU, dropout; ignores edges | 8,516 |
| GCN-2 | GCNConv 128 → 64 → 4 | 8,516 |
| GCN-3 | 128 → 64 → [64 → 64 + residual] → 4 | 12,676 |
| GCN-4 | 128 → 64 → [64 → 64 + residual] × 2 → 4 | 16,836 |
| GCN-8 | 128 → 64 → [64 → 64 + residual] × 6 → 4 | 33,476 |

The MLP has the same width, depth and dropout as GCN-2, so the gap between them measures the value of the graph. Residual connections between hidden layers ($h \leftarrow h + \mathrm{conv}(h)$, Kipf & Welling, App. B) let deeper models keep each page's own signal instead of averaging it away (over-smoothing). The 2-layer models have no input dropout, to match the smoke test.

---

## 5. Training and hyperparameter tuning

### 5.1 Training

Full-batch training: one epoch is one step (forward pass over the whole graph, loss on training pages, backpropagation, one Adam update). Cross-entropy loss. Early stopping on validation loss; the weights from the lowest validation loss are restored.

### 5.2 Sources for every hyperparameter

Tuned on validation (grid):

| Hyperparameter | Values | Source |
|---|---|---|
| Layers (GCN) | 2, 3, 4, 8 | Kipf & Welling: 2–3 best, tested up to 10 with residuals; Rong et al.: best GCN depth 4, tested 2, 4, 8, … |
| Dropout | 0.5, 0.8 | Kipf & Welling: 0.5; Shchur et al.: best GCN and best MLP 0.8 |
| DropEdge rate *p* (GCN) | 0, 0.3 | Kipf & Welling: none; Rong et al.: best GCN on Cora keeps 70 % of edges |
| Learning rate | 0.01, 0.005 | Shchur et al.: best GCN 0.01, best MLP 0.005 |

The MLP gets the same dropout and learning-rate grid (Shchur et al.: the same tuning procedure for every model), so the baseline is not weakened by less tuning.

Fixed:

| Setting | Value | Source |
|---|---|---|
| Hidden size | 64 | Shchur et al.: best GCN |
| L2 weight decay | 5 × 10⁻⁴ | Kipf & Welling |
| Optimiser | Adam | all three papers |
| Early stopping | patience 50 on validation loss, max 100,000 epochs, restore best weights | Shchur et al., App. C |
| Initialisation | Glorot | Kipf & Welling (PyG default) |

Not tuned, with reasons: **epochs** are set by early stopping in every run; the **optimiser** is Adam in every reference paper and comparing optimisers does not answer the project question.

### 5.3 Protocol

1. **Tune:** every setting is trained with model seeds 0, 1, 2. The setting with the highest **mean** validation macro-F1 wins (Shchur et al. also select by mean validation score). Averaging over 3 seeds stops a setting from winning by luck.
2. **Final:** the winning setting of each model is trained with **new** seeds 3–7 and scored on test once per seed. Results are reported as mean ± standard deviation.

### 5.4 Tuning history

**v1 (22 Oct, superseded).** 24 GCN + 4 MLP settings × 3 seeds, fixed 200 epochs (Kipf & Welling). Full results: [`results/tuning_v1.json`](results/tuning_v1.json).

| Validation macro-F1 (lr 0.01) | dropout 0.5, *p* 0 | 0.5, 0.3 | 0.8, 0 | 0.8, 0.3 |
|---|---|---|---|---|
| GCN-2 | 0.923 | 0.921 | 0.920 | 0.916 |
| GCN-3 | 0.942 | 0.937 | 0.938 | 0.930 |
| GCN-4 | **0.948** | 0.941 | 0.945 | 0.935 |
| MLP | 0.780 | – | 0.768 | – |

Two problems led to v2:

1. **The epoch limit cut training short.** Almost every run had its best epoch between 185 and 200, so validation loss was still falling at the limit. This also biased the grid towards fast-learning settings (lr 0.01, dropout 0.5). Fix: Shchur et al.'s early stopping (patience 50, max 100,000 epochs).
2. **The best depth was at the edge of the grid** (2 → 3 → 4 still rising). Fix: add depth 8, the next value tested by Rong et al.

These changes were based on validation results only; test labels stayed hidden (Section 6).

**v2 (22 Oct, final).** 32 GCN + 4 MLP settings × 3 seeds with early stopping (patience 50). Total 136 min on the M4 CPU. Full results: [`results/tuning.json`](results/tuning.json), log: [`results/tuning_v2.log`](results/tuning_v2.log).

| Validation macro-F1 (lr 0.01, mean ± std of 3 seeds) | dropout 0.5, *p* 0 | 0.5, 0.3 | 0.8, 0 | 0.8, 0.3 |
|---|---|---|---|---|
| GCN-2 | 0.927 ± 0.002 | 0.924 | 0.923 | 0.917 |
| GCN-3 | 0.947 ± 0.002 | 0.940 | 0.942 | 0.934 |
| GCN-4 | **0.952 ± 0.001** | 0.946 | 0.948 | 0.938 |
| GCN-8 | 0.951 ± 0.001 | 0.947 | 0.948 | 0.942 |
| MLP | 0.785 ± 0.001 | – | 0.772 | – |

What v2 shows:

1. **Early stopping fixed the cut-off.** Best epochs are now 226–992, all well past the old limit of 200 and far below the 100,000 cap. Every score rose a little (GCN-4: 0.948 → 0.952; MLP: 0.780 → 0.785).
2. **Depth peaks at 4.** GCN-8 is not better than GCN-4 (0.951 vs 0.952), so the best depth is no longer at the edge of the grid. GCN-8 also takes about 2× the time per run.
3. **DropEdge (*p* = 0.3) and dropout 0.8 lowered every GCN row**, the same as in v1.
4. **Learning rate does not matter much.** In each pair, lr 0.005 is within 0.002 of lr 0.01, but it needs about 1.5× more epochs.

**Chosen settings** (highest mean validation macro-F1, unchanged from v1):

| Model | Layers | Dropout | DropEdge *p* | lr | Validation macro-F1 |
|---|---|---|---|---|---|
| GCN | 4 | 0.5 | 0 | 0.01 | 0.952 ± 0.001 |
| MLP | 2 | 0.5 | 0 | 0.01 | 0.785 ± 0.001 |

The gap between GCN and MLP (+0.167 macro-F1) is the value the graph edges add. Tuning stops here: the next changes would be chosen after looking at these validation scores again, which adds more selection bias (Section 12).

---

## 6. Leakage controls

| Control | Where | What happens on a violation |
|---|---|---|
| Disjoint masks that cover every node | `dataset.check_split` | `ValueError` |
| Test labels replaced by −1 during training and tuning | `train.hide_test_labels` | `cross_entropy` raises `IndexError` if a test page enters the loss |
| No scoring of hidden labels | `train.evaluate` | `ValueError` |
| Test scored only in `final`, once per seed, after all choices are frozen | `train.final` | – |
| Temperature and threshold fitted on validation | TODO (`train.py`) | – |
| Fresh split seed after the exploratory smoke test | `dataset.SPLIT_SEED` | – |

In the transductive setting, test pages' **features and edges** are visible during message passing; their **labels** are not. This is the standard protocol (Kipf & Welling, 2017).

---

## 7. Results

### 7.1 Test macro-F1 (AC1)

The tuned setting of each model (Section 5.4) was trained with new seeds 3–7 and scored on the test split once per seed. Full output: [`results/final.json`](results/final.json), [`results/final.log`](results/final.log).

| Model | Test macro-F1 (mean ± std, 5 seeds) | Per seed | Params | Train time / run | Best epochs |
|---|---|---|---|---|---|
| GCN-4 | **0.945 ± 0.002** | 0.943, 0.947, 0.942, 0.946, 0.946 | 16,836 | 53 s | 185–351 |
| MLP | 0.776 ± 0.004 | 0.781, 0.779, 0.773, 0.771, 0.778 | 8,516 | 4 s | 350–556 |

**AC1 passes:** 0.945 ≥ 0.90, and the GCN is 0.168 above the MLP (target ≥ 0.05). The worst GCN seed (0.942) is still 0.161 above the best MLP seed (0.781). Test is about 0.007 below validation (0.952), a small drop, as expected after selecting on validation.

![Training curves](results/training_curves.png)

**Figure.** Loss and validation macro-F1 for the 5 final seeds. Seeds agree closely. The GCN validation loss flattens by about epoch 200 and early stopping ends each run 50 epochs after its best epoch (vertical lines). The gap between train and validation loss stays small, so there is little overfitting.

### 7.2 Resources (AC5, partial)

Training: 53 s per GCN run (limit 10 min). Peak memory for the whole final stage: 0.87 GB (limit 4 GB). Inference latency: TODO (`predict.py`).

### 7.3 Still to do

- Calibration: ECE before and after temperature scaling, reliability diagram.
- Reject option: coverage and accuracy at the validation-chosen threshold, including per-class coverage.
- t-SNE of the GCN embeddings, coloured by true class.
- Inference latency.
- Accuracy, per-class precision and recall, AUROC.
- Failure analysis of 3–5 misclassified pages.
- Recommendation to the triage team.

---

## 8. Usage

```bash
pip install torch==2.14.1 torch_geometric==2.8.1 scikit-learn==1.9.1 matplotlib==3.11.2

python dataset.py        # download, verify and audit the data
python train.py tune     # grid search on validation (writes results/tuning.json)
python train.py final    # 5-seed final runs, test scores, checkpoints, curves
python predict.py        # TODO: metrics, calibration, reject rule, t-SNE
```

Example output of `python dataset.py`:

```
           nodes: 22470
undirected_edges: 170823
        features: 128
    class_counts: {'company': 6495, 'government': 6880, 'politician': 5768, 'tvshow': 3327}
  edge_homophily: 0.885
  isolated_nodes: 0
   degree_median: 7
      degree_max: 709
     split_sizes: {'train': 15729, 'val': 3370, 'test': 3371}
```

Optional: put the MUSAE `facebook_target.csv` in `data/` to show page names in the failure analysis.

---

## 9. Dependencies

Tested on macOS 27.0.1, Apple M4, CPU only.

| Package | Version |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.14.1 |
| PyTorch Geometric | 2.8.1 |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| Ruff (style check) | 0.14.0 |

---

## 10. Code structure and style

```
GNN_49250523/
├── dataset.py      download, clean, split, leakage check, DropEdge, audit
├── modules.py      NodeClassifier interface, MLP, GCN, build_model (PyTorch only, no NumPy)
├── train.py        sourced grid, test-label lock, training, tuning, final runs, curves
├── predict.py      TODO
├── results/        tuning and final results (JSON, logs, plots)
└── pyproject.toml  Ruff configuration
```

**Style:** PEP 8 enforced by Ruff (`ruff check . && ruff format --check .`), Google-style docstrings, type hints on every function. Data checks raise exceptions rather than using `assert`, because Python removes `assert` under `-O`.

**Design (SOLID):** each function has one job (single responsibility). Models share one interface, so `train.py` works with any `NodeClassifier` (Liskov substitution, dependency inversion). A new model is added by one class and one entry in `MODELS`, without changing other code (open/closed).

---

## 11. Project log

| Date | Step | Notes |
|---|---|---|
| 21 Oct | Fork, branch `topic-recognition`, project folder | |
| 21 Oct | Feasibility smoke test and review | Found the wrong label order in the dataset README; switched from a self-built feature set to the course-specified `facebook.npz`. |
| 22 Oct | Review extended to GCN, GAT and GIN | GAT and GIN are covered in the lectures. |
| 22 Oct | Tutor check-off passed | |
| 22 Oct | Review critiqued by a second AI tool | Found the self-loop miscount and the "test no longer untouched" issue. |
| 22 Oct | Scope: GCN vs MLP | See Section 2, revisions. |
| 22 Oct | Fresh split seed 3710 | See Section 3.3. |
| 22 Oct | `dataset.py`, `modules.py` committed | Ruff, type hints, Google docstrings. |
| 22 Oct | Hyperparameter sources | Every grid value traced to Kipf & Welling, Shchur et al. or Rong et al. Dropout 0.3 and DropEdge 0.2 replaced by sourced values 0.8 and 0.3. Learning rate added to the grid for fairness to the MLP. |
| 22 Oct | Tuning v1 → v2 | See Section 5.4. |
| 22 Oct | Tuning v2 complete | GCN-4, dropout 0.5, no DropEdge, lr 0.01 chosen (validation macro-F1 0.952). Depth 8 did not beat depth 4. |
| 22 Oct | Final runs (seeds 3–7) | GCN test macro-F1 0.945 ± 0.002, MLP 0.776 ± 0.004. AC1 passes. |

---

## 12. Limitations

- **One split.** Shchur et al. use 100 splits × 20 seeds and show that one fixed split can mislead. This study uses 1 split × 5 seeds because of time; split variation is not measured.
- **Transductive only.** Results apply to this fixed graph, not to pages that join later.
- **Validation reuse.** The validation set guided early stopping, the grid, the v1 → v2 change, the temperature and the threshold, so validation scores are slightly optimistic. The test set was used for none of these.
- **Method of a source.** Rong et al. selected their reported settings by best *test* accuracy. Only their values are used here, not their selection method.

---

## 13. References

- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
- Shchur, O., Mumme, M., Bojchevski, A., & Günnemann, S. (2018). Pitfalls of graph neural network evaluation. *NeurIPS R2L Workshop*. [arXiv:1811.05868](https://arxiv.org/abs/1811.05868)
- Rong, Y., Huang, W., Xu, T., & Huang, J. (2020). DropEdge: Towards deep graph convolutional networks on node classification. *ICLR*. [arXiv:1907.10903](https://arxiv.org/abs/1907.10903)
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Rozemberczki, B., Allen, C., & Sarkar, R. (2021). Multi-scale attributed node embedding. *Journal of Complex Networks*.
- Veličković, P., et al. (2018). Graph attention networks. *ICLR*. (smoke test only)
- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *ICLR*. (smoke test only)

---

## 14. Artificial Intelligence Usage Disclosure

| Tool | Used for | How I verified it |
|---|---|---|
| Claude Code (Claude Opus 5.5, Anthropic) | Drafting `dataset.py`, `modules.py`, `train.py` and this README; the data audit (found the wrong label order and the self-loops); extracting hyperparameters from the three reference papers; explaining concepts (seeds, message passing, backpropagation, leakage) | Every file passes Ruff and was run before each commit. Tests: split disjointness, a planted split overlap is caught, test labels cannot enter the loss, parameter counts match hand calculations, labels match the MUSAE file for all pages. Hyperparameter values were taken from text extracted from the paper PDFs, with the passages quoted. I reviewed each commit before it was made and questioned the design choices (seeds, split, tuning, leakage) until I could explain them. TODO: confirm before submission that I have read the quoted passages myself. |
| OpenAI Codex CLI 0.163.0 | A critique of the feasibility review | I checked each point against the data and code. Accepted: the self-loop count, test-set reuse, uncalibrated ECE label, transductive wording, threshold rule. Rejected: removing the t-SNE plot (the spec requires it). |

AI-assisted commits carry a `Co-Authored-By` trailer.
