# Texture-Based Image Segmentation

**Statistical texture representation and unsupervised segmentation using GLCM,
Haralick features, Singular Value Decomposition, and K-Means clustering.**

Course project for **GNR 602 — Advanced Methods for Satellite Image Processing**,
Indian Institute of Technology Bombay.

---

## Overview

This project implements, entirely from scratch in NumPy, a complete pipeline for
unsupervised texture-based image segmentation. Two competing texture representations
derived from the same Gray-Level Co-occurrence Matrix (GLCM) are compared:

- **Haralick features** — 13 hand-crafted scalars (Energy, Contrast, Correlation,
  Entropy, Homogeneity, etc.) capturing distinct aspects of texture structure.
- **SVD features** — top-*k* singular values of the GLCM, treating it as a matrix
  and reading off its low-rank structure in a purely data-driven way.

Both are evaluated against a pre-trained **ResNet-18 CNN baseline** to situate
classical texture statistics relative to modern deep features.

Every algorithmic primitive (Otsu quantisation, GLCM, Haralick, SVD, K-Means,
ARI, Silhouette, DBI) is verified against an independent library implementation
to confirm correctness before use in the pipeline.

---

## Pipeline

```
                          Input image (grayscale)
                                 │
                ┌────────────────┴──────────────────┐
                │                                   │
                ▼                                   ▼
   Multi-level Otsu quantisation         Per-window raw patches
   (L gray levels, recursive binary)     (32×32, replicate to 3ch,
                │                         upsample 64×64, ImageNet norm)
                ▼                                   │
   Sliding-window GLCM                              ▼
   (W×W windows, step S,                  CNN forward pass
    rotation-invariant 4-angle average)   (ResNet-18, FC layer removed,
                │                          global avg pool → 512-dim)
                ├──► Haralick features                │
                │    (13-dim per window,              │
                │     optionally multi-distance       │
                │     d ∈ {1,3,5})                    │
                │                                     │
                └──► SVD features                     │
                     (top-k singular values           │
                      of GLCM, 8-dim per window)      │
                              │                       │
                              └──────────┬────────────┘
                                         ▼
                       Z-score standardisation
                       [+ optional PCA decorrelation]
                                         │
                                         ▼
                   K-Means clustering (k-means++ init,
                   10 restarts, empty-cluster reseeding)
                                         │
                                         ▼
                   Hungarian label alignment
                   (optimal permutation matching to ground truth)
                                         │
                                         ▼
                   Segmentation map
                   (closed-form nearest-window-centre pixel assignment)
                                         │
                                         ▼
                   Evaluation: ARI · Silhouette · Davies-Bouldin
                              · Pixel accuracy · Per-class IoU
```

---

## Key findings

Headline results across the three reference configurations. Higher ARI and
Silhouette are better; lower DBI is better.

| Experiment | ARI | Silhouette | DBI | Pixel acc |
|---|---:|---:|---:|---:|
| Collage A — Haralick, W=32, d=1 (baseline) | 0.526 | 0.526 | 0.664 | 76.3% |
| Collage A — SVD, W=32, d=1 | 0.428 | 0.376 | 0.930 | 73.4% |
| Collage A — CNN ResNet-18 | 0.671 | 0.100 | 2.597 | 85.9% |
| Collage A — Haralick, W=64, d={1,3,5}, PCA (**best hand-crafted**) | **0.775** | 0.526 | 0.664 | **93.4%** |
| texmos2 — Haralick, W=32, d=1 | 0.089 | 0.242 | 1.213 | 33.1% |
| texmos2 — SVD, W=32, d=1 | 0.106 | 0.226 | 1.306 | 28.2% |
| texmos2 — CNN ResNet-18 (**best on texmos2**) | **0.251** | 0.089 | 2.760 | ~40% |
| EuroSAT 2×2 — Haralick | 0.215 | 0.482 | 0.752 | 53.2% |
| EuroSAT 2×2 — CNN ResNet-18 (**best on EuroSAT**) | **0.223** | 0.147 | 2.241 | 50.1% |

- **Hand-crafted Haralick beats CNN** on the Brodatz benchmark (ARI 0.775 vs 0.671)
  when window size and multi-distance features are tuned with PCA.
- **Window size is the biggest lever**: doubling W from 32 to 64 raises Collage A
  Haralick ARI from 0.526 to 0.663 — a larger gain than any feature-engineering
  experiment.
- **CNN wins where intra-class variability is high** (texmos2 fine sub-regions,
  EuroSAT real satellite imagery). Hand-crafted statistics struggle when the same
  class looks very different in different patches.
- **Internal and external metrics can disagree**: CNN has the best ARI but the
  worst Silhouette and DBI — geometrically tangled clusters that still align with
  ground truth. Haralick on Collage A has the *cleanest* clusters by Silhouette
  and DBI even at the unoptimised baseline.

---

## Repository layout

```
Texture-Based-Image-Segmentation/
├── main.ipynb          # Complete pipeline — all code lives here (10 sections)
├── app.py              # Streamlit interactive UI
├── pipeline.py         # Pipeline functions re-exported for the UI
├── requirements.txt    # Python dependencies
├── textures/           # USC-SIPI Brodatz tiles + texmos mosaics (committed)
│   ├── 1.1.01.tiff … 1.5.07.tiff
│   ├── texmos2.p512.tiff, texmos2.s512.tiff   (8-class benchmark + GT)
│   └── texmos3.p512.tiff, texmos3.s512.tiff
├── data/               # Generated data — gitignored
│   ├── brodatz/
│   ├── collage/
│   └── eurosat/        # EuroSAT RGB patches (download separately — see below)
└── USCTextureMosaics.pdf   # USC documentation for texmos mosaics
```

---

## Notebook structure

The notebook is organised into 10 labelled sections, each self-contained:

| Section | Content |
|---|---|
| 1 | Imports and global parameters (single source of truth) |
| 2 | Dataset construction — Brodatz 2×2 collage and texmos2 benchmark |
| 3 | Multi-level Otsu quantisation (from scratch + verification) |
| 4 | GLCM construction — sliding windows, rotation invariance (from scratch + verification) |
| 5 | 13 Haralick features (from scratch + verification vs mahotas) |
| 6 | SVD features — top-k singular values, cumulative energy justification |
| 7 | Standardisation + K-Means (k-means++, multi-restart, from scratch + verification) |
| 8 | Segmentation maps, disagreement overlays, interior vs boundary accuracy |
| 9 | ARI, Silhouette, DBI — from scratch + verification vs sklearn |
| 10 | Extensions: multi-distance GLCM, window sweep, CNN baseline, EuroSAT, grid search |

Global parameters (window size, gray levels, K, random seed) are defined once
in Section 1. Change there and re-run all cells to reproduce any experiment.

---

## Evaluation metrics

Three independent metrics are reported for every experiment so that no single
number drives the conclusion. All three are implemented from scratch in Section 9
and verified against scikit-learn to within 10⁻¹⁰ absolute error.

| Metric | Range | Better when | What it measures |
|---|---|---|---|
| **Adjusted Rand Index (ARI)** | [−1, 1] | **higher** | How well the predicted clusters agree with the ground-truth labels, *corrected for chance*. ARI = 0 means agreement no better than random; ARI = 1 means a perfect match (up to label permutation). The Hungarian algorithm is used beforehand to align cluster IDs to GT classes. |
| **Silhouette coefficient** | [−1, 1] | **higher** | An *unsupervised* score that compares each point's distance to its own cluster (cohesion) against its distance to the nearest other cluster (separation). High Silhouette = tight, well-separated clusters. Computed entirely from the feature vectors — does not look at ground truth. |
| **Davies–Bouldin Index (DBI)** | [0, ∞) | **lower** | Another *unsupervised* score: the average ratio of within-cluster scatter to between-cluster centroid distance. Low DBI = compact clusters whose centroids are far apart. Like Silhouette, computed from features alone. |

**Why three metrics, not just ARI?** ARI is the supervised gold standard but can
only be computed when ground truth is available. Silhouette and DBI evaluate
cluster *geometry* in feature space and reveal failure modes ARI hides. The CNN
results are a clean illustration: it wins on ARI on Collage A (0.671), yet has
the worst Silhouette (0.100) and DBI (2.597) of any feature set — its clusters
overlap heavily in feature space but happen to align with ground-truth labels.
Hand-crafted Haralick has lower ARI at default settings but cleaner geometry.

**Pixel accuracy** is reported alongside as the most intuitive read-out: the
fraction of pixels whose predicted segmentation label matches ground truth after
Hungarian alignment. It correlates with ARI but is more sensitive to boundary
errors (a 32×32 window straddling two textures contributes 1024 mixed pixels).

---

## Datasets

### Brodatz / USC-SIPI (committed)
Included under `textures/`. No download needed. Primary collage uses:

| Quadrant | File | Texture |
|---|---|---|
| Top-left | `1.1.01.tiff` | D9 Grass |
| Top-right | `1.1.02.tiff` | D12 Bark |
| Bottom-left | `1.1.09.tiff` | D29 Beach sand |
| Bottom-right | `1.5.07.tiff` | D94 Brick wall |

### texmos2 benchmark (committed)
8-class USC SIPI texture mosaic (`textures/texmos2.p512.tiff` + ground truth
`texmos2.s512.tiff`). Gray-coded labels decoded automatically by the notebook.

### EuroSAT (download required)
RGB satellite patches used for the satellite extension in Section 10.

1. Download the **RGB version** of EuroSAT from
   [https://github.com/phelber/EuroSAT](https://github.com/phelber/EuroSAT)
   or directly from the Zenodo archive linked there.
2. Extract so that the folder structure is:
   ```
   data/eurosat/EuroSAT/
       AnnualCrop/    Highway/       Residential/
       Forest/        Industrial/    River/
       HerbaceousVegetation/  Pasture/  SeaLake/
       PermanentCrop/
   ```
3. The notebook and UI pick this up automatically from `data/eurosat/EuroSAT/`.

---

## Setup

### Prerequisites

- **Python 3.10 or later** (tested on 3.10–3.12). Earlier versions miss type
  hints used in the notebook.
- ~2 GB free disk space (most of it is the optional EuroSAT dataset).
- A CUDA-capable GPU is optional. The CNN baseline (Section 10.3) uses
  pre-trained ResNet-18 in inference mode and runs on CPU in ~30 s — usable
  but a noticeable bottleneck if you re-run the notebook many times.

### Installation

```bash
git clone https://github.com/<anmolsureka30>/Texture-Based-Image-Segmentation.git
cd Texture-Based-Image-Segmentation

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

The first time `pip install` runs, expect it to take 3–5 minutes — `torch` and
`scikit-image` are the biggest downloads. PyTorch installs the CPU build by
default; if you have CUDA, replace the torch line in `requirements.txt` with
the CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/)
*before* running `pip install`.

### Running the notebook

```bash
jupyter notebook main.ipynb
# or, if you prefer JupyterLab:
jupyter lab main.ipynb
```

In the notebook, choose **Kernel → Restart & Run All** to execute all 64 cells
top-to-bottom. End-to-end runtime is **3–5 minutes on CPU** (the CNN cells in
Section 10.3 dominate). EuroSAT cells in Section 10.4 will be skipped gracefully
if the dataset is not present.

**What you'll see, by section:**
- Sections 1–6 — feature extraction (Otsu, GLCM, Haralick, SVD) with verification
  prints (`max abs diff < 1e-10`) and per-class fingerprint heatmaps.
- Section 7–8 — K-Means clustering and segmentation maps. Section 8 prints both
  the default (W=32) and the optimised (W=64 + multi-distance + PCA) results.
- Section 9 — the canonical metrics table and per-point silhouette diagram.
- Section 10 — extension experiments (multi-distance GLCM, window sweep, CNN,
  EuroSAT, grid search). Section 10.6 ends with a master comparison table
  ranking every experiment by ARI.

### Running the Streamlit UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Layout:

- **Left sidebar** — pick a dataset (Brodatz Collage A / texmos2 / EuroSAT 2×2 /
  upload your own image), then configure window size, gray levels, GLCM distance,
  feature set (Haralick / SVD / CNN), optional PCA, and K. Press **▶ Run pipeline**.
- **⭐ Load best settings** — applies the optimal configuration discovered by the
  Section 10.6 grid search, *per dataset* (Haralick + W=64 + multi-d + PCA for
  Brodatz, CNN ResNet-18 for texmos2 / EuroSAT). Use this to reproduce the
  headline numbers from the results table above.
- **Main panel** — original image, Otsu-quantised view, ground-truth mask (when
  available), and the predicted segmentation map. Below: the ARI / Silhouette /
  DBI / pixel-accuracy table for the current run.

Each pipeline run takes 5–60 s depending on settings (CNN at W=32 is the slowest;
Haralick at W=64 is the fastest).

### Troubleshooting

- `ModuleNotFoundError: No module named 'torch'` — the venv isn't active. Run
  `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix) again.
- Streamlit reports a port conflict — pass `--server.port 8502` (or any other
  free port).
- Notebook re-execution writes images that VS Code doesn't pick up — close the
  notebook tab and reopen it, or `Ctrl+Shift+P → Revert File`.
- EuroSAT cells skipped silently — check that
  `data/eurosat/EuroSAT/AnnualCrop/` (etc.) exist; the notebook only proceeds
  if at least the four classes used in the 2×2 collage are present.

---

## Reproducibility

- All random operations use `SEED = 0` (set in Section 1, propagated everywhere).
- K-Means runs 10 restarts and picks the lowest-inertia solution.
- Re-running the notebook end-to-end reproduces every number and figure in
  the report. Metric values match sklearn to within 10⁻¹⁰.

---

## Verification gaps (documented, not bugs)

| Primitive | Deviation | Reason |
|---|---|---|
| Multi-Otsu (N=4) vs skimage | ≤ 6 gray levels | Recursive binary (greedy) vs exhaustive joint search |
| GLCM 45°/135° vs skimage | ~6×10⁻³ | Symmetric-normalisation convention difference |
| Haralick Difference Variance vs mahotas | ~1.3×10³ relative | Deliberate: we use the statistically correct variance of the difference distribution; mahotas uses variance of probability values |
| K-Means inertia vs sklearn | < 0.05% | Both stochastic (k-means++) |

---

## Team

| Name | Roll number |
|---|---|
| Rishabh Kumar | 24B2419 |
| Anmol Sureka | 24B2470 |
| Aniruddha Deore | 24B2182 |

**Course:** GNR 602 Advanced Methods for Satellite Image Processing
