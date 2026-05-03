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
Input image
    │
    ▼
Multi-level Otsu quantisation  (L gray levels, recursive binary)
    │
    ▼
Sliding-window GLCM            (W×W windows, step S, rotation-invariant 4-angle average)
    │
    ├──► Haralick features  (13-dim per window, optionally multi-distance d∈{1,3,5})
    ├──► SVD features        (top-k singular values of GLCM, 8-dim per window)
    └──► CNN features        (ResNet-18 global-pooled embedding, 512-dim per window)
    │
    ▼
Z-score standardisation  [+ optional PCA decorrelation]
    │
    ▼
K-Means clustering         (k-means++ init, 10 restarts, empty-cluster reseeding)
    │
    ▼
Hungarian label alignment  (optimal permutation matching to ground truth)
    │
    ▼
Segmentation map           (closed-form nearest-window-centre pixel assignment)
    │
    ▼
Evaluation: ARI · Silhouette · Davies-Bouldin · Pixel accuracy · Per-class IoU
```

---

## Key findings

| Experiment | ARI | Pixel accuracy |
|---|---|---|
| Collage A — Haralick, W=32, d=1 (baseline) | 0.526 | 76.3% |
| Collage A — Haralick, W=64, d={1,3,5}, PCA (**best hand-crafted**) | **0.775** | **93.4%** |
| Collage A — CNN ResNet-18 | 0.671 | — |
| texmos2 — CNN ResNet-18 (**best**) | 0.251 | ~40% |
| EuroSAT 2×2 — Haralick | 0.215 | — |

- **Hand-crafted Haralick beats CNN** on the Brodatz benchmark (ARI 0.775 vs 0.671)
  when window size and multi-distance features are tuned.
- **Window size is the biggest lever**: doubling W from 32 to 64 gives a larger ARI
  gain (+0.137) than any feature-engineering experiment.
- **Internal and external metrics can disagree**: CNN has the best ARI but the worst
  Silhouette and DBI — tangled clusters that still align with ground truth.

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
├── modules/            # Dormant scaffolding — not imported by the notebook
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
Python 3.10 or later. A CUDA-capable GPU is optional (CNN baseline runs on CPU,
but is noticeably slower — ~30 s per run).

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

### Running the notebook

```bash
jupyter notebook main.ipynb
```

Run all cells top-to-bottom (Kernel → Restart & Run All). EuroSAT cells in
Section 10 will be skipped gracefully if the dataset is not present.

### Running the Streamlit UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Select a dataset, configure parameters in
the sidebar, and press **▶ Run pipeline**. The **⭐ Load best settings** button
applies the optimal configuration discovered by the Section 10 grid search.

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

**Course:** GNR 602 — Advanced Methods for Satellite Image Processing
**Institute:** Indian Institute of Technology Bombay
