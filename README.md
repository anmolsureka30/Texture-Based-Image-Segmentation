# Texture-Based Image Segmentation

Statistical texture representation and unsupervised segmentation using GLCM,
Haralick features, SVD, and K-Means clustering. Course programming project.

## Pipeline

Brodatz collage -> multi-level Otsu quantization -> sliding-window GLCM ->
Haralick (13) / SVD (top-k singular values) features -> K-Means -> evaluation
(ARI, Silhouette, Davies-Bouldin) -> comparison across feature sets, scales,
and CNN baselines. Satellite extension uses EuroSAT patches.

## Repo layout

```
modules/            # reusable logic (imported by both notebook and UI)
  dataset.py        # Brodatz loader, collage builder, EuroSAT loader
  quantize.py       # multi-level Otsu (from scratch)
  glcm.py           # GLCM + sliding-window (from scratch)
  haralick.py       # 13 Haralick features (from scratch)
  svd_features.py   # top-k singular values of the GLCM
  clustering.py     # K-Means + standardisation (from scratch)
  evaluation.py     # ARI / Silhouette / DBI, Hungarian label matching
main.ipynb          # laboratory notebook (10 labelled sections)
app.py              # Streamlit UI (added in Phase 9)
textures/           # USC-SIPI Brodatz + texmos mosaics (committed)
data/               # generated collages + EuroSAT (gitignored)
requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

1. Notebook: `jupyter notebook main.ipynb` -> Run All.
2. UI (once Phase 9 is built): `streamlit run app.py`.

## Datasets

- **Brodatz / USC-SIPI** (committed under `textures/`). Primary collage uses
  D9 Grass, D12 Bark, D29 Sand, D94 Brick.
- **EuroSAT** (satellite extension, gitignored). Download RGB version into
  `data/eurosat/`; scripts convert to grayscale 2x2 collages with ground truth.

## Reproducibility

Global parameters (window size, gray levels, K, random seed) live in the
first cell of `main.ipynb`. Change once, re-run all. Seed defaults to 0.
