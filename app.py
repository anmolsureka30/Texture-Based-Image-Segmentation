"""
Texture-Based Image Segmentation  —  Streamlit UI
Interactive front-end for the GLCM / Haralick / SVD / K-Means pipeline.

Best-discovered settings per dataset:
  Collage A:  Haralick, W=64, N=8, multi(1,3,5), PCA  → ARI 0.775 (beats CNN 0.671)
  texmos2:    CNN (ResNet18), W=64, N=16, PCA=off      → ARI 0.251 (limited by sub-window regions)
  EuroSAT:    CNN (ResNet18), W=64, N=8,  PCA=off      → ARI ~0.30  (grayscale loses colour)
"""
import streamlit as st

st.set_page_config(
    page_title="Texture Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import io, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score as _sk_silhouette

from pipeline import (
    DATA_DIR, BRODATZ_FILES,
    build_brodatz_collage, load_texmos2, build_eurosat_collage,
    otsu_multilevel, sliding_glcms, build_glcm_sym,
    haralick_features, svd_features, cnn_features,
    standardize, kmeans, hungarian_align, labels_to_map, iou_per_class,
    adjusted_rand_index, silhouette_score_scratch, davies_bouldin_scratch,
)

# ── CSS (theme-neutral — no background/text colour overrides) ────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 1rem; }
[data-testid="metric-container"] {
    border: 1px solid rgba(150,150,150,0.25);
    border-radius: 10px;
    padding: 0.65rem 1.1rem;
}
[data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 6px 20px; font-weight: 500; }
[data-testid="stExpander"] summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
_BRODATZ_NAMES   = [f[1] for f in BRODATZ_FILES]
_BRODATZ_FILEMAP = {f[1]: f[0] for f in BRODATZ_FILES}
_EUROSAT_DIR     = DATA_DIR / "eurosat" / "EuroSAT"
_EUROSAT_CLASSES = (
    sorted(d.name for d in _EUROSAT_DIR.iterdir() if d.is_dir())
    if _EUROSAT_DIR.exists() else []
)
_SIL_MAX_N = 2000   # above this, use sklearn C-impl instead of from-scratch O(n²)


# ── Small helpers ─────────────────────────────────────────────────────────────
def _cmap(n):       return plt.colormaps["tab10"].resampled(max(n, 2))
def _fig_to_png(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0); return buf.read()
def _seg_to_png(seg, k):
    cmap = _cmap(k)
    rgb  = (cmap(seg / max(k-1, 1))[:, :, :3] * 255).astype(np.uint8)
    buf  = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0); return buf.read()
def _imfig(arr, title, cmap="gray", vmin=None, vmax=None, figsize=(4, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=4); ax.axis("off"); fig.tight_layout(pad=0.3)
    return fig
def _confusion_matrix_arr(true, pred, k):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(true.astype(int), pred.astype(int)):
        if 0 <= t < k and 0 <= p < k: cm[t, p] += 1
    return cm
def _silhouette(X, labels):
    """Return (mean, per_point). Uses sklearn for large n to avoid OOM."""
    n = X.shape[0]
    if n > _SIL_MAX_N:
        mean_s = float(_sk_silhouette(X, labels))
        return mean_s, None          # per-point not available at large scale
    return silhouette_score_scratch(X, labels)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Texture Segmentation")
    st.caption("GLCM · Haralick · SVD · K-Means")
    st.divider()

    st.markdown("### Dataset")
    dataset_choice = st.radio(
        "dataset", ["Brodatz Collage A", "texmos2", "EuroSAT 2×2", "Upload image"],
        label_visibility="collapsed",
    )

    brodatz_selection = eurosat_selection = uploaded_file = None

    if dataset_choice == "Brodatz Collage A":
        st.markdown("**Texture quadrants** (TL · TR · BL · BR)")
        ca, cb = st.columns(2)
        tl = ca.selectbox("TL", _BRODATZ_NAMES, 0, key="tl")
        tr = cb.selectbox("TR", _BRODATZ_NAMES, 1, key="tr")
        bl = ca.selectbox("BL", _BRODATZ_NAMES, 2, key="bl")
        br = cb.selectbox("BR", _BRODATZ_NAMES, 3, key="br")
        brodatz_selection = [
            (_BRODATZ_FILEMAP[tl], tl), (_BRODATZ_FILEMAP[tr], tr),
            (_BRODATZ_FILEMAP[bl], bl), (_BRODATZ_FILEMAP[br], br),
        ]

    elif dataset_choice == "EuroSAT 2×2":
        st.markdown("**Land-cover classes** (TL · TR · BL · BR)")
        if not _EUROSAT_CLASSES:
            st.error(f"EuroSAT not found:\n`{_EUROSAT_DIR}`")
        else:
            n = len(_EUROSAT_CLASSES)
            ca, cb = st.columns(2)
            e_tl = ca.selectbox("TL", _EUROSAT_CLASSES, 0,          key="e_tl")
            e_tr = cb.selectbox("TR", _EUROSAT_CLASSES, min(1,n-1), key="e_tr")
            e_bl = ca.selectbox("BL", _EUROSAT_CLASSES, min(2,n-1), key="e_bl")
            e_br = cb.selectbox("BR", _EUROSAT_CLASSES, min(3,n-1), key="e_br")
            eurosat_selection = [e_tl, e_tr, e_bl, e_br]

    elif dataset_choice == "Upload image":
        uploaded_file = st.file_uploader("Grayscale image", type=["png","jpg","jpeg","tiff"])

    st.divider()

    # ── Per-dataset best-known settings (discovered via grid search §10.6) ────
    _BEST = {
        "Brodatz Collage A": dict(win=64, n_lev=8,  dist="multi (1,3,5)", feat="Haralick",      pca=True,  k=4,
                                  note="ARI 0.775 — best hand-crafted, beats CNN (0.671)"),
        "texmos2":           dict(win=64, n_lev=16, dist="multi (1,3,5)", feat="CNN (ResNet18)", pca=False, k=8,
                                  note="ARI 0.251 — CNN best for texmos2; limited by sub-window regions (<32px)"),
        "EuroSAT 2×2":       dict(win=64, n_lev=8,  dist="multi (1,3,5)", feat="CNN (ResNet18)", pca=False, k=4,
                                  note="ARI ~0.30 — CNN best; grayscale loses colour info for land-cover"),
        "Upload image":      dict(win=64, n_lev=8,  dist="multi (1,3,5)", feat="Haralick",      pca=True,  k=4,
                                  note="Set K = number of distinct textures in your image"),
    }
    _b = _BEST[dataset_choice]

    # ── Auto-apply best settings whenever the dataset changes ─────────────────
    # Key insight: Streamlit uses session_state[widget_key] to control what a
    # widget displays.  Writing directly to those keys (not shadow variables)
    # is the only reliable way to programmatically update slider/radio values.
    def _apply_best(cfg):
        st.session_state["_nlev_w"]  = cfg["n_lev"]
        st.session_state["_win_w"]   = cfg["win"]
        st.session_state["_k_w"]     = cfg["k"]
        st.session_state["_dist_w"]  = cfg["dist"]
        st.session_state["_feat_w"]  = cfg["feat"]
        st.session_state["_pca_w"]   = cfg["pca"]

    if st.session_state.get("_last_ds") != dataset_choice:
        _apply_best(_b)
        st.session_state["_last_ds"] = dataset_choice

    st.markdown("### Pipeline parameters")
    st.caption(f"Best known for **{dataset_choice}**: {_b['note']}")

    # ── Best-settings preset button (writes directly to widget keys) ──────────
    if st.button("⭐ Load best settings", help=_b["note"]):
        _apply_best(_b)
        st.rerun()

    # ── K guidance banner ────────────────────────────────────────────────────
    _k_hint = {
        "Brodatz Collage A": ("4 quadrants → **K = 4**",   "normal"),
        "texmos2":           ("8 texture classes → **K must be 8**", "inverse"),
        "EuroSAT 2×2":       ("4 quadrants → **K = 4**",   "normal"),
        "Upload image":      ("Set K = number of distinct textures", "off"),
    }
    _hint_txt, _ = _k_hint[dataset_choice]
    st.info(_hint_txt, icon="💡")

    # ── Dataset-specific limitation warnings ──────────────────────────────────
    _LIMITS = {
        "texmos2": (
            "⚠️ **texmos2 accuracy ceiling**\n\n"
            "texmos2 contains texture regions as small as **16×16 px** — smaller than "
            "any useful GLCM window. Every window centred on a small region also "
            "captures surrounding textures, making its GLCM a mix of two classes. "
            "No algorithm can reliably label these windows correctly from texture "
            "features alone.\n\n"
            "**Best achievable:** CNN ARI ≈ 0.25 · pixel accuracy ≈ 40–50%.  "
            "Collage A achieves ~93% because its four regions are each 256×256 px — "
            "8× larger than the window. These datasets are structurally incomparable."
        ),
        "EuroSAT 2×2": (
            "⚠️ **EuroSAT accuracy ceiling**\n\n"
            "EuroSAT images are RGB satellite photos converted to **grayscale**, losing "
            "the colour information that most reliably distinguishes land-cover types "
            "(e.g. vegetation vs. roads). High intra-class variability and seasonal "
            "lighting effects further confuse GLCM-based features.\n\n"
            "**Best achievable:** CNN ARI ≈ 0.30 · pixel accuracy ≈ 55–65% with grayscale.  "
            "CNN is the best available option here — hand-crafted features are insufficient."
        ),
    }
    if dataset_choice in _LIMITS:
        with st.expander("ℹ️ Why accuracy is limited for this dataset", expanded=False):
            st.markdown(_LIMITS[dataset_choice])

    # ── Widgets — default from session state (set by _apply_best above) ───────
    n_levels    = st.slider("Gray levels (N_LEVELS)", 4, 32,
                            step=4, key="_nlev_w")
    window_size = st.select_slider("Window size (WINDOW px)", [16, 32, 64],
                                   key="_win_w")
    auto_step   = st.toggle("Auto STEP = WINDOW / 2", value=True)
    step_size   = (window_size // 2 if auto_step
                   else st.slider("Step size (STEP px)", 4, 64, window_size // 2))
    k_clusters  = st.slider("Clusters (K)", 2, 10, key="_k_w")
    _dist_opts  = ["1", "3", "5", "multi (1,3,5)"]
    distance_opt = st.selectbox("GLCM distance", _dist_opts,
                                index=_dist_opts.index(st.session_state["_dist_w"]),
                                key="_dist_w")

    st.divider()

    st.markdown("### Feature set")
    _feat_opts = ["Haralick", "SVD", "CNN (ResNet18)"]
    feature_choice = st.radio("feat", _feat_opts,
                               index=_feat_opts.index(st.session_state["_feat_w"]),
                               label_visibility="collapsed", key="_feat_w")
    if "CNN" in feature_choice:
        st.info("ResNet18 cached after first load (~45 MB).", icon="ℹ️")

    st.divider()

    st.markdown("### Pre-processing")
    use_pca = st.toggle(
        "PCA decorrelation before K-Means",
        key="_pca_w",
        help="Reduces correlated Haralick/SVD features to independent principal components "
             "(95% variance retained). Gives +0.10 ARI on Collage A vs no PCA.",
    )
    if use_pca and "CNN" in feature_choice:
        st.caption("PCA is applied to CNN features too — may or may not help.")

    st.divider()
    run_btn = st.button("▶  Run pipeline", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading ResNet18 weights…")
def _load_resnet():
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m = torch.nn.Sequential(*list(m.children())[:-1])
    m.eval()
    return m.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


@st.cache_data(show_spinner=False)
def _load_dataset(choice, b_sel, e_sel, up_bytes):
    if choice == "Brodatz Collage A": return build_brodatz_collage(b_sel)
    if choice == "texmos2":           return load_texmos2()
    if choice == "EuroSAT 2×2":
        if e_sel is None: raise ValueError("No EuroSAT classes selected.")
        return build_eurosat_collage(_EUROSAT_DIR, e_sel)
    if up_bytes is None: return None, None, []
    pil = Image.open(io.BytesIO(up_bytes)).convert("L")
    return np.asarray(pil, dtype=np.uint8), None, []


@st.cache_data(show_spinner=False)
def _run_pipeline(img_raw, img_q, feat, dist_opt, w, s, levels, k, apply_pca):
    """
    Feature extraction + optional PCA + K-Means.

    Returns
    -------
    X         : raw feature matrix (n_windows, raw_dims)
    X_s       : z-scored X
    X_cluster : PCA-transformed X_s if apply_pca else X_s  (used for clustering)
    labels    : K-Means assignments (n_windows,)
    centers   : list of (row, col) window centres
    inertia   : K-Means inertia of best restart
    pca_dims  : dimensionality after PCA (== raw_dims if no PCA)
    """
    distances = [1, 3, 5] if "multi" in dist_opt else [int(dist_opt.split()[0])]

    if "CNN" in feat:
        X, centers = cnn_features(img_raw, window=w, step=s)
    else:
        blocks, centers = [], None
        for d in distances:
            glcms, c = sliding_glcms(img_q, levels, window=w, step=s, distance=d)
            if centers is None: centers = c
            fn = haralick_features if "Haralick" in feat else svd_features
            blocks.append(np.stack([fn(g) for g in glcms]))
        X = np.concatenate(blocks, axis=1)

    X_s = standardize(X)

    if apply_pca:
        pca_xform = PCA(n_components=0.95, random_state=0)
        X_cluster = pca_xform.fit_transform(X_s)
    else:
        X_cluster = X_s

    pca_dims = X_cluster.shape[1]
    labels, _, inertia, _, _ = kmeans(X_cluster, k, n_restarts=10, seed=0)
    return X, X_s, X_cluster, labels, centers, inertia, pca_dims


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Texture-Based Image Segmentation")
st.caption(
    "GLCM-based texture features (Haralick / SVD / CNN) + K-Means.  "
    "**Collage A best:** Haralick W=64, N=8, multi-distance + PCA → ARI **0.775** (beats CNN 0.671).  "
    "**texmos2/EuroSAT best:** CNN (ResNet18) — GLCM windows exceed small region sizes.  "
    "Configure sidebar → ▶ Run."
)

if not run_btn and "res" not in st.session_state:
    st.info("👈  Configure the sidebar and press **▶ Run pipeline**.", icon="ℹ️")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTE
# ═══════════════════════════════════════════════════════════════════════════════
if run_btn:
    if dataset_choice == "Upload image" and uploaded_file is None:
        st.error("Please upload an image first."); st.stop()
    if dataset_choice == "EuroSAT 2×2" and not _EUROSAT_CLASSES:
        st.error("EuroSAT dataset not found."); st.stop()

    t0 = time.time()
    if "CNN" in feature_choice: _load_resnet()

    with st.spinner("Loading dataset…"):
        up_bytes = uploaded_file.getvalue() if uploaded_file else None
        img_raw, gt_raw, class_labels = _load_dataset(
            dataset_choice, brodatz_selection, eurosat_selection, up_bytes)

    if img_raw is None:
        st.error("Could not load image."); st.stop()

    with st.spinner("Quantising…"):
        img_q, _ = otsu_multilevel(img_raw, n_levels)

    with st.spinner(f"Extracting {feature_choice} features"
                    + (" + PCA…" if use_pca else "…")):
        X, X_s, X_cl, raw_labels, centers, inertia, pca_dims = _run_pipeline(
            img_raw, img_q, feature_choice, distance_opt,
            window_size, step_size, n_levels, k_clusters, use_pca,
        )

    has_gt = gt_raw is not None
    if has_gt:
        k_gt    = int(gt_raw.max()) + 1
        gt_win  = np.array([gt_raw[r, c] for r, c in centers])
        aligned = hungarian_align(raw_labels, gt_win, k_clusters)
    else:
        k_gt, gt_win, aligned = k_clusters, None, raw_labels

    seg_map = labels_to_map(aligned, img_raw.shape, window=window_size, step=step_size)

    if has_gt:
        # Compare seg_map directly against the pixel-level ground truth.
        # Previous code indexed gt_raw with window-grid indices (0..n_rows-1)
        # instead of pixel coordinates, so gt_px was always 0 (top-left class)
        # — fixed by using gt_raw directly.
        disagree = seg_map != gt_raw
        px_acc   = (~disagree).mean()
        iou      = iou_per_class(aligned, gt_win, k_clusters)
        conf_mat = _confusion_matrix_arr(gt_win, aligned, k_clusters)
    else:
        disagree = iou = conf_mat = None
        px_acc = float("nan")

    ari      = adjusted_rand_index(gt_win, aligned) if has_gt else float("nan")
    sil, spp = _silhouette(X_cl, aligned)
    dbi      = davies_bouldin_scratch(X_cl, aligned)
    elapsed  = time.time() - t0

    st.session_state["res"] = dict(
        img_raw=img_raw, img_q=img_q, gt_raw=gt_raw,
        seg_map=seg_map, aligned=aligned, disagree=disagree,
        X_s=X_s, X_cl=X_cl, spp=spp, centers=centers,
        class_labels=class_labels, gt_win=gt_win, k_gt=k_gt,
        iou=iou, conf_mat=conf_mat, px_acc=px_acc,
        ari=ari, sil=sil, dbi=dbi, inertia=inertia, elapsed=elapsed,
        feature_choice=feature_choice, k_clusters=k_clusters,
        n_levels=n_levels, window_size=window_size, step_size=step_size,
        distance_opt=distance_opt, dataset_choice=dataset_choice,
        use_pca=use_pca, pca_dims=pca_dims,
    )
    st.session_state.pop("cmp", None)


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════
r = st.session_state.get("res")
if r is None: st.stop()

img_raw = r["img_raw"]; img_q = r["img_q"]; gt_raw = r["gt_raw"]
seg_map = r["seg_map"]; aligned = r["aligned"]; disagree = r["disagree"]
X_s = r["X_s"]; X_cl = r["X_cl"]; spp = r["spp"]; centers = r["centers"]
class_labels = r["class_labels"]; gt_win = r["gt_win"]; k_gt = r["k_gt"]
iou = r["iou"]; conf_mat = r["conf_mat"]; px_acc = r["px_acc"]
ari = r["ari"]; sil = r["sil"]; dbi = r["dbi"]
inertia = r["inertia"]; elapsed = r["elapsed"]
feat = r["feature_choice"]; K = r["k_clusters"]; N_LEV = r["n_levels"]
WIN = r["window_size"]; STP = r["step_size"]; dist_opt = r["distance_opt"]
use_pca = r["use_pca"]; pca_dims = r["pca_dims"]
has_gt = gt_raw is not None
CK = _cmap(K)

pca_note = f" + PCA({pca_dims}D)" if use_pca else ""
st.success(
    f"Pipeline done in **{elapsed:.1f} s** — "
    f"feature dims: **{X_s.shape[1]}**{pca_note} → K-Means on **{X_cl.shape[1]}D**, "
    f"windows: **{X_s.shape[0]}**", icon="✅"
)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.subheader("Key metrics")
def _quality(v, high_good=True):
    if np.isnan(v): return None, "off"
    if high_good:
        lbl = "Good" if v >= 0.6 else ("Fair" if v >= 0.4 else "Poor")
        col = "normal" if v >= 0.6 else ("off" if v >= 0.4 else "inverse")
    else:
        lbl = "Good" if v <= 0.8 else ("Fair" if v <= 1.2 else "Poor")
        col = "normal" if v <= 0.8 else ("off" if v <= 1.2 else "inverse")
    return lbl, col

m1, m2, m3, m4 = st.columns(4)
_d, _c = _quality(ari); m1.metric("Adjusted Rand Index",  f"{ari:.3f}" if not np.isnan(ari) else "N/A", delta=_d, delta_color=_c or "off")
_d, _c = _quality(sil); m2.metric("Silhouette Score",     f"{sil:.3f}", delta=_d, delta_color=_c or "off")
_d, _c = _quality(dbi, high_good=False); m3.metric("Davies-Bouldin Index", f"{dbi:.3f}", delta=_d, delta_color=_c or "off")
m4.metric("Pixel Accuracy", f"{px_acc:.1%}" if not np.isnan(px_acc) else "N/A")

# Cluster size bar
pred_counts = pd.Series(aligned.astype(int)).value_counts().sort_index().rename("Predicted")
pred_counts.index = [class_labels[i] if class_labels and i < len(class_labels) else f"C{i}" for i in pred_counts.index]
df_sz = pred_counts.to_frame()
if has_gt:
    gc = pd.Series(gt_win.astype(int)).value_counts().sort_index().rename("Ground truth")
    gc.index = [class_labels[i] if class_labels and i < len(class_labels) else f"C{i}" for i in gc.index]
    df_sz = pd.concat([df_sz, gc], axis=1).fillna(0).astype(int)
fig_sz, ax_sz = plt.subplots(figsize=(10, 2.2))
df_sz.plot(kind="bar", ax=ax_sz, color=["#2980b9","#e67e22"][:len(df_sz.columns)], edgecolor="white", rot=0)
ax_sz.set_ylabel("Windows"); ax_sz.grid(axis="y", alpha=0.3); ax_sz.legend(fontsize=9)
fig_sz.tight_layout(pad=0.3); st.pyplot(fig_sz, use_container_width=True); plt.close(fig_sz)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_res, tab_diag, tab_cmp, tab_exp = st.tabs([
    "🗺️  Results", "🔬  Diagnostics", "⚖️  Haralick vs SVD", "💾  Export",
])

# ══════════════ TAB 1 — Results ═══════════════════════════════════════════════
with tab_res:
    st.markdown("#### Input images")
    c1, c2, c3 = st.columns(3)
    c1.pyplot(_imfig(img_raw, "Original (grayscale)"), use_container_width=True); plt.close("all")

    fig_q, ax_q = plt.subplots(figsize=(4,4))
    ax_q.imshow(img_q, cmap=plt.colormaps["tab20"].resampled(N_LEV), vmin=-0.5, vmax=N_LEV-0.5, interpolation="nearest")
    ax_q.set_title(f"Otsu-quantised ({N_LEV} levels)", fontsize=10, pad=4); ax_q.axis("off"); fig_q.tight_layout(pad=0.3)
    c2.pyplot(fig_q, use_container_width=True); plt.close(fig_q)

    if has_gt:
        fig_g, ax_g = plt.subplots(figsize=(4,4))
        ax_g.imshow(gt_raw, cmap=_cmap(k_gt), vmin=-0.5, vmax=k_gt-0.5, interpolation="nearest")
        ax_g.set_title("Ground truth", fontsize=10, pad=4); ax_g.axis("off")
        if class_labels:
            ax_g.legend(handles=[mpatches.Patch(color=_cmap(k_gt)(i),
                        label=class_labels[i] if i<len(class_labels) else str(i))
                        for i in range(k_gt)], loc="lower right", fontsize=7, framealpha=0.8)
        fig_g.tight_layout(pad=0.3); c3.pyplot(fig_g, use_container_width=True); plt.close(fig_g)
    else:
        c3.info("No ground truth for uploaded images.")

    st.markdown("#### Segmentation results")
    c4, c5, c6 = st.columns(3)

    fig_s, ax_s = plt.subplots(figsize=(4,4))
    ax_s.imshow(seg_map, cmap=CK, vmin=-0.5, vmax=K-0.5, interpolation="nearest")
    ax_s.set_title("Predicted segmentation", fontsize=10, pad=4); ax_s.axis("off")
    if class_labels:
        ax_s.legend(handles=[mpatches.Patch(color=CK(i),
                    label=class_labels[i] if i<len(class_labels) else str(i))
                    for i in range(K)], loc="lower right", fontsize=7, framealpha=0.8)
    fig_s.tight_layout(pad=0.3); c4.pyplot(fig_s, use_container_width=True); plt.close(fig_s)

    if has_gt:
        fig_d, ax_d = plt.subplots(figsize=(4,4))
        ax_d.imshow(img_raw, cmap="gray")
        ov = np.zeros((*img_raw.shape,4), dtype=np.float32)
        ov[disagree,0]=1.0; ov[disagree,3]=0.55
        ax_d.imshow(ov, interpolation="nearest")
        ax_d.set_title(f"Errors (red) — {disagree.mean()*100:.1f}% wrong", fontsize=10, pad=4); ax_d.axis("off")
        fig_d.tight_layout(pad=0.3); c5.pyplot(fig_d, use_container_width=True); plt.close(fig_d)

        xlabels = [class_labels[i] if class_labels and i<len(class_labels) else str(i) for i in range(K)]
        fig_iou, ax_iou = plt.subplots(figsize=(4,4))
        bars = ax_iou.bar(xlabels, iou, color=[CK(i) for i in range(K)], edgecolor="white")
        ax_iou.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
        ax_iou.axhline(iou.mean(), color="gray", ls="--", lw=1, label=f"Mean={iou.mean():.2f}")
        ax_iou.set_ylim(0,1.1); ax_iou.set_ylabel("IoU"); ax_iou.set_title("Per-class IoU", fontsize=10, pad=4)
        ax_iou.legend(fontsize=8); ax_iou.grid(axis="y", alpha=0.3); ax_iou.tick_params(axis="x", labelrotation=15)
        fig_iou.tight_layout(pad=0.3); c6.pyplot(fig_iou, use_container_width=True); plt.close(fig_iou)
    else:
        c5.info("No GT — disagreement unavailable."); c6.info("No GT — IoU unavailable.")


# ══════════════ TAB 2 — Diagnostics ═══════════════════════════════════════════
with tab_diag:

    st.markdown("#### 2-D PCA projection")
    pca_vis = PCA(n_components=2, random_state=0)
    Xp      = pca_vis.fit_transform(X_s)          # always visualise in raw-feature PCA
    var     = pca_vis.explained_variance_ratio_
    ncols   = 2 if has_gt else 1
    fig_pca, axes_pca = plt.subplots(1, ncols, figsize=(5*ncols, 4.5))
    if ncols == 1: axes_pca = [axes_pca]
    for ax, (lbl_arr, title) in zip(axes_pca, [(aligned,"Predicted clusters"),(gt_win,"Ground truth")][:ncols]):
        for cl in np.unique(lbl_arr):
            m = lbl_arr == cl
            lname = class_labels[int(cl)] if class_labels and int(cl)<len(class_labels) else f"Cluster {cl}"
            ax.scatter(Xp[m,0], Xp[m,1], s=14, alpha=0.65, color=_cmap(K)(int(cl)%10), edgecolors="none", label=lname)
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=9); ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=9)
        ax.set_title(title, fontsize=11); ax.legend(fontsize=7, markerscale=2, framealpha=0.8); ax.grid(alpha=0.2)
    fig_pca.tight_layout(); st.pyplot(fig_pca, use_container_width=True); plt.close(fig_pca)

    if has_gt:
        st.markdown("#### Confusion matrix  (rows = GT, cols = Predicted)")
        xlabels = [class_labels[i] if class_labels and i<len(class_labels) else str(i) for i in range(K)]
        cm_norm = conf_mat.astype(float); rs = cm_norm.sum(axis=1, keepdims=True)
        cm_norm = np.where(rs>0, cm_norm/rs, 0)
        fig_cm, ax_cm = plt.subplots(figsize=(max(5,K), max(4,K-1)))
        im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax_cm.set_xticks(range(K)); ax_cm.set_xticklabels(xlabels, rotation=20, ha="right")
        ax_cm.set_yticks(range(K)); ax_cm.set_yticklabels(xlabels)
        ax_cm.set_xlabel("Predicted (aligned)"); ax_cm.set_ylabel("Ground truth")
        for i in range(K):
            for j in range(K):
                ax_cm.text(j, i, f"{conf_mat[i,j]}", ha="center", va="center", fontsize=9,
                           color="white" if cm_norm[i,j]>0.55 else "black")
        plt.colorbar(im, ax=ax_cm, label="Recall (row-normalised)"); fig_cm.tight_layout()
        st.pyplot(fig_cm, use_container_width=True); plt.close(fig_cm)

    st.markdown("#### Sample GLCMs (one window per cluster)")
    sample_glcms, sample_titles = [], []
    for cl in np.unique(aligned)[:min(len(np.unique(aligned)),4)]:
        idxs = np.where(aligned==cl)[0]; mid = idxs[len(idxs)//2]
        rc, cc_c = centers[mid]; half = WIN//2
        patch = img_q[max(0,rc-half):rc+half, max(0,cc_c-half):cc_c+half]
        if patch.shape==(WIN,WIN):
            sample_glcms.append(build_glcm_sym(patch, N_LEV, distance=1))
            sample_titles.append(class_labels[int(cl)] if class_labels and int(cl)<len(class_labels) else f"Cluster {cl}")
    if sample_glcms:
        ng = len(sample_glcms)
        fig_gl, axes_gl = plt.subplots(1, ng, figsize=(4*ng, 3.5))
        if ng==1: axes_gl=[axes_gl]
        for ax, g, t in zip(axes_gl, sample_glcms, sample_titles):
            ax.imshow(np.log1p(g), cmap="viridis", interpolation="nearest")
            ax.set_title(t, fontsize=10); ax.set_xlabel("Neighbour j"); ax.set_ylabel("Reference i")
        fig_gl.suptitle("Rotation-invariant GLCMs (log-scaled), d=1", fontsize=11)
        fig_gl.tight_layout(); st.pyplot(fig_gl, use_container_width=True); plt.close(fig_gl)

    if spp is not None and len(spp)>0:
        st.markdown("#### Per-window silhouette diagram")
        fig_sil, ax_sil = plt.subplots(figsize=(9, 3.5))
        y_lo = 5
        for cl in range(K):
            mask = aligned==cl
            if not mask.any(): continue
            vals = np.sort(spp[mask]); y_hi = y_lo+len(vals)
            ax_sil.fill_betweenx(np.arange(y_lo, y_hi), 0, vals, color=CK(cl), alpha=0.85)
            ax_sil.text(-0.08, (y_lo+y_hi)/2, class_labels[cl] if class_labels and cl<len(class_labels) else str(cl), fontsize=8, va="center")
            y_lo = y_hi+5
        ax_sil.axvline(sil, color="red", ls="--", lw=1.5, label=f"Mean={sil:.3f}")
        ax_sil.axvline(0, color="gray", lw=0.8); ax_sil.set_xlabel("Silhouette s(i)")
        ax_sil.set_ylabel("Windows"); ax_sil.set_xlim(-0.45,1.05); ax_sil.legend(fontsize=9)
        ax_sil.grid(alpha=0.2, axis="x"); fig_sil.tight_layout()
        st.pyplot(fig_sil, use_container_width=True); plt.close(fig_sil)
    elif spp is None:
        st.caption(f"Silhouette diagram skipped for large n={X_cl.shape[0]} (>2000 windows). Mean shown in metrics above.")

    with st.expander("Run configuration"):
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Dataset | `{r["dataset_choice"]}` |
| Feature set | `{feat}` |
| GLCM distance | `{dist_opt}` |
| PCA pre-processing | `{use_pca}` ({pca_dims}D after PCA) |
| Raw feature dims | `{X_s.shape[1]}` |
| Clustering dims | `{X_cl.shape[1]}` |
| Windows | `{X_s.shape[0]}` |
| WINDOW / STEP | `{WIN}` / `{STP}` |
| N_LEVELS | `{N_LEV}` |
| K | `{K}` |
| K-Means inertia | `{inertia:.2f}` |
""")


# ══════════════ TAB 3 — Haralick vs SVD ═══════════════════════════════════════
with tab_cmp:
    st.markdown(
        "Runs both Haralick and SVD on the current image with identical parameters "
        f"(WINDOW={WIN}, N={N_LEV}, distance={dist_opt}, PCA={use_pca}) and "
        "compares metrics side-by-side."
    )
    run_cmp = st.button("⚖️  Run comparison", type="secondary")

    if run_cmp or "cmp" in st.session_state:
        if run_cmp:
            with st.spinner("Running Haralick…"):
                _, Xh_s, Xh_cl, lh, _, ih, ph = _run_pipeline(
                    img_raw, img_q, "Haralick", dist_opt, WIN, STP, N_LEV, K, use_pca)
            with st.spinner("Running SVD…"):
                _, Xs_s, Xs_cl, ls, _, is_, ps = _run_pipeline(
                    img_raw, img_q, "SVD", dist_opt, WIN, STP, N_LEV, K, use_pca)

            lh_al = hungarian_align(lh, gt_win, K) if has_gt else lh
            ls_al = hungarian_align(ls, gt_win, K) if has_gt else ls
            ari_h = adjusted_rand_index(gt_win, lh_al) if has_gt else float("nan")
            ari_s = adjusted_rand_index(gt_win, ls_al) if has_gt else float("nan")
            sil_h, _ = _silhouette(Xh_cl, lh_al)
            sil_s, _ = _silhouette(Xs_cl, ls_al)
            dbi_h = davies_bouldin_scratch(Xh_cl, lh_al)
            dbi_s = davies_bouldin_scratch(Xs_cl, ls_al)
            seg_h = labels_to_map(lh_al, img_raw.shape, WIN, STP)
            seg_s = labels_to_map(ls_al, img_raw.shape, WIN, STP)

            st.session_state["cmp"] = dict(
                ari_h=ari_h, ari_s=ari_s, sil_h=sil_h, sil_s=sil_s,
                dbi_h=dbi_h, dbi_s=dbi_s, seg_h=seg_h, seg_s=seg_s,
                dim_h=ph, dim_s=ps,
                sizes_h=np.bincount(lh_al, minlength=K).tolist(),
                sizes_s=np.bincount(ls_al, minlength=K).tolist(),
            )

        cmp = st.session_state["cmp"]
        cmp_df = pd.DataFrame({
            "Metric":    ["ARI ↑", "Silhouette ↑", "DBI ↓", "Cluster dims"],
            "Haralick":  [f"{cmp['ari_h']:.4f}", f"{cmp['sil_h']:.4f}", f"{cmp['dbi_h']:.4f}", str(cmp['dim_h'])],
            "SVD":       [f"{cmp['ari_s']:.4f}", f"{cmp['sil_s']:.4f}", f"{cmp['dbi_s']:.4f}", str(cmp['dim_s'])],
        }).set_index("Metric")
        st.markdown("#### Metrics at a glance")
        st.dataframe(cmp_df, use_container_width=True)

        winner = "Haralick" if (not np.isnan(cmp["ari_h"]) and cmp["ari_h"] >= cmp["ari_s"]) else "SVD"
        st.success(f"🏆  **{winner}** wins on ARI.", icon="🏆")

        cc1, cc2 = st.columns(2)
        for col, seg, title in [(cc1, cmp["seg_h"], f"Haralick (ARI={cmp['ari_h']:.3f})"),
                                 (cc2, cmp["seg_s"], f"SVD (ARI={cmp['ari_s']:.3f})")]:
            fig_c, ax_c = plt.subplots(figsize=(4,4))
            ax_c.imshow(seg, cmap=CK, vmin=-0.5, vmax=K-0.5, interpolation="nearest")
            ax_c.set_title(title, fontsize=10); ax_c.axis("off"); fig_c.tight_layout(pad=0.3)
            col.pyplot(fig_c, use_container_width=True); plt.close(fig_c)

        xlabels_cmp = [class_labels[i] if class_labels and i<len(class_labels) else str(i) for i in range(K)]
        fig_cs, ax_cs = plt.subplots(figsize=(9, 2.5))
        x = np.arange(K); bw = 0.35
        ax_cs.bar(x-bw/2, cmp["sizes_h"], bw, label="Haralick", color="#2980b9", edgecolor="white")
        ax_cs.bar(x+bw/2, cmp["sizes_s"], bw, label="SVD",      color="#e67e22", edgecolor="white")
        ax_cs.axhline(X_s.shape[0]//K, color="gray", ls="--", lw=1, label=f"Balanced (~{X_s.shape[0]//K})")
        ax_cs.set_xticks(x); ax_cs.set_xticklabels(xlabels_cmp, rotation=15)
        ax_cs.set_ylabel("Windows"); ax_cs.legend(); ax_cs.grid(axis="y", alpha=0.3)
        fig_cs.tight_layout(); st.pyplot(fig_cs, use_container_width=True); plt.close(fig_cs)
    else:
        st.info("Press **⚖️ Run comparison** to compare Haralick vs SVD with current settings.")


# ══════════════ TAB 4 — Export ════════════════════════════════════════════════
with tab_exp:
    st.markdown("#### Download results")
    e1, e2, e3 = st.columns(3)
    e1.download_button("📥  Segmentation map (PNG)", data=_seg_to_png(seg_map, K),
                       file_name="segmentation_map.png", mime="image/png", use_container_width=True)

    mrow = {"dataset": r["dataset_choice"], "feature_set": feat,
            "K": K, "WINDOW": WIN, "STEP": STP, "N_LEVELS": N_LEV,
            "GLCM_distance": dist_opt, "PCA": use_pca, "PCA_dims": pca_dims,
            "ARI": round(ari,6) if not np.isnan(ari) else "N/A",
            "Silhouette": round(sil,6), "DBI": round(dbi,6),
            "Pixel_acc": round(px_acc,6) if not np.isnan(px_acc) else "N/A",
            "Inertia": round(inertia,4)}
    e2.download_button("📥  Metrics (CSV)", data=pd.DataFrame([mrow]).to_csv(index=False),
                       file_name="metrics.csv", mime="text/csv", use_container_width=True)

    if has_gt and disagree is not None:
        fig_de, ax_de = plt.subplots(figsize=(5,5))
        ax_de.imshow(img_raw, cmap="gray")
        ov=np.zeros((*img_raw.shape,4),dtype=np.float32); ov[disagree,0]=1.0; ov[disagree,3]=0.6
        ax_de.imshow(ov, interpolation="nearest"); ax_de.axis("off"); fig_de.tight_layout(pad=0)
        e3.download_button("📥  Disagreement map (PNG)", data=_fig_to_png(fig_de),
                           file_name="disagreement_map.png", mime="image/png", use_container_width=True)
        plt.close(fig_de)

    st.divider()
    st.markdown("#### Full metrics record")
    st.dataframe(pd.DataFrame([mrow]), use_container_width=True)
