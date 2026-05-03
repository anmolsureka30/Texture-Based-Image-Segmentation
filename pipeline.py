import numpy as np
from pathlib import Path
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.models import resnet18, ResNet18_Weights

TEX_DIR = Path('textures')
DATA_DIR = Path('data')
TILE = 256
N_LEVELS = 16
WINDOW = 32
STEP = 16
DISTANCE = 1
K = 4
SVD_K = 8

BRODATZ_FILES = [
    ("1.1.01.tiff", "Grass (D9)"),
    ("1.1.02.tiff", "Bark (D12)"),
    ("1.1.07.tiff", "Beach sand (D29)"),
    ("1.1.12.tiff", "Brick wall (D94)"),
]

TEXMOS2_LEVEL_TO_NAME = {
    32:  'Grass (D9)',
    64:  'Water (D38)',
    96:  'Sand (D29)',
    128: 'Wool (D19)',
    160: 'Pigskin (D92)',
    192: 'Leather (D24)',
    224: 'Raffia (D84)',
    255: 'Wood (D68)',
}

HARALICK_NAMES = [
    'ASM (Energy)', 'Contrast', 'Correlation', 'Variance',
    'Homogeneity (IDM)', 'Sum Average', 'Sum Variance', 'Sum Entropy',
    'Entropy', 'Diff Variance', 'Diff Entropy',
    'Info Corr 1', 'Info Corr 2',
]

def load_grayscale(path):
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')
    return np.asarray(img, dtype=np.uint8)

def build_brodatz_collage(files, tile=TILE):
    tiles, tex_labels = [], []
    for fname, name in files:
        arr = load_grayscale(TEX_DIR / fname)
        tiles.append(arr[:tile, :tile])
        tex_labels.append(name)
    tl, tr, bl, br = tiles
    collage = np.concatenate(
        [np.concatenate([tl, tr], axis=1),
         np.concatenate([bl, br], axis=1)], axis=0
    ).astype(np.uint8)
    
    gt = np.zeros((2*tile, 2*tile), dtype=np.uint8)
    gt[:tile,  :tile]  = 0
    gt[:tile,  tile:]  = 1
    gt[tile:,  :tile]  = 2
    gt[tile:,  tile:]  = 3
    return collage, gt, tex_labels

def load_texmos2():
    texmos2 = load_grayscale(TEX_DIR / 'texmos2.p512.tiff')
    texmos2_gt_raw = load_grayscale(TEX_DIR / 'texmos2.s512.tiff')
    levels_sorted = sorted(TEXMOS2_LEVEL_TO_NAME.keys())
    level_to_classid = {lvl: i for i, lvl in enumerate(levels_sorted)}
    texmos2_gt = np.zeros_like(texmos2_gt_raw, dtype=np.uint8)
    for lvl, cid in level_to_classid.items():
        texmos2_gt[texmos2_gt_raw == lvl] = cid
    texmos2_labels = [TEXMOS2_LEVEL_TO_NAME[l] for l in levels_sorted]
    return texmos2, texmos2_gt, texmos2_labels

def build_eurosat_collage(eurosat_dir, classes, tile=TILE, seed=0):
    rng = np.random.default_rng(seed)
    eurosat_dir = Path(eurosat_dir)
    tiles = []
    for cls in classes:
        folder = eurosat_dir / cls
        jpgs = sorted(folder.glob('*.jpg'))
        if not jpgs:
            raise FileNotFoundError(f'No .jpg files in {folder}')
        pick = jpgs[rng.integers(0, len(jpgs))]
        img = Image.open(pick).convert('L').resize((tile, tile), Image.BILINEAR)
        tiles.append(np.asarray(img, dtype=np.uint8))
    tl, tr, bl, br = tiles
    coll = np.concatenate([np.concatenate([tl, tr], axis=1), np.concatenate([bl, br], axis=1)], axis=0)
    g = np.zeros((2 * tile, 2 * tile), dtype=np.uint8)
    g[:tile, :tile] = 0
    g[:tile, tile:] = 1
    g[tile:, :tile] = 2
    g[tile:, tile:] = 3
    return coll, g, list(classes)

def otsu_binary_threshold(hist):
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0
    p = hist / total
    levels = np.arange(len(hist))
    cum_p = np.cumsum(p)
    cum_mu = np.cumsum(p * levels)
    mu_T = cum_mu[-1]
    denom = cum_p * (1.0 - cum_p)
    numer = (mu_T * cum_p - cum_mu) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b2 = np.where(denom > 0, numer / denom, 0.0)
    return int(np.argmax(sigma_b2))

def otsu_multilevel(image, n_levels):
    img_flat = image.ravel()
    full_hist = np.bincount(img_flat, minlength=256)
    ranges = [(0, 255)]
    while len(ranges) < n_levels:
        best_i, best_var = (-1, -1.0)
        for i, (lo, hi) in enumerate(ranges):
            if hi - lo < 1:
                continue
            sub = img_flat[(img_flat >= lo) & (img_flat <= hi)]
            if sub.size < 2:
                continue
            v = float(sub.var())
            if v > best_var:
                best_var, best_i = (v, i)
        if best_i < 0:
            break
        lo, hi = ranges[best_i]
        sub_hist = full_hist[lo:hi + 1]
        t_rel = otsu_binary_threshold(sub_hist)
        t_abs = lo + t_rel
        if t_abs <= lo or t_abs >= hi:
            ranges.pop(best_i)
            ranges.append((lo, lo))
            continue
        ranges.pop(best_i)
        ranges.append((lo, t_abs))
        ranges.append((t_abs + 1, hi))
    ranges.sort()
    thresholds = np.array([hi for lo, hi in ranges[:-1]], dtype=np.int32)
    quantized = np.digitize(image, thresholds).astype(np.uint8)
    return (quantized, thresholds)

ANGLE_OFFSETS = {0: lambda d: (0, d), 45: lambda d: (-d, d), 90: lambda d: (-d, 0), 135: lambda d: (-d, -d)}

def build_glcm(patch, n_levels, distance=1, angle=0, symmetric=True, normalize=True):
    dr, dc = ANGLE_OFFSETS[angle](distance)
    H, W = patch.shape
    r0 = max(0, -dr)
    r1 = min(H, H - dr)
    c0 = max(0, -dc)
    c1 = min(W, W - dc)
    ref = patch[r0:r1, c0:c1]
    nbr = patch[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    glcm = np.zeros((n_levels, n_levels), dtype=np.float64)
    np.add.at(glcm, (ref.ravel().astype(np.intp), nbr.ravel().astype(np.intp)), 1.0)
    if symmetric:
        glcm = glcm + glcm.T
    if normalize:
        s = glcm.sum()
        if s > 0:
            glcm /= s
    return glcm

def build_glcm_sym(patch, n_levels, distance=1):
    acc = np.zeros((n_levels, n_levels), dtype=np.float64)
    for a in (0, 45, 90, 135):
        acc += build_glcm(patch, n_levels, distance=distance, angle=a, symmetric=True, normalize=True)
    return acc / 4.0

def sliding_glcms(image, n_levels, window=32, step=16, distance=1):
    H, W = image.shape
    centers = []
    glcms = []
    half = window // 2
    for r in range(0, H - window + 1, step):
        for c in range(0, W - window + 1, step):
            patch = image[r:r + window, c:c + window]
            glcms.append(build_glcm_sym(patch, n_levels, distance=distance))
            centers.append((r + half, c + half))
    return (np.stack(glcms, axis=0), centers)

def haralick_features(glcm, eps=1e-12):
    P = np.asarray(glcm, dtype=np.float64)
    L = P.shape[0]
    s = P.sum()
    if s > 0:
        P = P / s
    i_idx, j_idx = np.indices((L, L))
    levels = np.arange(L)
    px = P.sum(axis=1)
    py = P.sum(axis=0)
    mu_x = (levels * px).sum()
    mu_y = (levels * py).sum()
    sx = np.sqrt(((levels - mu_x) ** 2 * px).sum())
    sy = np.sqrt(((levels - mu_y) ** 2 * py).sum())
    pxpy = np.zeros(2 * L - 1)
    np.add.at(pxpy, (i_idx + j_idx).ravel(), P.ravel())
    pxmy = np.zeros(L)
    np.add.at(pxmy, np.abs(i_idx - j_idx).ravel(), P.ravel())
    k_sum = np.arange(2 * L - 1)
    k_diff = np.arange(L)
    HX = -(px * np.log2(px + eps)).sum()
    HY = -(py * np.log2(py + eps)).sum()
    HXY = -(P * np.log2(P + eps)).sum()
    pxpy_outer = np.outer(px, py)
    HXY1 = -(P * np.log2(pxpy_outer + eps)).sum()
    HXY2 = -(pxpy_outer * np.log2(pxpy_outer + eps)).sum()
    f1 = (P ** 2).sum()
    f2 = (k_diff ** 2 * pxmy).sum()
    f3 = ((i_idx * j_idx * P).sum() - mu_x * mu_y) / (sx * sy) if sx > 0 and sy > 0 else 0.0
    f4 = ((i_idx - mu_x) ** 2 * P).sum()
    f5 = (P / (1.0 + (i_idx - j_idx) ** 2)).sum()
    f6 = (k_sum * pxpy).sum()
    f7 = ((k_sum - f6) ** 2 * pxpy).sum()
    f8 = -(pxpy * np.log2(pxpy + eps)).sum()
    f9 = HXY
    mu_d = (k_diff * pxmy).sum()
    f10 = ((k_diff - mu_d) ** 2 * pxmy).sum()
    f11 = -(pxmy * np.log2(pxmy + eps)).sum()
    denom12 = max(HX, HY)
    f12 = (HXY - HXY1) / denom12 if denom12 > 0 else 0.0
    arg = 1.0 - np.exp(-2.0 * (HXY2 - HXY))
    f13 = np.sqrt(max(0.0, arg))
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13], dtype=np.float64)

def svd_features(glcm, k=SVD_K):
    sigma = np.linalg.svd(glcm, compute_uv=False)
    if len(sigma) >= k:
        return sigma[:k]
    out = np.zeros(k, dtype=np.float64)
    out[:len(sigma)] = sigma
    return out

def standardize(X):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd_safe

def _init_kmeanspp(X, k, rng):
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=np.float64)
    centroids[0] = X[rng.integers(0, n)]
    d2 = np.sum((X - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        s = d2.sum()
        probs = d2 / s if s > 0 else np.full(n, 1.0 / n)
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
        d2 = np.minimum(d2, np.sum((X - centroids[i]) ** 2, axis=1))
    return centroids

def _kmeans_once(X, k, max_iter, tol, seed):
    rng = np.random.default_rng(seed)
    C = _init_kmeanspp(X, k, rng)
    labels = np.zeros(X.shape[0], dtype=np.int32)
    n_used = 0
    for it in range(max_iter):
        d = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        new_labels = d.argmin(axis=1).astype(np.int32)
        new_C = np.empty_like(C)
        for c in range(k):
            mask = new_labels == c
            if mask.any():
                new_C[c] = X[mask].mean(axis=0)
            else:
                far = d.min(axis=1).argmax()
                new_C[c] = X[far]
        shift = np.linalg.norm(new_C - C)
        C, labels = (new_C, new_labels)
        n_used = it + 1
        if shift < tol:
            break
    inertia = float(np.sum((X - C[labels]) ** 2))
    return (labels, C, inertia, n_used)

def kmeans(X, k, n_restarts=10, max_iter=300, tol=0.0001, seed=0):
    best = None
    for r in range(n_restarts):
        out = _kmeans_once(X, k, max_iter, tol, seed=seed + r)
        if best is None or out[2] < best[2]:
            best = (*out, seed + r)
    return best

def hungarian_align(pred, true, k):
    cm = np.zeros((k, k), dtype=np.int64)
    for p, t in zip(pred, true):
        if 0 <= int(t) < k and 0 <= int(p) < k:
            cm[int(p), int(t)] += 1
    row, col = linear_sum_assignment(-cm)
    mapping = dict(zip(row, col))
    return np.array([mapping.get(int(p), int(p)) for p in pred])

def labels_to_map(labels, image_shape, window=WINDOW, step=STEP):
    H, W = image_shape
    n_rows = (H - window) // step + 1
    n_cols = (W - window) // step + 1
    half = window // 2
    rr, cc = np.indices((H, W))
    win_r = np.clip((rr - half + step // 2) // step, 0, n_rows - 1)
    win_c = np.clip((cc - half + step // 2) // step, 0, n_cols - 1)
    grid = np.asarray(labels).reshape(n_rows, n_cols)
    return grid[win_r, win_c].astype(np.int32)

def iou_per_class(pred, true, k):
    out = np.zeros(k, dtype=np.float64)
    for c in range(k):
        inter = ((pred == c) & (true == c)).sum()
        union = ((pred == c) | (true == c)).sum()
        out[c] = inter / union if union > 0 else 0.0
    return out

def adjusted_rand_index(true_labels, pred_labels):
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)
    n = len(true)
    classes_t = np.unique(true)
    classes_p = np.unique(pred)
    cont = np.zeros((len(classes_t), len(classes_p)), dtype=np.int64)
    for i, ct in enumerate(classes_t):
        rows = true == ct
        for j, cp in enumerate(classes_p):
            cont[i, j] = (rows & (pred == cp)).sum()
    sum_ij = (cont * (cont - 1) // 2).sum()
    sum_a = (cont.sum(axis=1) * (cont.sum(axis=1) - 1) // 2).sum()
    sum_b = (cont.sum(axis=0) * (cont.sum(axis=0) - 1) // 2).sum()
    n_pairs = n * (n - 1) // 2
    expected = sum_a * sum_b / n_pairs
    max_term = (sum_a + sum_b) / 2.0
    if max_term == expected:
        return 0.0
    return float((sum_ij - expected) / (max_term - expected))

def silhouette_score_scratch(X, labels):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n = X.shape[0]
    uniq = np.unique(labels)
    K_local = len(uniq)
    if K_local < 2:
        return 0.0, np.zeros(n)
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=2))
    mean_to_cluster = np.zeros((n, K_local))
    sizes = np.zeros(K_local, dtype=np.int64)
    for k_idx, cl in enumerate(uniq):
        mask = labels == cl
        sizes[k_idx] = mask.sum()
        mean_to_cluster[:, k_idx] = D[:, mask].mean(axis=1)
    label_to_idx = {cl: i for i, cl in enumerate(uniq)}
    own_idx = np.array([label_to_idx[l] for l in labels])
    own_size = sizes[own_idx]
    own_mean = mean_to_cluster[np.arange(n), own_idx]
    a = np.where(own_size > 1, own_mean * own_size / np.maximum(own_size - 1, 1), 0.0)
    masked = mean_to_cluster.copy()
    masked[np.arange(n), own_idx] = np.inf
    b = masked.min(axis=1)
    s = (b - a) / np.maximum(a, b)
    s = np.where(own_size > 1, s, 0.0)
    return float(s.mean()), s

def davies_bouldin_scratch(X, labels):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    K_local = len(uniq)
    if K_local < 2:
        return 0.0
    centroids = np.zeros((K_local, X.shape[1]))
    spreads = np.zeros(K_local)
    for k_idx, cl in enumerate(uniq):
        mask = labels == cl
        if mask.sum() == 0:
            continue
        centroids[k_idx] = X[mask].mean(axis=0)
        spreads[k_idx] = np.sqrt(((X[mask] - centroids[k_idx]) ** 2).sum(axis=1)).mean()
    D = np.sqrt(np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(D, np.inf)
    R = (spreads[:, None] + spreads[None, :]) / D
    np.fill_diagonal(R, 0.0)
    return float(R.max(axis=1).mean())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
_imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

def get_resnet():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model.to(device)

_resnet_model = None

def cnn_features(image_uint8, window=WINDOW, step=STEP, batch=64, resize_to=64):
    global _resnet_model
    if _resnet_model is None:
        _resnet_model = get_resnet()
        
    H, W = image_uint8.shape
    half = window // 2
    centers_local, patches = ([], [])
    for r in range(0, H - window + 1, step):
        for c in range(0, W - window + 1, step):
            p = image_uint8[r:r + window, c:c + window]
            centers_local.append((r + half, c + half))
            patches.append(np.stack([p, p, p], axis=0).astype(np.float32) / 255.0)
            
    x = torch.from_numpy(np.stack(patches)).to(device)
    x = torch.nn.functional.interpolate(x, size=(resize_to, resize_to), mode='bilinear', align_corners=False)
    x = (x - _imagenet_mean) / _imagenet_std
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            f = _resnet_model(x[i:i + batch])
            out.append(f.squeeze(-1).squeeze(-1).cpu().numpy())
    return (np.concatenate(out, axis=0), centers_local)