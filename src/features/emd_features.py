from typing import Iterable, List, Sequence, Tuple

import numpy as np
try:
    # Preferred: pure-Python library avoiding naming conflict with earth-mover 'pyemd'
    # pip package: EMD-signal, import name: emd
    import emd  # type: ignore
    _EMD_BACKEND = "emd-signal"
except Exception:  # pragma: no cover
    # Fallback to PyEMD if available
    from PyEMD import EMD  # type: ignore
    _EMD_BACKEND = "PyEMD"
from scipy.stats import skew, kurtosis  # type: ignore


def _safe_entropy(values: np.ndarray, num_bins: int = 64) -> float:
    if values.size == 0:
        return 0.0
    # Use histogram of absolute values to be robust to sign
    v = np.asarray(values, dtype=np.float32)
    if not np.any(np.isfinite(v)):
        return 0.0
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    v = np.abs(v)
    if np.allclose(v.max(), v.min()):
        return 0.0
    hist, _ = np.histogram(v, bins=num_bins, range=(v.min(), v.max()))
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _imf_stats(imf: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    m = float(np.mean(imf))
    sd = float(np.std(imf))
    energy = float(np.sum(imf.astype(np.float64) ** 2))
    ent = _safe_entropy(imf)
    sk = float(skew(imf, bias=False, nan_policy="omit"))
    ku = float(kurtosis(imf, fisher=True, bias=False, nan_policy="omit"))
    return m, sd, energy, ent, sk, ku


def compute_imfs_1d(signal_1d: np.ndarray, max_imfs: int = 4) -> List[np.ndarray]:
    x = np.asarray(signal_1d, dtype=np.float32).copy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # If constant signal, return zeros
    if np.allclose(x, x[0]):
        return [np.zeros_like(x) for _ in range(max_imfs)]
    if _EMD_BACKEND == "emd-signal":
        # emd.sift.sift returns array shape (n_imfs, n_samples)
        imfs_arr = emd.sift.sift(x)
        imfs_list = [imfs_arr[i] for i in range(min(imfs_arr.shape[0], max_imfs))]
    else:
        emd_inst = EMD()
        imfs_out = emd_inst.emd(x)
        imfs_list = [] if imfs_out is None else [imfs_out[i] for i in range(len(imfs_out))]
    if len(imfs_list) == 0:
        return [np.zeros_like(x) for _ in range(max_imfs)]
    imf_list = imfs_list[: max_imfs]
    # Pad with zeros to fixed length
    while len(imf_list) < max_imfs:
        imf_list.append(np.zeros_like(x))
    return imf_list


def extract_emd_features(
    luminance: np.ndarray,
    max_imfs: int = 4,
    projections: Sequence[str] = ("rows_mean", "cols_mean"),
) -> np.ndarray:
    """
    Compute EMD IMFs on 1D projections of the luminance image and return concatenated stats.

    - projections supported:
      - "rows_mean": mean across rows -> vector length W
      - "cols_mean": mean across cols -> vector length H
      - "rows_median": median across rows -> vector length W
      - "cols_median": median across cols -> vector length H
    - For each projection, compute up to max_imfs IMFs, then stats per IMF:
      [mean, std, energy, entropy, skewness, kurtosis]
    Return a 1D feature vector.
    """
    lum = np.asarray(luminance, dtype=np.float32)
    assert lum.ndim == 2, "luminance must be 2D"
    feats: List[float] = []

    for proj in projections:
        if proj == "rows_mean":
            series = lum.mean(axis=0)
        elif proj == "cols_mean":
            series = lum.mean(axis=1)
        elif proj == "rows_median":
            series = np.median(lum, axis=0)
        elif proj == "cols_median":
            series = np.median(lum, axis=1)
        else:
            raise ValueError(f"Unsupported projection: {proj}")

        imfs = compute_imfs_1d(series, max_imfs=max_imfs)
        for imf in imfs:
            feats.extend(_imf_stats(imf))

    feat_arr = np.asarray(feats, dtype=np.float32)
    # Replace any NaN/Inf produced by stats on degenerate IMFs
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_arr


