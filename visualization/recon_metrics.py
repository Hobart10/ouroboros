"""
Comprehensive reconstruction quality metrics for Ouroboros vocalizations.

Metrics
-------
    SPL          sound pressure level (dB, RMS-based)
    mean_f0      mean voiced fundamental frequency (Hz)
    dtw_f0       normalized DTW distance on F0 contours
    mod_corr     modulation spectrum correlation (Pearson r)
    lsd          log-spectral distortion (dB)
    r2_waveform  waveform R²
    snr_db       signal-to-noise ratio (dB)
    ssim_spec    spectrogram SSIM

All batch functions return dicts of per-call arrays so that statistical
tests can be run directly on them.
"""

import numpy as np
import librosa
import librosa.sequence
from scipy.ndimage import uniform_filter
from scipy.stats import wilcoxon, kruskal, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Optional, Tuple, List


def _structural_similarity(
    x: np.ndarray, y: np.ndarray, data_range: float, win_size: int = 11
) -> float:
    """SSIM using scipy.ndimage — no skimage dependency."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    xf, yf = x.astype(float), y.astype(float)
    ux  = uniform_filter(xf,      win_size)
    uy  = uniform_filter(yf,      win_size)
    uxx = uniform_filter(xf * xf, win_size)
    uyy = uniform_filter(yf * yf, win_size)
    uxy = uniform_filter(xf * yf, win_size)
    vx  = uxx - ux * ux
    vy  = uyy - uy * uy
    vxy = uxy - ux * uy
    num = (2 * ux * uy + C1) * (2 * vxy + C2)
    den = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    return float(np.mean(num / (den + 1e-10)))


# ---------------------------------------------------------------------------
# High-resolution spectrogram  (matches MATLAB pipeline defaults)
# ---------------------------------------------------------------------------

def make_highres_spec(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_ms: float = 1.0,
    win_ms: float = 8.0,
    fmin: float = 500.0,
    fmax: Optional[float] = None,
    ref_db: float = 20.0,
    min_db: float = -80.0,
    preemph: float = 0.97,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Log-power spectrogram with pre-emphasis.
    Defaults match the MATLAB pipeline (N_FFT=2048, HOP_MS=1, WIN_MS=8).

    Parameters
    ----------
    n_fft    : FFT size; increase to 4096 for higher frequency resolution
    hop_ms   : hop size in ms; decrease to 0.5 for finer time resolution
    win_ms   : analysis window in ms

    Returns  (S_db [F x T], freqs_hz [F], times_s [T])
    """
    if preemph > 0:
        audio = np.concatenate([[audio[0]], audio[1:] - preemph * audio[:-1]])

    hop_length = max(1, round(hop_ms * 1e-3 * sr))
    win_length = max(hop_length + 1, round(win_ms * 1e-3 * sr))
    n_fft_eff  = max(n_fft, win_length)

    S = librosa.stft(
        audio.astype(np.float32),
        n_fft=n_fft_eff, hop_length=hop_length,
        win_length=win_length, window="hann",
    )
    S_db = librosa.amplitude_to_db(np.abs(S), ref=ref_db)
    S_db = np.clip(S_db, min_db, 0.0)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft_eff)
    times = librosa.frames_to_time(
        np.arange(S.shape[1]), sr=sr, hop_length=hop_length
    )

    mask = np.ones(len(freqs), dtype=bool)
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax
    return S_db[mask], freqs[mask], times


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_spl(audio: np.ndarray, ref: float = 1.0) -> float:
    """RMS-based SPL in dB (relative to ref)."""
    rms = np.sqrt(np.mean(audio.astype(float) ** 2))
    return float(20.0 * np.log10(rms / ref + 1e-12))


def extract_f0(
    audio: np.ndarray,
    sr: int,
    fmin: float = 300.0,
    fmax: float = 12000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F0 contour via probabilistic YIN (pyin).
    Unvoiced frames are NaN.

    Returns (f0 [T], voiced_flag [T], times_s [T])
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio.astype(np.float32),
        fmin=fmin, fmax=fmax, sr=sr, fill_na=np.nan,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
    return f0, voiced_flag, times


def dtw_f0_distance(
    f0_orig: np.ndarray,
    f0_recon: np.ndarray,
) -> float:
    """
    Normalized DTW distance on voiced F0 frames (Hz).
    Returns NaN when fewer than 4 voiced frames exist in either signal.
    """
    orig_v  = f0_orig[~np.isnan(f0_orig)]
    recon_v = f0_recon[~np.isnan(f0_recon)]

    if len(orig_v) < 4 or len(recon_v) < 4:
        return np.nan

    # librosa dtw expects (D, N) feature matrices
    D, _ = librosa.sequence.dtw(orig_v[None, :], recon_v[None, :], metric="euclidean")
    return float(D[-1, -1] / (D.shape[0] + D.shape[1]))


def modulation_spectrum_corr(
    orig: np.ndarray,
    recon: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_ms: float = 1.0,
    win_ms: float = 8.0,
) -> float:
    """
    Mean Pearson r between modulation spectra (power spectrum of per-band
    amplitude envelope) of original vs reconstructed audio.
    """
    hop = max(1, round(hop_ms * 1e-3 * sr))
    win = max(hop + 1, round(win_ms * 1e-3 * sr))
    nf  = max(n_fft, win)

    min_len = min(len(orig), len(recon))
    S_o = np.abs(librosa.stft(orig[:min_len].astype(np.float32),  n_fft=nf, hop_length=hop, win_length=win))
    S_r = np.abs(librosa.stft(recon[:min_len].astype(np.float32), n_fft=nf, hop_length=hop, win_length=win))

    corrs = []
    for fi in range(S_o.shape[0]):
        env_o, env_r = S_o[fi], S_r[fi]
        if env_o.std() < 1e-10 or env_r.std() < 1e-10:
            continue
        mod_o = np.abs(np.fft.rfft(env_o)) ** 2
        mod_r = np.abs(np.fft.rfft(env_r)) ** 2
        r, _  = pearsonr(mod_o, mod_r)
        corrs.append(r)

    return float(np.nanmean(corrs)) if corrs else np.nan


def log_spectral_distortion(
    orig: np.ndarray,
    recon: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_ms: float = 1.0,
    win_ms: float = 8.0,
    eps: float = 1e-10,
) -> float:
    """
    Log-Spectral Distortion (LSD) in dB.
    LSD = sqrt( mean( (10·log10(P_orig) − 10·log10(P_recon))² ) )
    """
    hop = max(1, round(hop_ms * 1e-3 * sr))
    win = max(hop + 1, round(win_ms * 1e-3 * sr))
    nf  = max(n_fft, win)

    min_len = min(len(orig), len(recon))
    P_o = np.abs(librosa.stft(orig[:min_len].astype(np.float32),  n_fft=nf, hop_length=hop, win_length=win)) ** 2
    P_r = np.abs(librosa.stft(recon[:min_len].astype(np.float32), n_fft=nf, hop_length=hop, win_length=win)) ** 2

    log_diff = 10 * np.log10(P_o + eps) - 10 * np.log10(P_r + eps)
    return float(np.sqrt(np.mean(log_diff ** 2)))


def waveform_r2(orig: np.ndarray, recon: np.ndarray) -> float:
    min_len = min(len(orig), len(recon))
    o, r = orig[:min_len], recon[:min_len]
    ss_res = float(np.sum((o - r) ** 2))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def waveform_snr(orig: np.ndarray, recon: np.ndarray) -> float:
    min_len = min(len(orig), len(recon))
    o, r = orig[:min_len], recon[:min_len]
    sig_pow  = float(np.mean(o ** 2))
    noise_pow = float(np.mean((o - r) ** 2))
    return float(10.0 * np.log10(sig_pow / (noise_pow + 1e-12)))


def spectrogram_ssim(
    orig: np.ndarray,
    recon: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_ms: float = 1.0,
    win_ms: float = 8.0,
) -> float:
    """SSIM on normalized log-power spectrograms."""
    kw = dict(n_fft=n_fft, hop_ms=hop_ms, win_ms=win_ms, preemph=0.0)
    S_o, _, _ = make_highres_spec(orig,  sr, **kw)
    S_r, _, _ = make_highres_spec(recon, sr, **kw)
    rows = min(S_o.shape[0], S_r.shape[0])
    cols = min(S_o.shape[1], S_r.shape[1])
    s_o = S_o[:rows, :cols]
    s_r = S_r[:rows, :cols]
    dr = float(s_o.max() - s_o.min()) + 1e-8
    return _structural_similarity(s_o, s_r, data_range=dr)


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def batch_all_metrics(
    originals: List[np.ndarray],
    reconstructions: List[np.ndarray],
    sr: int,
    f0_fmin: float = 300.0,
    f0_fmax: float = 12000.0,
    n_fft: int = 2048,
    hop_ms: float = 1.0,
    win_ms: float = 8.0,
) -> dict:
    """
    Compute all metrics for a paired list of original and reconstructed calls.

    Returns dict of 1-D arrays (one value per call):
        spl_orig, spl_recon, spl_delta   — SPL in dB
        f0_orig,  f0_recon,  f0_delta    — mean voiced F0 in Hz
        dtw_f0                            — DTW distance on F0 contour
        mod_corr                          — modulation spectrum correlation
        lsd                               — log-spectral distortion (dB)
        r2_waveform                       — waveform R²
        snr_db                            — SNR (dB)
        ssim_spec                         — spectrogram SSIM
        duration_orig                     — call duration (s)
    """
    keys = [
        "spl_orig", "spl_recon", "spl_delta",
        "f0_orig",  "f0_recon",  "f0_delta",
        "dtw_f0", "mod_corr", "lsd",
        "r2_waveform", "snr_db", "ssim_spec",
        "duration_orig",
    ]
    results = {k: [] for k in keys}
    spec_kw = dict(n_fft=n_fft, hop_ms=hop_ms, win_ms=win_ms)

    for orig, recon in tqdm(zip(originals, reconstructions),
                             total=len(originals), desc="Computing metrics"):
        o = orig.ravel().astype(float)
        r = recon.ravel().astype(float)
        min_len = min(len(o), len(r))
        o, r = o[:min_len], r[:min_len]

        # SPL
        spl_o = compute_spl(o)
        spl_r = compute_spl(r)
        results["spl_orig"].append(spl_o)
        results["spl_recon"].append(spl_r)
        results["spl_delta"].append(spl_r - spl_o)

        # F0
        f0_o, _, _ = extract_f0(o, sr, fmin=f0_fmin, fmax=f0_fmax)
        f0_r, _, _ = extract_f0(r, sr, fmin=f0_fmin, fmax=f0_fmax)
        mean_f0_o = float(np.nanmean(f0_o))
        mean_f0_r = float(np.nanmean(f0_r))
        results["f0_orig"].append(mean_f0_o)
        results["f0_recon"].append(mean_f0_r)
        results["f0_delta"].append(mean_f0_r - mean_f0_o)

        # DTW on F0
        results["dtw_f0"].append(dtw_f0_distance(f0_o, f0_r))

        # Modulation spectrum
        results["mod_corr"].append(modulation_spectrum_corr(o, r, sr, **spec_kw))

        # LSD
        results["lsd"].append(log_spectral_distortion(o, r, sr, **spec_kw))

        # Waveform R² / SNR
        results["r2_waveform"].append(waveform_r2(o, r))
        results["snr_db"].append(waveform_snr(o, r))

        # SSIM
        results["ssim_spec"].append(spectrogram_ssim(o, r, sr, **spec_kw))

        # Duration
        results["duration_orig"].append(len(o) / sr)

    return {k: np.array(v, dtype=float) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Statistical reporting
# ---------------------------------------------------------------------------

_PAIRED_METRICS = [
    ("spl_orig",    "spl_recon",  "SPL (dB)"),
    ("f0_orig",     "f0_recon",   "Mean F0 (Hz)"),
]

_SCALAR_METRICS = [
    ("dtw_f0",      "DTW F0 distance"),
    ("mod_corr",    "Modulation spectrum r"),
    ("lsd",         "Log-spectral distortion (dB)"),
    ("r2_waveform", "Waveform R²"),
    ("snr_db",      "SNR (dB)"),
    ("ssim_spec",   "Spectrogram SSIM"),
]


def statistical_report(
    metrics: dict,
    duration_bins: Optional[np.ndarray] = None,
    out_path: Optional[str] = None,
) -> None:
    """
    Print (and optionally save) a statistical summary.

    Paired metrics (SPL, F0): Wilcoxon signed-rank on orig vs recon values.
    Scalar metrics: median ± IQR; optional stratification by duration bin
    with Kruskal-Wallis across bins.

    Parameters
    ----------
    metrics      : output of batch_all_metrics()
    duration_bins: optional bin edges for stratifying by call duration,
                   e.g. np.array([0, 0.3, 1.0, 3.0])
    out_path     : if given, also writes the report to this file
    """
    import io, sys

    buf = io.StringIO()
    _old = sys.stdout
    sys.stdout = buf

    print("=" * 60)
    print("  RECONSTRUCTION QUALITY REPORT")
    print("=" * 60)
    n = len(metrics["r2_waveform"])
    print(f"  N calls: {n}\n")

    # --- paired tests -------------------------------------------------
    print("── Paired comparison (orig vs recon) ──────────────────────")
    print(f"{'Metric':<26} {'Orig med':>9}  {'Recon med':>9}  {'Δ med':>8}  {'Wilcoxon p':>12}")
    print("-" * 70)
    for key_o, key_r, label in _PAIRED_METRICS:
        vo = metrics[key_o][np.isfinite(metrics[key_o])]
        vr = metrics[key_r][np.isfinite(metrics[key_r])]
        # align lengths (some calls may have NaN F0)
        n_pair = min(len(vo), len(vr))
        vo, vr = vo[:n_pair], vr[:n_pair]
        try:
            _, p = wilcoxon(vo, vr)
        except Exception:
            p = np.nan
        delta = np.median(vr - vo)
        print(f"{label:<26} {np.median(vo):>9.3f}  {np.median(vr):>9.3f}  "
              f"{delta:>+8.3f}  {p:>12.4g}")

    # --- scalar metrics -----------------------------------------------
    print("\n── Scalar reconstruction metrics ──────────────────────────")
    print(f"{'Metric':<30} {'Median':>8}  {'P25':>8}  {'P75':>8}")
    print("-" * 58)
    for key, label in _SCALAR_METRICS:
        v = metrics[key][np.isfinite(metrics[key])]
        p25, p50, p75 = np.percentile(v, [25, 50, 75])
        print(f"{label:<30} {p50:>8.4f}  {p25:>8.4f}  {p75:>8.4f}")

    # --- duration-stratified breakdown --------------------------------
    if duration_bins is not None:
        dur = metrics["duration_orig"]
        bin_idx = np.digitize(dur, duration_bins) - 1
        n_bins  = len(duration_bins) - 1
        print("\n── Stratified by duration ─────────────────────────────────")
        for key, label in _SCALAR_METRICS:
            v = metrics[key]
            groups = [v[(bin_idx == b) & np.isfinite(v)] for b in range(n_bins)]
            medians = [np.median(g) if len(g) > 0 else np.nan for g in groups]
            header  = f"\n  {label}"
            print(header)
            for b in range(n_bins):
                lo = duration_bins[b]
                hi = duration_bins[b + 1]
                print(f"    [{lo:.2f}–{hi:.2f}s]  n={len(groups[b])}  "
                      f"median={medians[b]:.4f}")
            valid_groups = [g for g in groups if len(g) >= 3]
            if len(valid_groups) >= 2:
                stat, p = kruskal(*valid_groups)
                print(f"    Kruskal-Wallis p = {p:.4g}")

    print("=" * 60)

    sys.stdout = _old
    report_text = buf.getvalue()
    print(report_text)
    if out_path:
        with open(out_path, "w") as f:
            f.write(report_text)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_metric_violins(
    metrics: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Violin plots for all scalar reconstruction metrics."""
    scalar_keys   = [k for k, _ in _SCALAR_METRICS]
    scalar_labels = [l for _, l in _SCALAR_METRICS]

    fig, axes = plt.subplots(1, len(scalar_keys), figsize=(3 * len(scalar_keys), 5))
    for ax, key, label in zip(axes, scalar_keys, scalar_labels):
        v = metrics[key][np.isfinite(metrics[key])]
        sns.violinplot(y=v, ax=ax, color="steelblue", inner="box", linewidth=1)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("")
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("Reconstruction metrics", y=1.01, fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_paired_scatter(
    metrics: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Paired scatter: original vs reconstructed values for SPL and F0."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (key_o, key_r, label) in zip(axes, _PAIRED_METRICS):
        vo = metrics[key_o]
        vr = metrics[key_r]
        mask = np.isfinite(vo) & np.isfinite(vr)
        ax.scatter(vo[mask], vr[mask], s=12, alpha=0.5, color="steelblue", linewidths=0)
        lim = [min(vo[mask].min(), vr[mask].min()),
               max(vo[mask].max(), vr[mask].max())]
        ax.plot(lim, lim, "k--", lw=1, label="identity")
        ax.set_xlabel(f"Original {label}")
        ax.set_ylabel(f"Reconstructed {label}")
        ax.set_title(label)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metric_vs_duration(
    metrics: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter of each scalar metric vs call duration."""
    scalar_keys   = [k for k, _ in _SCALAR_METRICS]
    scalar_labels = [l for _, l in _SCALAR_METRICS]
    dur = metrics["duration_orig"]

    ncols = 3
    nrows = int(np.ceil(len(scalar_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.ravel()

    for ax, key, label in zip(axes, scalar_keys, scalar_labels):
        v = metrics[key]
        mask = np.isfinite(v) & np.isfinite(dur)
        ax.scatter(dur[mask], v[mask], s=10, alpha=0.5, color="steelblue", linewidths=0)
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(scalar_keys):]:
        ax.set_visible(False)

    plt.suptitle("Reconstruction quality vs. call duration", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_orig_recon_pca(
    X_orig: np.ndarray,
    X_recon: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    PCA scatter of latent feature vectors from original and reconstructed audio.
    If latents are sufficient, each orig[i] should be close to its recon[i].

    Parameters
    ----------
    X_orig  : (N, D) feature matrix from originals
    X_recon : (N, D) feature matrix from reconstructions (same ordering)
    labels  : optional (N,) cluster labels for colouring originals
    """
    from sklearn.decomposition import PCA

    X_all = np.vstack([X_orig, X_recon])
    pca   = PCA(n_components=2)
    Z_all = pca.fit_transform(X_all)
    Z_o   = Z_all[:len(X_orig)]
    Z_r   = Z_all[len(X_orig):]

    fig, ax = plt.subplots(figsize=(7, 6))

    if labels is not None:
        cmap = plt.get_cmap("tab10", len(np.unique(labels)))
        for k, lab in enumerate(np.unique(labels)):
            mask = labels == lab
            ax.scatter(Z_o[mask, 0], Z_o[mask, 1], s=30, alpha=0.7,
                       color=cmap(k), marker="o", label=f"Orig {lab}", linewidths=0)
            ax.scatter(Z_r[mask, 0], Z_r[mask, 1], s=30, alpha=0.7,
                       color=cmap(k), marker="x", label=f"Recon {lab}", linewidths=0.8)
    else:
        ax.scatter(Z_o[:, 0], Z_o[:, 1], s=20, alpha=0.6, color="steelblue",
                   marker="o", label="Original", linewidths=0)
        ax.scatter(Z_r[:, 0], Z_r[:, 1], s=20, alpha=0.6, color="tomato",
                   marker="x", label="Reconstructed", linewidths=0.8)

    # draw lines connecting paired points (sub-sample to avoid clutter)
    n_lines = min(200, len(Z_o))
    idx_lines = np.random.choice(len(Z_o), n_lines, replace=False)
    for i in idx_lines:
        ax.plot([Z_o[i, 0], Z_r[i, 0]], [Z_o[i, 1], Z_r[i, 1]],
                color="grey", alpha=0.2, lw=0.5)

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    ax.set_title("PCA: original vs reconstructed latents")
    ax.legend(markerscale=1.3, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # quantify pairing quality
    pair_dist = np.linalg.norm(Z_o - Z_r, axis=1)
    cross_dist = np.mean([
        np.linalg.norm(Z_o[i] - Z_o[j])
        for i in np.random.choice(len(Z_o), 200, replace=True)
        for j in np.random.choice(len(Z_o), 1, replace=True)
    ])
    print(f"  PCA pairing: mean orig-recon dist = {pair_dist.mean():.4f}, "
          f"mean cross-call dist = {cross_dist:.4f}, "
          f"ratio = {pair_dist.mean() / (cross_dist + 1e-8):.3f} (lower → better)")

    return fig
