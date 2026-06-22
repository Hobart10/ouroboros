"""
Dimensionality reduction and UMAP visualization for Ouroboros latent features.

Pipeline:
    load variable-length vocalizations
    → model.get_funcs() per call  (omega, gamma time series)
    → aggregate to fixed-length embedding vector
    → UMAP
    → scatter plot coloured by duration / cluster label
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from typing import Optional

from utils import deriv_approx_dy, get_spec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 1.  Latent extraction
# ---------------------------------------------------------------------------

def extract_ouroboros_embeddings(
    model: torch.nn.Module,
    audio_list: list,
    sr: int,
    smoothing: bool = True,
    include_weights: bool = False,
    max_len_t: float = 4.0,
) -> dict:
    """
    Run each vocalization through a trained Ouroboros and return
    per-call latent time-series.

    Parameters
    ----------
    model       : trained Ouroboros (eval mode expected)
    audio_list  : list of np.ndarray, each shape (1, T_i, 1) — variable T_i
    sr          : sample rate (Hz)
    smoothing   : whether to smooth omega/gamma before returning
    include_weights : whether to also return kernel weights (large array)
    max_len_t   : seconds; longer calls are processed chunk-by-chunk

    Returns
    -------
    dict with keys
        'omega'    : list of (T_i,) arrays — instantaneous frequency
        'gamma'    : list of (T_i,) arrays — instantaneous damping
        'weights'  : list of (T_i, P) arrays, only if include_weights=True
        'durations': np.ndarray of call durations in seconds
        'sr'       : sample rate
    """
    dt = 1.0 / sr
    model.eval()

    omegas, gammas, weights_list, durations = [], [], [], []
    skipped = 0

    for i, aud in enumerate(tqdm(audio_list, desc="Extracting latents")):
        # aud shape: (1, T, 1)
        T = aud.shape[1]
        if T < 10:
            skipped += 1
            continue

        dy = deriv_approx_dy(aud)

        x_t  = torch.from_numpy(aud).to(torch.float32).to(DEVICE)
        dy_t = torch.from_numpy(dy).to(torch.float32).to(DEVICE)

        try:
            with torch.no_grad():
                omega, gamma, _, weights, _ = model.get_funcs(
                    x_t, dy_t, dt,
                    smoothing=smoothing,
                    max_len_t=max_len_t,
                )
        except Exception as e:
            print(f"  call {i} failed ({e}), skipping")
            skipped += 1
            continue

        omegas.append(omega.detach().cpu().numpy().squeeze())
        gammas.append(gamma.detach().cpu().numpy().squeeze())
        durations.append(T / sr)

        if include_weights:
            weights_list.append(weights.detach().cpu().numpy().squeeze())

    if skipped:
        print(f"  skipped {skipped}/{len(audio_list)} calls")

    result = {
        "omega":     omegas,
        "gamma":     gammas,
        "durations": np.array(durations),
        "sr":        sr,
    }
    if include_weights:
        result["weights"] = weights_list

    return result


# ---------------------------------------------------------------------------
# 2.  Aggregate time-series latents → fixed-length embedding
# ---------------------------------------------------------------------------

def aggregate_latents(
    latents: dict,
    method: str = "mean_std",
    percentiles: tuple = (25, 50, 75),
) -> np.ndarray:
    """
    Collapse per-call latent time-series into a fixed-length feature vector.

    Parameters
    ----------
    latents    : output of extract_ouroboros_embeddings()
    method     : one of
                   'mean_std'   — [mean(ω), std(ω), mean(γ), std(γ)]               4-dim
                   'percentile' — percentiles of ω and γ                            6-dim (3 ptiles × 2)
                   'full'       — mean_std + percentiles + duration                 11-dim
    percentiles: which percentiles to use when method includes them

    Returns
    -------
    X : np.ndarray of shape (N_calls, D)
    """
    omegas    = latents["omega"]
    gammas    = latents["gamma"]
    durations = latents["durations"]
    N = len(omegas)

    rows = []
    for i in range(N):
        om = np.asarray(omegas[i]).ravel()
        gm = np.asarray(gammas[i]).ravel()

        feats = []
        if method in ("mean_std", "full"):
            feats += [om.mean(), om.std(), gm.mean(), gm.std()]

        if method in ("percentile", "full"):
            feats += list(np.percentile(om, percentiles))
            feats += list(np.percentile(gm, percentiles))

        if method == "full":
            feats += [durations[i]]

        rows.append(feats)

    X = np.array(rows, dtype=np.float32)

    # z-score across calls, per feature — UMAP is sensitive to scale
    mu  = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True) + 1e-8
    X   = (X - mu) / sig

    return X


# ---------------------------------------------------------------------------
# 3.  UMAP
# ---------------------------------------------------------------------------

def compute_umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    """
    Run UMAP on feature matrix X.

    Returns
    -------
    embedding : np.ndarray of shape (N, n_components)
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn:  pip install umap-learn")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)
    return embedding


# ---------------------------------------------------------------------------
# 4.  Visualization
# ---------------------------------------------------------------------------

def plot_umap(
    embedding: np.ndarray,
    durations: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[list] = None,
    title: str = "Ouroboros latents — UMAP",
    save_path: Optional[str] = None,
    alpha: float = 0.7,
    s: float = 18,
) -> plt.Figure:
    """
    Scatter plot of a 2-D UMAP embedding.

    Coloring priority:  labels > durations > default grey.

    Parameters
    ----------
    embedding   : (N, 2) array from compute_umap()
    durations   : (N,) call durations in seconds — used for continuous coloring
    labels      : (N,) integer cluster labels — used for categorical coloring
    label_names : list of str, legend entries for each unique label
    title       : figure title
    save_path   : if given, saves figure to this path
    alpha/s     : scatter transparency and marker size
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    if labels is not None:
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("tab10", len(unique_labels))
        for k, lab in enumerate(unique_labels):
            mask = labels == lab
            name = label_names[k] if (label_names and k < len(label_names)) else str(lab)
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                s=s, alpha=alpha, color=cmap(k), label=name, linewidths=0,
            )
        ax.legend(framealpha=0.8, markerscale=1.5)

    elif durations is not None:
        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=durations, cmap="viridis", s=s, alpha=alpha, linewidths=0,
        )
        plt.colorbar(sc, ax=ax, label="Duration (s)")

    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=s, alpha=alpha,
                   color="steelblue", linewidths=0)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# 5.  Reconstruction metrics (per call)
# ---------------------------------------------------------------------------

def reconstruction_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sr: int,
    win_len: int = 128,
) -> dict:
    """
    Compute waveform and spectrogram similarity between original and
    reconstructed audio for a single call.

    Parameters
    ----------
    original      : 1-D waveform array
    reconstructed : 1-D waveform array (same length)
    sr            : sample rate
    win_len       : STFT window length for spectrogram comparison

    Returns
    -------
    dict with keys: r2_waveform, snr_db, ssim_spec, mse_spec
    """
    from scipy.stats import pearsonr
    from skimage.metrics import structural_similarity as ssim

    # align lengths
    min_len = min(len(original), len(reconstructed))
    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # waveform R²
    ss_res = np.sum((orig - recon) ** 2)
    ss_tot = np.sum((orig - orig.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))

    # SNR
    signal_power = np.mean(orig ** 2)
    noise_power  = np.mean((orig - recon) ** 2)
    snr_db = float(10 * np.log10(signal_power / (noise_power + 1e-12)))

    # spectrogram comparison
    duration = min_len / sr
    s_orig, *_ = get_spec(
        orig, sr, onset=0.0, offset=duration,
        shoulder=0.0, win_len=win_len, interp=False, normalize=True,
    )
    s_recon, *_ = get_spec(
        recon, sr, onset=0.0, offset=duration,
        shoulder=0.0, win_len=win_len, interp=False, normalize=True,
    )

    # match shapes for SSIM
    min_rows = min(s_orig.shape[0], s_recon.shape[0])
    min_cols = min(s_orig.shape[1], s_recon.shape[1])
    s_orig  = s_orig[:min_rows, :min_cols]
    s_recon = s_recon[:min_rows, :min_cols]

    data_range = float(s_orig.max() - s_orig.min()) + 1e-8
    ssim_val = float(ssim(s_orig, s_recon, data_range=data_range))
    mse_spec = float(np.mean((s_orig - s_recon) ** 2))

    return {
        "r2_waveform": r2,
        "snr_db":      snr_db,
        "ssim_spec":   ssim_val,
        "mse_spec":    mse_spec,
    }


def batch_reconstruction_metrics(
    originals: list,
    reconstructions: list,
    sr: int,
) -> dict:
    """
    Run reconstruction_metrics() over a list of call pairs and
    return arrays of per-call metric values.
    """
    keys = ["r2_waveform", "snr_db", "ssim_spec", "mse_spec"]
    results = {k: [] for k in keys}

    for orig, recon in tqdm(zip(originals, reconstructions), total=len(originals),
                             desc="Computing metrics"):
        m = reconstruction_metrics(orig.ravel(), recon.ravel(), sr)
        for k in keys:
            results[k].append(m[k])

    return {k: np.array(v) for k, v in results.items()}


def report_metrics(metrics: dict, group_labels: Optional[np.ndarray] = None) -> None:
    """
    Print median ± IQR for each metric, optionally broken down by group.
    Also runs a Wilcoxon signed-rank test between groups (if 2 groups provided).
    """
    from scipy.stats import wilcoxon, kruskal

    keys = list(metrics.keys())

    if group_labels is None:
        print(f"{'Metric':<18} {'Median':>8}  {'IQR (25–75)':>18}")
        print("-" * 48)
        for k in keys:
            v = metrics[k]
            p25, p50, p75 = np.percentile(v, [25, 50, 75])
            print(f"{k:<18} {p50:>8.4f}  [{p25:.4f}, {p75:.4f}]")
    else:
        unique_groups = np.unique(group_labels)
        header = f"{'Metric':<18}" + "".join(f"  Group {g} (med)" for g in unique_groups)
        print(header)
        print("-" * (18 + 18 * len(unique_groups)))
        for k in keys:
            v = metrics[k]
            row = f"{k:<18}"
            group_vals = []
            for g in unique_groups:
                mask = group_labels == g
                p50 = np.median(v[mask])
                row += f"  {p50:>10.4f}    "
                group_vals.append(v[mask])
            print(row)

            # statistical test
            if len(unique_groups) == 2:
                # paired Wilcoxon if same N, else unpaired Kruskal
                try:
                    stat, p = wilcoxon(group_vals[0], group_vals[1])
                    print(f"  {'':18} Wilcoxon p = {p:.4g}")
                except ValueError:
                    stat, p = kruskal(*group_vals)
                    print(f"  {'':18} Kruskal-Wallis p = {p:.4g}")
