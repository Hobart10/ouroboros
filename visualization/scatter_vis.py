"""
Interactive UMAP scatter visualization with inline spectrogram display.

Two outputs:
  1. Python matplotlib figure — click a point to see its spectrogram in a
     side panel.  Saves static PNG for publication.
  2. .mat export — feeds directly into the existing MATLAB function
     plotSpecEmbedings_Estim_dot_warbler_audio_dyna for the full interactive
     experience (hover, GIF export, border-grid spectrograms).

Visualization recommendation
-----------------------------
Use the Python scatter for quick analysis and publication figures.
Use the MATLAB export for interactive exploration (hover spectrograms,
animated GIF, the N×N border-box grid) — the MATLAB function already
provides all of that and is faster to navigate interactively.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional, List
from tqdm import tqdm

from visualization.recon_metrics import make_highres_spec


# ---------------------------------------------------------------------------
# Core scatter with click-to-show spectrogram
# ---------------------------------------------------------------------------

def plot_scatter_interactive(
    embedding: np.ndarray,
    audio_list: List[np.ndarray],
    sr: int,
    durations: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    title: str = "UMAP — click a point to see its spectrogram",
    n_fft: int = 2048,
    hop_ms: float = 1.0,
    win_ms: float = 8.0,
    fmin: float = 500.0,
    fmax: Optional[float] = None,
    save_path: Optional[str] = None,
    alpha: float = 0.7,
    s: float = 16,
) -> plt.Figure:
    """
    Interactive matplotlib scatter.  Click a scatter point to display its
    high-resolution spectrogram in the right panel.

    Parameters
    ----------
    embedding  : (N, 2) UMAP coordinates
    audio_list : list of N arrays, each 1-D waveform (variable length)
    sr         : sample rate
    durations  : (N,) durations in seconds (used for colouring if no labels)
    labels     : (N,) integer cluster labels
    n_fft / hop_ms / win_ms / fmin / fmax : spectrogram parameters
        increase n_fft (e.g. 4096) for finer frequency resolution
        decrease hop_ms (e.g. 0.5) for finer time resolution
    save_path  : saves a static PNG of the current figure
    """
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.35)
    ax_sc  = fig.add_subplot(gs[0])
    ax_sp  = fig.add_subplot(gs[1])

    # --- draw scatter ---------------------------------------------------
    if labels is not None:
        unique_labs = np.unique(labels)
        cmap_cat = plt.get_cmap("tab10", len(unique_labs))
        point_colors = np.zeros((len(embedding), 3))
        for k, lab in enumerate(unique_labs):
            mask = labels == lab
            name = label_names[k] if (label_names and k < len(label_names)) else str(lab)
            col  = cmap_cat(k)[:3]
            ax_sc.scatter(embedding[mask, 0], embedding[mask, 1],
                          s=s, alpha=alpha, color=col, label=name, linewidths=0)
            point_colors[mask] = col
        ax_sc.legend(framealpha=0.8, markerscale=1.5, fontsize=8)
        cbar = None

    elif durations is not None:
        norm = Normalize(vmin=durations.min(), vmax=durations.max())
        sc   = ax_sc.scatter(embedding[:, 0], embedding[:, 1],
                              c=durations, cmap="viridis", s=s, alpha=alpha,
                              linewidths=0, norm=norm)
        plt.colorbar(sc, ax=ax_sc, label="Duration (s)", shrink=0.8)
        cmap_cont = plt.get_cmap("viridis")
        point_colors = cmap_cont(norm(durations))[:, :3]
        cbar = sc

    else:
        ax_sc.scatter(embedding[:, 0], embedding[:, 1], s=s, alpha=alpha,
                      color="steelblue", linewidths=0)
        point_colors = np.full((len(embedding), 3), [0.27, 0.51, 0.71])

    ax_sc.set_xlabel("UMAP 1");  ax_sc.set_ylabel("UMAP 2")
    ax_sc.set_title(title, fontsize=10)
    ax_sc.spines[["top", "right"]].set_visible(False)

    # placeholder for spectrogram panel
    ax_sp.set_facecolor("black")
    ax_sp.set_xticks([]);  ax_sp.set_yticks([])
    ax_sp.set_title("Click a point", fontsize=10)

    # highlight ring for last clicked point
    highlight, = ax_sc.plot([], [], "o", ms=10, mfc="none", mew=2, color="orange", zorder=5)

    # --- click handler --------------------------------------------------
    spec_kw = dict(n_fft=n_fft, hop_ms=hop_ms, win_ms=win_ms,
                   fmin=fmin, fmax=fmax, preemph=0.97)

    def on_click(event):
        if event.inaxes is not ax_sc:
            return
        dists = (embedding[:, 0] - event.xdata) ** 2 + \
                (embedding[:, 1] - event.ydata) ** 2
        idx = int(np.argmin(dists))
        aud = audio_list[idx].ravel()
        try:
            S_db, freqs, times = make_highres_spec(aud, sr, **spec_kw)
        except Exception as e:
            ax_sp.set_title(f"Call {idx}: spec error ({e})", fontsize=9)
            fig.canvas.draw_idle()
            return

        ax_sp.cla()
        ax_sp.imshow(
            S_db, aspect="auto", origin="lower", cmap="inferno",
            extent=[times[0], times[-1], freqs[0] / 1000, freqs[-1] / 1000],
        )
        ax_sp.set_xlabel("Time (s)");  ax_sp.set_ylabel("Frequency (kHz)")
        dur_s = len(aud) / sr
        ax_sp.set_title(
            f"Call {idx}  |  {dur_s:.3f} s  |  "
            f"n_fft={spec_kw['n_fft']}  hop={spec_kw['hop_ms']} ms",
            fontsize=9,
        )
        dot_color = point_colors[idx]
        for spine in ax_sp.spines.values():
            spine.set_edgecolor(dot_color);  spine.set_linewidth(2)

        highlight.set_data([embedding[idx, 0]], [embedding[idx, 1]])
        highlight.set_color(dot_color)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved static figure → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Border-grid layout  (mirrors MATLAB's N×N spectrogram border)
# ---------------------------------------------------------------------------

def plot_border_grid(
    embedding: np.ndarray,
    audio_list: List[np.ndarray],
    sr: int,
    n_grid: int = 6,
    durations: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    n_fft: int = 2048,
    hop_ms: float = 1.0,
    win_ms: float = 8.0,
    fmin: float = 500.0,
    fmax: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    N×N grid layout matching the MATLAB border-box design:
    spectrogram thumbnails ring the outside; UMAP scatter in the centre.

    Parameters
    ----------
    n_grid : grid dimension (n_grid=6 → 6×6 outer ring = 20 thumbnails)
    """
    n_border = 4 * n_grid - 4

    # compute border cell (row, col) positions clockwise from top-left
    box_rc = []
    for k in range(n_border):
        if k < n_grid:
            r, c = 0, k
        elif k < 2 * n_grid - 1:
            r, c = k - n_grid + 1, n_grid - 1
        elif k < 3 * n_grid - 2:
            r, c = n_grid - 1, (n_grid - 3) - (k - 2 * n_grid)
        else:
            r, c = n_border - k, 0
        box_rc.append((r, c))

    # pick representative points nearest each border position
    pts    = embedding
    border_xy = np.array([[c / (n_grid - 1), 1 - r / (n_grid - 1)]
                           for r, c in box_rc])
    # normalize embedding to [0,1]
    e_norm = (pts - pts.min(0)) / (pts.ptp(0) + 1e-8)

    available = np.ones(len(pts), dtype=bool)
    chosen = []
    for bxy in border_xy:
        d = np.sum((e_norm - bxy) ** 2, axis=1)
        d[~available] = np.inf
        idx = int(np.argmin(d))
        chosen.append(idx)
        available[idx] = False

    # --- figure layout ---------------------------------------------------
    fig = plt.figure(figsize=(10, 10), facecolor="white")

    PAD    = 0.008
    cell_w = (1 - (n_grid + 1) * PAD) / n_grid
    cell_h = cell_w

    spec_kw = dict(n_fft=n_fft, hop_ms=hop_ms, win_ms=win_ms,
                   fmin=fmin, fmax=fmax, preemph=0.97)

    # border spectrogram axes
    for bi, (r, c) in enumerate(box_rc):
        x0 = PAD + c * (cell_w + PAD)
        y0 = 1 - PAD - (r + 1) * (cell_h + PAD)
        ax = fig.add_axes([x0, y0, cell_w, cell_h])
        ax.set_xticks([]);  ax.set_yticks([]);  ax.set_aspect("auto")

        idx = chosen[bi]
        aud = audio_list[idx].ravel()
        try:
            S_db, _, _ = make_highres_spec(aud, sr, **spec_kw)
            ax.imshow(S_db, aspect="auto", origin="lower", cmap="inferno")
        except Exception:
            ax.set_facecolor("black")

        if labels is not None:
            cmap_cat = plt.get_cmap("tab10", len(np.unique(labels)))
            col = cmap_cat(int(labels[idx]))[:3]
        elif durations is not None:
            norm = Normalize(durations.min(), durations.max())
            col  = plt.get_cmap("viridis")(norm(durations[idx]))[:3]
        else:
            col = (0, 0, 0)

        for spine in ax.spines.values():
            spine.set_edgecolor(col);  spine.set_linewidth(1.5)

    # central scatter axes
    cx0 = PAD + 1 * (cell_w + PAD)
    cy0 = 1 - PAD - (n_grid - 1) * (cell_h + PAD)
    cw  = (n_grid - 2) * (cell_w + PAD) - PAD
    ax_sc = fig.add_axes([cx0, cy0, cw, cw])

    if labels is not None:
        cmap_cat = plt.get_cmap("tab10", len(np.unique(labels)))
        for k, lab in enumerate(np.unique(labels)):
            mask = labels == lab
            name = label_names[k] if (label_names and k < len(label_names)) else str(lab)
            ax_sc.scatter(embedding[mask, 0], embedding[mask, 1], s=12, alpha=0.7,
                          color=cmap_cat(k), label=name, linewidths=0)
        ax_sc.legend(fontsize=7, markerscale=1.5, framealpha=0.8)
    elif durations is not None:
        norm = Normalize(durations.min(), durations.max())
        ax_sc.scatter(embedding[:, 0], embedding[:, 1],
                      c=durations, cmap="viridis", s=12, alpha=0.7,
                      linewidths=0, norm=norm)
    else:
        ax_sc.scatter(embedding[:, 0], embedding[:, 1], s=12, alpha=0.7,
                      color="steelblue", linewidths=0)

    ax_sc.set_xticks([]);  ax_sc.set_yticks([])
    ax_sc.set_facecolor("white")
    for spine in ax_sc.spines.values():
        spine.set_visible(False)

    # dashed lines from each scatter point to its border box
    dash_ax = fig.add_axes([0, 0, 1, 1])
    dash_ax.set_xlim(0, 1);  dash_ax.set_ylim(0, 1)
    dash_ax.patch.set_alpha(0)
    dash_ax.set_xticks([]);  dash_ax.set_yticks([])
    for sp in dash_ax.spines.values():
        sp.set_visible(False)
    dash_ax.set_zorder(0)

    # helper: data coords in scatter → figure [0,1]
    def to_fig(ax, xd, yd):
        ap = ax.get_position()
        xl, xr = ax.get_xlim();  yb, yt = ax.get_ylim()
        fx = ap.x0 + (xd - xl) / (xr - xl) * ap.width
        fy = ap.y0 + (yd - yb) / (yt - yb) * ap.height
        return fx, fy

    for bi, (r, c) in enumerate(box_rc):
        x0 = PAD + c * (cell_w + PAD) + cell_w / 2
        y0 = 1 - PAD - (r + 1) * (cell_h + PAD) + cell_h / 2
        idx = chosen[bi]
        px, py = to_fig(ax_sc, embedding[idx, 0], embedding[idx, 1])
        dash_ax.plot([px, x0], [py, y0], "--", color="grey", lw=0.6, alpha=0.5)

    plt.suptitle("Vocalization UMAP — border grid", y=1.01, fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved border-grid figure → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Export to .mat for MATLAB frontend
# ---------------------------------------------------------------------------

def export_for_matlab(
    embedding: np.ndarray,
    audio_list: List[np.ndarray],
    sr: int,
    durations: np.ndarray,
    out_path: str,
    labels: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_ms: float = 1.0,
    win_ms: float = 8.0,
    fmin: float = 500.0,
    fmax: Optional[float] = None,
    spec_height: int = 128,
    spec_width: int = 128,
) -> None:
    """
    Export embedding coordinates and spectrograms to a .mat file for use with
    the MATLAB function plotSpecEmbedings_Estim_dot_warbler_audio_dyna.

    The saved struct fields match the clusteringData table schema:
        embedX, embedY, Duration, spectrograms (H x W x N uint8),
        cluster_labels, sr

    In MATLAB, load with:
        d = load('ouroboros_embedding.mat');
        % then pass d.embedX, d.embedY etc to your visualization function

    Parameters
    ----------
    spec_height/spec_width : thumbnail spectrogram size saved into the mat file.
        Use larger values (e.g. 256×512) for higher-quality MATLAB display.
    """
    from scipy.io import savemat
    from skimage.transform import resize as sk_resize

    spec_kw = dict(n_fft=n_fft, hop_ms=hop_ms, win_ms=win_ms,
                   fmin=fmin, fmax=fmax, preemph=0.97)

    N = len(audio_list)
    spectrograms = np.zeros((spec_height, spec_width, N), dtype=np.uint8)

    print(f"Computing {N} spectrograms for MATLAB export...")
    for i, aud in enumerate(tqdm(audio_list)):
        try:
            S_db, _, _ = make_highres_spec(aud.ravel(), sr, **spec_kw)
            # normalize to [0,1] then resize to fixed thumbnail
            S_n  = (S_db - S_db.min()) / (S_db.ptp() + 1e-8)
            S_rs = sk_resize(S_n, (spec_height, spec_width), anti_aliasing=True)
            spectrograms[:, :, i] = (S_rs * 255).astype(np.uint8)
        except Exception:
            pass

    mat_dict = {
        "embedX":      embedding[:, 0].astype(np.float64),
        "embedY":      embedding[:, 1].astype(np.float64),
        "Duration":    durations.astype(np.float64),
        "spectrograms": spectrograms,
        "sr":          float(sr),
        "n_fft":       float(n_fft),
        "hop_ms":      float(hop_ms),
        "win_ms":      float(win_ms),
    }
    if labels is not None:
        mat_dict["cluster_labels"] = labels.astype(np.float64)

    savemat(out_path, mat_dict)
    print(f"MATLAB export saved → {out_path}")
    print(f"  Load in MATLAB:  d = load('{out_path}');")
