"""
Full DR/UMAP pipeline: latent extraction → UMAP → visualization → metrics.

Usage
-----
    python run_dr_umap.py \
        --model_dir  path/to/checkpoints \
        --audio_dir  path/to/audio \
        --seg_dir    path/to/segments \
        --out_dir    path/to/output \
        [--max_vocs    2000] \
        [--padding     0.05] \
        [--method      full] \
        [--n_fft       2048] \
        [--hop_ms      1.0]  \
        [--win_ms      8.0]  \
        [--fmin        500]  \
        [--fmax        12000]\
        [--no_recon        ] \
        [--export_matlab   ] \
        [--interactive     ]

Outputs (all in --out_dir)
--------------------------
    embeddings.npz          raw latents + aggregated features + UMAP
    umap_duration.png       scatter coloured by duration (static)
    umap_border_grid.png    MATLAB-style N×N border spectrogram layout
    metrics.npz             per-call reconstruction metrics
    metrics_report.txt      median ± IQR + Wilcoxon / Kruskal-Wallis
    metrics_violins.png     violin plots of all metrics
    metrics_paired.png      orig vs recon scatter (SPL, F0)
    metrics_vs_duration.png metric values vs call duration
    pca_orig_recon.png      PCA: orig vs recon latent space
    ouroboros_embedding.mat MATLAB-compatible export (with --export_matlab)
"""

import argparse
import os
import numpy as np
import matplotlib
from tqdm import tqdm
matplotlib.use("Agg")   # non-interactive backend for batch runs
import matplotlib.pyplot as plt

from data.load_data import get_segmented_audio
from train.train import load_model
from visualization.recon_vis import reconstruct_data

from visualization.dr_umap import (
    extract_ouroboros_embeddings,
    aggregate_latents,
    compute_umap,
    plot_umap,
)
from visualization.recon_metrics import (
    batch_all_metrics,
    statistical_report,
    plot_metric_violins,
    plot_paired_scatter,
    plot_metric_vs_duration,
    plot_orig_recon_pca,
)
from visualization.scatter_vis import (
    plot_scatter_interactive,
    plot_border_grid,
    export_for_matlab,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",    required=True)
    p.add_argument("--audio_dir",    required=True)
    p.add_argument("--seg_dir",      required=True)
    p.add_argument("--out_dir",      required=True)
    p.add_argument("--max_vocs",     type=int,   default=2000)
    p.add_argument("--padding",      type=float, default=0.05,
                   help="Context padding around each vocalization (s)")
    p.add_argument("--method",       default="full",
                   choices=["mean_std", "percentile", "full"],
                   help="Latent aggregation method")
    p.add_argument("--n_neighbors",  type=int,   default=15)
    p.add_argument("--min_dist",     type=float, default=0.1)
    # spectrogram parameters (shared by metrics + visualization)
    p.add_argument("--n_fft",        type=int,   default=2048,
                   help="FFT size; increase to 4096 for finer frequency resolution")
    p.add_argument("--hop_ms",       type=float, default=1.0,
                   help="STFT hop in ms; decrease to 0.5 for finer time resolution")
    p.add_argument("--win_ms",       type=float, default=8.0,
                   help="STFT window in ms")
    p.add_argument("--fmin",         type=float, default=500.0)
    p.add_argument("--fmax",         type=float, default=None)
    p.add_argument("--no_recon",     action="store_true",
                   help="Skip reconstruction metrics (faster)")
    p.add_argument("--export_matlab",action="store_true",
                   help="Export .mat file for MATLAB visualization")
    p.add_argument("--interactive",  action="store_true",
                   help="Show interactive matplotlib window. Requires a display "
                        "(X11 forwarding or local run). Do not use on headless cluster.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    spec_kw = dict(
        n_fft=args.n_fft,
        hop_ms=args.hop_ms,
        win_ms=args.win_ms,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    # ------------------------------------------------------------------
    # 1. Load variable-length vocalizations
    # ------------------------------------------------------------------
    print("Loading vocalizations...")
    audio_list, sr = get_segmented_audio(
        args.audio_dir, args.seg_dir,
        training=False,
        max_vocs=args.max_vocs,
        padding=args.padding,
    )
    print(f"  {len(audio_list)} calls at {sr} Hz")
    dt = 1.0 / sr

    durations = np.array([a.shape[1] / sr for a in audio_list])
    print(f"  duration range: {durations.min():.3f}s – {durations.max():.3f}s"
          f"  (median {np.median(durations):.3f}s)")

    # ------------------------------------------------------------------
    # 2. Load trained model
    # ------------------------------------------------------------------
    print("Loading model...")
    model, _, _, epoch = load_model(args.model_dir)
    model.eval()
    print(f"  checkpoint epoch {epoch}")

    # ------------------------------------------------------------------
    # 3. Extract Ouroboros latents (variable-length, no padding)
    # ------------------------------------------------------------------
    print("Extracting latents (omega, gamma)...")
    latents = extract_ouroboros_embeddings(
        model, audio_list, sr, smoothing=True, include_weights=False,
    )

    # ------------------------------------------------------------------
    # 4. Aggregate time-series → fixed-length embedding
    # ------------------------------------------------------------------
    print(f"Aggregating with method='{args.method}'...")
    X = aggregate_latents(latents, method=args.method)
    print(f"  feature matrix: {X.shape}")

    # ------------------------------------------------------------------
    # 5. UMAP
    # ------------------------------------------------------------------
    print("Running UMAP...")
    embedding = compute_umap(X, n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    np.savez(
        os.path.join(args.out_dir, "embeddings.npz"),
        features=X, umap=embedding, durations=durations,
    )
    print("  saved embeddings.npz")

    # ------------------------------------------------------------------
    # 6. Scatter plots
    # ------------------------------------------------------------------
    print("Generating scatter visualizations...")

    # static scatter coloured by duration
    fig = plot_umap(
        embedding, durations=durations,
        title=f"Ouroboros UMAP  (n={len(audio_list)}, epoch={epoch})",
        save_path=os.path.join(args.out_dir, "umap_duration.png"),
    )
    plt.close(fig)

    # MATLAB-style border-grid
    print("  border-grid layout...")
    fig = plot_border_grid(
        embedding, audio_list, sr,
        n_grid=6, durations=durations,
        save_path=os.path.join(args.out_dir, "umap_border_grid.png"),
        **spec_kw,
    )
    plt.close(fig)

    # interactive window — requires display (X11 / local run only)
    if args.interactive:
        import os
        if not os.environ.get("DISPLAY") and not os.name == "nt":
            print("WARNING: --interactive requested but DISPLAY env var is not set. "
                  "Skipping interactive window. Use X11 forwarding (ssh -X) or run locally.")
        else:
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as _plt2
            fig_i = plot_scatter_interactive(
                embedding, audio_list, sr,
                durations=durations, **spec_kw,
            )
            _plt2.show()

    # ------------------------------------------------------------------
    # 7. Reconstruction + metrics
    # ------------------------------------------------------------------
    if not args.no_recon:
        print("Reconstructing vocalizations...")
        originals, reconstructions = [], []
        for aud in tqdm(audio_list, desc="Integrating"):
            orig  = aud.squeeze()
            recon = reconstruct_data(model, sr, orig)
            originals.append(orig)
            reconstructions.append(recon)

        print("Computing all reconstruction metrics...")
        metrics = batch_all_metrics(
            originals, reconstructions, sr,
            n_fft=args.n_fft, hop_ms=args.hop_ms, win_ms=args.win_ms,
        )
        np.savez(os.path.join(args.out_dir, "metrics.npz"), **metrics)

        # stratify by duration (short / medium / long)
        dur_bins = np.percentile(durations, [0, 33, 67, 100])
        dur_bins[0]  = 0.0
        dur_bins[-1] = np.inf

        statistical_report(
            metrics,
            duration_bins=dur_bins,
            out_path=os.path.join(args.out_dir, "metrics_report.txt"),
        )

        print("Generating metric plots...")
        fig = plot_metric_violins(
            metrics,
            save_path=os.path.join(args.out_dir, "metrics_violins.png"),
        )
        plt.close(fig)

        fig = plot_paired_scatter(
            metrics,
            save_path=os.path.join(args.out_dir, "metrics_paired.png"),
        )
        plt.close(fig)

        fig = plot_metric_vs_duration(
            metrics,
            save_path=os.path.join(args.out_dir, "metrics_vs_duration.png"),
        )
        plt.close(fig)

        # PCA: do reconstructions cluster with originals?
        print("Extracting latents from reconstructions for PCA comparison...")
        recon_audio_list = [r[None, :, None] for r in reconstructions]
        latents_recon = extract_ouroboros_embeddings(
            model, recon_audio_list, sr, smoothing=True, include_weights=False,
        )
        X_recon = aggregate_latents(latents_recon, method=args.method)

        fig = plot_orig_recon_pca(
            X, X_recon,
            save_path=os.path.join(args.out_dir, "pca_orig_recon.png"),
        )
        plt.close(fig)

        print("  saved all metric plots")

    # ------------------------------------------------------------------
    # 8. MATLAB export
    # ------------------------------------------------------------------
    if args.export_matlab:
        print("Exporting to .mat for MATLAB visualization...")
        export_for_matlab(
            embedding, audio_list, sr, durations,
            out_path=os.path.join(args.out_dir, "ouroboros_embedding.mat"),
            **spec_kw,
            spec_height=256,
            spec_width=256,
        )

    print("\nDone.  Outputs in:", args.out_dir)


if __name__ == "__main__":
    main()
