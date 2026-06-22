"""
Run the full DR/UMAP pipeline on a set of vocalizations using a trained Ouroboros.

Usage
-----
    python run_dr_umap.py \
        --model_dir  path/to/model_checkpoints \
        --audio_dir  path/to/audio \
        --seg_dir    path/to/segments \
        --out_dir    path/to/output \
        [--max_vocs  2000] \
        [--method    full] \
        [--padding   0.05]

Outputs (all in --out_dir)
--------------------------
    embeddings.npz     — raw latents + aggregated features + UMAP coords
    umap_duration.png  — scatter coloured by call duration
    metrics.npz        — per-call reconstruction metrics (r2, snr, ssim, mse)
    metrics_report.txt — median ± IQR summary
"""

import argparse
import os
import numpy as np

from data.load_data import get_segmented_audio
from train.train import load_model
from train.eval import integrate_model_d2

from visualization.dr_umap import (
    extract_ouroboros_embeddings,
    aggregate_latents,
    compute_umap,
    plot_umap,
    batch_reconstruction_metrics,
    report_metrics,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",  required=True)
    p.add_argument("--audio_dir",  required=True)
    p.add_argument("--seg_dir",    required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--max_vocs",   type=int,   default=2000)
    p.add_argument("--padding",    type=float, default=0.05,
                   help="Context padding around each vocalization (s)")
    p.add_argument("--method",     default="full",
                   choices=["mean_std", "percentile", "full"],
                   help="Latent aggregation method")
    p.add_argument("--n_neighbors",type=int,   default=15)
    p.add_argument("--min_dist",   type=float, default=0.1)
    p.add_argument("--no_recon",   action="store_true",
                   help="Skip reconstruction metrics (faster)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load audio (variable length, analysis mode)
    # ------------------------------------------------------------------
    print("Loading vocalizations...")
    audio_list, sr = get_segmented_audio(
        args.audio_dir, args.seg_dir,
        training=False,
        max_vocs=args.max_vocs,
        padding=args.padding,
    )
    print(f"  loaded {len(audio_list)} calls at {sr} Hz")
    dt = 1.0 / sr

    durations = np.array([a.shape[1] / sr for a in audio_list])
    print(f"  duration range: {durations.min():.3f}s – {durations.max():.3f}s")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print("Loading model...")
    model, _, _, epoch = load_model(args.model_dir)
    model.eval()
    print(f"  loaded checkpoint epoch {epoch}")

    # ------------------------------------------------------------------
    # 3. Extract latents (variable-length, no padding needed)
    # ------------------------------------------------------------------
    print("Extracting Ouroboros latents...")
    latents = extract_ouroboros_embeddings(
        model, audio_list, sr,
        smoothing=True,
        include_weights=False,
    )

    # ------------------------------------------------------------------
    # 4. Aggregate → fixed-length embedding
    # ------------------------------------------------------------------
    print(f"Aggregating latents with method='{args.method}'...")
    X = aggregate_latents(latents, method=args.method)
    print(f"  feature matrix: {X.shape}")

    # ------------------------------------------------------------------
    # 5. UMAP
    # ------------------------------------------------------------------
    print("Running UMAP...")
    embedding = compute_umap(
        X,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    # ------------------------------------------------------------------
    # 6. Save embeddings
    # ------------------------------------------------------------------
    np.savez(
        os.path.join(args.out_dir, "embeddings.npz"),
        features=X,
        umap=embedding,
        durations=latents["durations"],
    )
    print(f"  saved embeddings.npz")

    # ------------------------------------------------------------------
    # 7. Plot — coloured by duration
    # ------------------------------------------------------------------
    fig = plot_umap(
        embedding,
        durations=latents["durations"],
        title=f"Ouroboros UMAP  (n={len(audio_list)}, epoch={epoch})",
        save_path=os.path.join(args.out_dir, "umap_duration.png"),
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8. Reconstruction metrics (optional)
    # ------------------------------------------------------------------
    if not args.no_recon:
        print("Computing reconstruction metrics...")
        originals = [a.squeeze() for a in audio_list]
        reconstructions = []

        from tqdm import tqdm
        for aud in tqdm(audio_list, desc="Integrating"):
            recon = integrate_model_d2(model, aud.squeeze(), dt, smoothing=True, verbose=False)
            reconstructions.append(recon)

        metrics = batch_reconstruction_metrics(originals, reconstructions, sr)

        np.savez(os.path.join(args.out_dir, "metrics.npz"), **metrics)

        report_path = os.path.join(args.out_dir, "metrics_report.txt")
        import sys
        with open(report_path, "w") as f:
            old_stdout = sys.stdout
            sys.stdout = f
            print(f"Reconstruction metrics  (n={len(originals)} calls)\n")
            report_metrics(metrics)
            sys.stdout = old_stdout
        print(f"  saved metrics_report.txt")

        # quick summary to console
        report_metrics(metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
