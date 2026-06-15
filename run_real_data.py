"""
Small-scale test run on real data.

Run from the project root:
    uv run python run_real_data.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

AUDIO_DIR   = "data/audio"
SEG_DIR     = "data/segs"
MODEL_DIR   = "trainingTest_02"
OUT_DIR     = os.path.join(MODEL_DIR, "output")
AUDIO_EXT   = ".WAV"
MAX_VOCS    = 500
CONTEXT     = 0.25   # seconds per training chunk
N_EPOCHS    = 400
BATCH_SIZE  = 64     # larger batch → fewer batches/epoch → better GPU utilization
NUM_WORKERS = 0      # 0 = main process only (avoids Windows multiprocessing overhead)
VIS_FREQ    = 0      # disable per-batch SVG plots during training (saves ~1000 files)
VAL_FREQ    = 10     # validate every 10 epochs (reduces overhead vs every-epoch default)
SAVE_FREQ   = 50     # checkpoint every 50 epochs (reduces disk I/O)

os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":

    # ── 1. Quick data check ───────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Data check")
    print("=" * 60)

    from data.load_data import get_segmented_audio
    from data.data_utils import get_loaders

    chunks, sr = get_segmented_audio(
        AUDIO_DIR, SEG_DIR,
        audio_id=AUDIO_EXT,
        training=True,
        context_len=CONTEXT,
        max_vocs=MAX_VOCS,
        seed=42,
    )
    print(f"  Loaded {len(chunks)} chunks, shape: {chunks[0].shape}, sr: {sr} Hz")
    assert len(chunks) > 0, "No chunks loaded — check AUDIO_DIR / SEG_DIR / AUDIO_EXT"

    dt = 1 / sr
    data = np.stack(chunks, axis=0)
    dls  = get_loaders(data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                       cv=True, seed=42, dt=dt)
    print(f"  Train: {len(dls['train'])} batches  |  "
          f"Val: {len(dls['val'])} batches  |  "
          f"Test: {len(dls['test'])} batches")

    # ── 2. Training ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Training")
    print("=" * 60)

    from train.train_model import train_model

    best_model = train_model(
        audio_dirs=[AUDIO_DIR],
        seg_dirs=[SEG_DIR],
        model_dir=MODEL_DIR,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        context_len=CONTEXT,
        max_vocs=MAX_VOCS,
        save_freq=SAVE_FREQ,
        max_jobs=NUM_WORKERS,
        audio_id=AUDIO_EXT,
        vis_freq=VIS_FREQ,
        val_freq=VAL_FREQ,
    )
    print("  Training complete.")

    # ── 3. Evaluation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Evaluation (R²)")
    print("=" * 60)

    from train.eval import eval_model_error

    best_model.eval()
    (mean_train_r2, mean_val_r2), (sd_train, sd_val), _ = eval_model_error(
        dls, best_model, dt, comparison="val"
    )
    print(f"  Train R²: {mean_train_r2:.4f} ± {sd_train:.4f}")
    print(f"  Val   R²: {mean_val_r2:.4f}   ± {sd_val:.4f}")

    # ── 4. Spectrogram visualization ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Spectrogram visualization")
    print("=" * 60)

    from train.eval import integrate_model_d2
    from utils      import get_spec

    analysis_chunks, sr = get_segmented_audio(
        AUDIO_DIR, SEG_DIR,
        audio_id=AUDIO_EXT,
        training=False,
        padding=0.05,
        max_vocs=5,
        seed=42,
    )

    for i, seg in enumerate(analysis_chunks[:5]):
        audio = seg.squeeze()
        print(f"  Integrating voc {i} ({len(audio)/sr:.2f}s, {len(audio)} samples)...")

        try:
            recon = integrate_model_d2(best_model, audio, dt, verbose=False)
        except Exception as e:
            print(f"  Integration failed for voc {i}: {e}")
            continue

        s_orig,  t1, f1, _ = get_spec(audio, sr, onset=0, offset=len(audio)/sr,
                                       normalize=False, interp=False)
        s_recon, t2, f2, _ = get_spec(recon,  sr, onset=0, offset=len(recon)/sr,
                                       normalize=False, interp=False)

        vmin = min(s_orig.min(), s_recon.min())
        vmax = max(s_orig.max(), s_recon.max())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(s_orig,  origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                       extent=[t1[0], t1[-1], f1[0], f1[-1]])
        axes[0].set_title(f"Original [voc {i}]")
        axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Freq (Hz)")

        axes[1].imshow(s_recon, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                       extent=[t2[0], t2[-1], f2[0], f2[-1]])
        axes[1].set_title(f"Reconstructed [voc {i}]")
        axes[1].set_xlabel("Time (s)")

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"spec_comparison_{i}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("DONE — spectrograms saved to", OUT_DIR)
    print("=" * 60)
