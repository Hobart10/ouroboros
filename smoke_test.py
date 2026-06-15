"""
Smoke test: runs the full pipeline on a tiny synthetic dataset
to verify the environment and code work before using real data.

Run from the project root:
    uv run python smoke_test.py
    # or, if venv is activated:
    python smoke_test.py
"""

import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# ── 0. Config ────────────────────────────────────────────────────────────────
AUDIO_DIR  = "smoke_data/audio"
SEG_DIR    = "smoke_data/segs"
MODEL_DIR  = "smoke_data/model"
OUT_DIR    = "smoke_data/output"
SR         = 48000
N_VOCS     = 10        # tiny dataset
VOC_LEN    = 0.5       # seconds per vocalization
CONTEXT    = 0.25      # training chunk length (must be <= VOC_LEN)
N_EPOCHS   = 5         # just enough to confirm training runs
BATCH_SIZE = 4
NUM_WORKERS = 0        # 0 = main process only; required on Windows without __main__ guard

for d in [AUDIO_DIR, SEG_DIR, MODEL_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)


# Windows multiprocessing requires all DataLoader worker code to be inside __main__
if __name__ == "__main__":

    # ── 1. Generate tiny synthetic dataset ───────────────────────────────────
    print("=" * 60)
    print("STEP 1: Generating synthetic audio data")
    print("=" * 60)

    t = np.arange(0, VOC_LEN, 1 / SR)
    rng = np.random.default_rng(42)

    for i in range(N_VOCS):
        freq = rng.uniform(1000, 4000)
        amp  = rng.uniform(0.1, 0.5)
        noise = rng.normal(0, 0.01, size=t.shape)
        audio = (amp * np.sin(2 * np.pi * freq * t) + noise).astype(np.float32)

        sf.write(os.path.join(AUDIO_DIR, f"test_{i:03d}.wav"), audio, SR)
        np.savetxt(
            os.path.join(SEG_DIR, f"test_{i:03d}.txt"),
            np.array([[0.05, VOC_LEN - 0.05]]),
        )

    print(f"  Created {N_VOCS} .wav files in '{AUDIO_DIR}'")
    print(f"  Created {N_VOCS} .txt seg files in '{SEG_DIR}'")

    # ── 2. Load & verify data ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading & verifying data")
    print("=" * 60)

    from data.load_data import get_segmented_audio
    from data.data_utils import get_loaders

    chunks, sr = get_segmented_audio(
        AUDIO_DIR, SEG_DIR,
        training=True,
        context_len=CONTEXT,
        seed=42,
    )
    print(f"  Loaded {len(chunks)} chunks, shape: {chunks[0].shape}, sr: {sr}")
    assert len(chunks) > 0, "No audio chunks loaded — check audio/seg dirs"

    data = np.stack(chunks, axis=0)
    dt   = 1 / sr
    dls  = get_loaders(data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                       cv=True, seed=42, dt=dt)
    print(f"  Train batches: {len(dls['train'])}, "
          f"Val batches: {len(dls['val'])}, "
          f"Test batches: {len(dls['test'])}")

    # ── 3. Train (smoke: 5 epochs) ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Training (smoke — 5 epochs)")
    print("=" * 60)

    from train.train_model import train_model

    best_model = train_model(
        audio_dirs=[AUDIO_DIR],
        seg_dirs=[SEG_DIR],
        model_dir=MODEL_DIR,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        context_len=CONTEXT,
        save_freq=N_EPOCHS,
        max_vocs=N_VOCS,
        max_jobs=0,        # num_workers=0: no subprocess spawning on Windows
    )
    print("  Training complete.")

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation (R²)")
    print("=" * 60)

    from train.eval import eval_model_error

    model = best_model
    model.eval()

    (mean_train_r2, mean_val_r2), (sd_train, sd_val), _ = eval_model_error(
        dls, model, dt, comparison="val"
    )
    print(f"  Train R²: {mean_train_r2:.4f} ± {sd_train:.4f}")
    print(f"  Val   R²: {mean_val_r2:.4f}   ± {sd_val:.4f}")

    # ── 5. Spectrogram visualization ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Spectrogram visualization")
    print("=" * 60)

    from train.eval import integrate_model_d2
    from utils      import get_spec

    analysis_chunks, sr = get_segmented_audio(
        AUDIO_DIR, SEG_DIR,
        training=False,
        padding=0.02,
        seed=42,
    )

    for i, seg in enumerate(analysis_chunks[:3]):
        audio = seg.squeeze()
        print(f"  Integrating vocalization {i} (len={len(audio)} samples)...")

        try:
            recon = integrate_model_d2(model, audio, dt, verbose=False)
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
        axes[0].set_title(f"Original  [voc {i}]")
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
    print("SMOKE TEST PASSED — all steps completed successfully.")
    print(f"Spectrograms saved to '{OUT_DIR}/'")
    print("=" * 60)
