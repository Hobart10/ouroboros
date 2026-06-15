"""Quick pipeline test: 50 vocs, 8 epochs, 3 spectrograms."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AUDIO_DIR  = "data/audio"
SEG_DIR    = "data/segs"
MODEL_DIR  = "trainingTest_quicktest"
OUT_DIR    = os.path.join(MODEL_DIR, "output")
AUDIO_EXT  = ".WAV"
MAX_VOCS   = 50
CONTEXT    = 0.25
N_EPOCHS   = 8
BATCH_SIZE = 32

os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    from data.load_data import get_segmented_audio
    from data.data_utils import get_loaders

    print("=== STEP 1: Data ===")
    chunks, sr = get_segmented_audio(
        AUDIO_DIR, SEG_DIR, audio_id=AUDIO_EXT,
        training=True, context_len=CONTEXT, max_vocs=MAX_VOCS, seed=42)
    print(f"  {len(chunks)} chunks, sr={sr}")
    dt = 1 / sr
    dls = get_loaders(np.stack(chunks), batch_size=BATCH_SIZE, num_workers=0,
                      cv=True, seed=42, dt=dt)
    print(f"  Train:{len(dls['train'])}  Val:{len(dls['val'])}  Test:{len(dls['test'])}")

    print("\n=== STEP 2: Training ===")
    from train.train_model import train_model
    best_model = train_model(
        audio_dirs=[AUDIO_DIR], seg_dirs=[SEG_DIR], model_dir=MODEL_DIR,
        n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, context_len=CONTEXT,
        max_vocs=MAX_VOCS, save_freq=999, max_jobs=0,
        audio_id=AUDIO_EXT, vis_freq=0,
    )
    print("  Training done.")

    print("\n=== STEP 3: Evaluation ===")
    from train.eval import eval_model_error
    best_model.eval()
    (r2_tr, r2_val), (sd_tr, sd_val), _ = eval_model_error(
        dls, best_model, dt, comparison="val")
    print(f"  Train R2={r2_tr:.4f} +/- {sd_tr:.4f}")
    print(f"  Val   R2={r2_val:.4f} +/- {sd_val:.4f}")

    print("\n=== STEP 4: Spectrograms ===")
    from train.eval import integrate_model_d2
    from utils import get_spec
    analysis, _ = get_segmented_audio(
        AUDIO_DIR, SEG_DIR, audio_id=AUDIO_EXT,
        training=False, padding=0.05, max_vocs=3, seed=42)
    for i, seg in enumerate(analysis[:3]):
        audio = seg.squeeze()
        try:
            recon = integrate_model_d2(best_model, audio, dt, verbose=False)
        except Exception as e:
            print(f"  voc {i} failed: {e}"); continue
        s_o, t1, f1, _ = get_spec(audio, sr, onset=0, offset=len(audio)/sr,
                                   normalize=False, interp=False)
        s_r, t2, f2, _ = get_spec(recon,  sr, onset=0, offset=len(recon)/sr,
                                   normalize=False, interp=False)
        vmin = min(s_o.min(), s_r.min())
        vmax = max(s_o.max(), s_r.max())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(s_o, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                       extent=[t1[0], t1[-1], f1[0], f1[-1]])
        axes[0].set_title(f"Original [voc {i}]")
        axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Freq (Hz)")
        axes[1].imshow(s_r, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                       extent=[t2[0], t2[-1], f2[0], f2[-1]])
        axes[1].set_title(f"Reconstructed [voc {i}]")
        axes[1].set_xlabel("Time (s)")
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"spec_{i}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"  Saved {out}")
    print("\nDONE")
