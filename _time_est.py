"""Quick timing estimate: runs 3 real training epochs and extrapolates."""
import time, numpy as np, torch

if __name__ == "__main__":
    from data.load_data import get_segmented_audio
    from data.data_utils import get_loaders
    from model.kernels import fullPolyModule
    from model.model import Ouroboros
    from torch.optim import Adam
    from utils import sse

    AUDIO_DIR, SEG_DIR = "data/audio", "data/segs"
    MAX_VOCS, CONTEXT, BATCH_SIZE = 200, 0.25, 8  # small run: scale up later
    N_LAMBDAS, N_EPOCHS_FULL = 7, 500

    print("Loading data...")
    chunks, sr = get_segmented_audio(AUDIO_DIR, SEG_DIR, audio_id=".WAV",
        training=True, context_len=CONTEXT, max_vocs=MAX_VOCS, seed=42)
    print(f"  {len(chunks)} chunks loaded")
    dt = 1/sr
    dls = get_loaders(np.stack(chunks), batch_size=BATCH_SIZE, num_workers=0,
                      cv=True, seed=42, dt=dt)
    n_train = len(dls['train'])
    n_val   = len(dls['val'])
    print(f"  Train batches/epoch: {n_train} | Val batches/epoch: {n_val}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    kernel = fullPolyModule(nTerms=15, device=device, x_dim=1, z_dim=2,
                            activation=lambda x: x, lam=1.01)
    model  = Ouroboros(d_data=1, n_layers=3, d_state=1, d_conv=4,
                       expand_factor=10, tau=dt, kernel=kernel).to(device)
    opt    = Adam(model.parameters(), lr=1e-3)
    loss_fn = lambda y, yhat: sse(yhat, y, reduction="mean")

    print("Timing 2 train epochs (train only, no val)...")
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()

    for epoch in range(2):
        model.train()
        for batch in dls['train']:
            x, dxdt, dx2dt2 = batch
            x, dxdt = x.to(device).float(), dxdt.to(device).float()
            dx2 = dx2dt2.to(device).float() / (dt**2) * model.tau**2
            opt.zero_grad()
            yhat, weights = model(x, dxdt, dt, False)
            # reg penalty (mirrors train loop)
            B, L, P, _ = weights.shape
            lam_mat = torch.arange(P, dtype=torch.float32, device=device
                )[None,None,:,None].expand(B,L,-1,P)
            w = model.kernel.lam ** (lam_mat + lam_mat.transpose(-1,-2))
            loss = loss_fn(dx2, yhat[:,:L,:]) + (w*weights**2).sum(dim=(-1,-2,-3)).mean()
            loss.backward(); opt.step()
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - t0
    per_epoch = elapsed / 2
    # val adds ~n_val/n_train overhead each epoch
    val_ratio = n_val / n_train
    per_epoch_with_val = per_epoch * (1 + val_ratio)

    # scale batches per epoch to full 1000-voc run: ~600 train chunks / bsz=8 = 75 batches
    # (measured with MAX_VOCS=200 so n_train batches are smaller — scale accordingly)
    scale = 75 / n_train  # approx ratio for full dataset
    full_epoch = per_epoch_with_val * scale

    print(f"\n--- Results (scaled to 1000 vocs, bsz=8) ---")
    print(f"Measured {n_train} train batches + {n_val} val batches per epoch")
    print(f"Per epoch (measured):       {per_epoch:.1f}s train-only")
    print(f"Per epoch (w/ val, scaled): {full_epoch:.1f}s")
    print(f"Per lambda ({N_EPOCHS_FULL} ep):  {full_epoch*N_EPOCHS_FULL/60:.0f} min")
    print(f"Total ({N_LAMBDAS} lambdas):      {full_epoch*N_EPOCHS_FULL*N_LAMBDAS/3600:.1f}h")
    print(f"\nMax epochs in 12h:  {int(43200/(full_epoch*N_LAMBDAS))}")
