"""Quick GPU profiling — measures actual per-epoch time with bsz=64."""
import torch, time, numpy as np
from data.load_data import get_segmented_audio
from data.data_utils import get_loaders
from model.kernels import fullPolyModule
from model.model import Ouroboros
from torch.optim import Adam
from utils import sse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

chunks, sr = get_segmented_audio("data/audio", "data/segs", audio_id=".WAV",
    training=True, context_len=0.25, max_vocs=100, seed=42)
dt = 1 / sr
dls = get_loaders(np.stack(chunks), batch_size=64, num_workers=0, cv=True, seed=42, dt=dt)
n_train = len(dls["train"])
n_val   = len(dls["val"])
print(f"Batches/epoch: train={n_train}  val={n_val}  (100 vocs, bsz=64)")

kernel = fullPolyModule(nTerms=15, device=DEVICE, x_dim=1, z_dim=2,
                        activation=lambda x: x, lam=1.01)
model  = Ouroboros(d_data=1, n_layers=3, d_state=1, d_conv=4,
                   expand_factor=10, tau=dt, kernel=kernel).to(DEVICE)
opt    = Adam(model.parameters(), lr=1e-3)

all_cuda = all(p.device.type == DEVICE for p in model.parameters())
print(f"All params on {DEVICE}: {all_cuda}")

if DEVICE == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
for ep in range(3):
    model.train()
    for batch in dls["train"]:
        x, dxdt, dx2dt2 = batch
        x    = x.to(DEVICE).float()
        dxdt = dxdt.to(DEVICE).float()
        dx2  = dx2dt2.to(DEVICE).float() / (dt**2) * model.tau**2
        opt.zero_grad()
        yhat, weights = model(x, dxdt, dt, False)
        B, L, P, _ = weights.shape
        lam_mat = torch.arange(P, dtype=torch.float32, device=DEVICE)[None, None, :, None].expand(B, L, -1, P)
        w = model.kernel.lam ** (lam_mat + lam_mat.transpose(-1, -2))
        loss = sse(dx2, yhat[:, :L, :], reduction="mean") + (w * weights**2).sum(dim=(-1,-2,-3)).mean()
        loss.backward()
        opt.step()

if DEVICE == "cuda":
    torch.cuda.synchronize()
elapsed = (time.time() - t0) / 3
print(f"\nPer epoch (100 vocs, bsz=64): {elapsed:.1f}s")
if DEVICE == "cuda":
    print(f"GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GPU mem peak:      {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# Scale to full run: 500 vocs → ~n_train * 5 batches
scale = (n_train * 5) / n_train  # approximate 500/100 = 5x more batches
est = elapsed * scale * 400 * 7 / 3600
print(f"\nEstimate 500 vocs / 400 ep / 7 lam: {est:.1f}h (train-only, no val/save overhead)")
