import os, math, argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import tensorflow_datasets as tfds

# ----------------------------
# Data: BAIR (TFDS) -> [B, T, C, H, W], values in [-1, 1]
# ----------------------------
def iter_bair(batch=8, frames=16, split="train"):
    ds = tfds.load("bair_robot_pushing_small", split=split, shuffle_files=True, as_supervised=False)
    ds = ds.shuffle(2048).repeat()
    buf = []
    for ex in tfds.as_numpy(ds):
        v = ex["image_main"]  # (≈30, 64, 64, 3), uint8
        if v.shape[0] < frames: continue
        s = np.random.randint(0, v.shape[0] - frames + 1)
        clip = v[s:s+frames]  # [T,H,W,C]
        x = torch.from_numpy(clip).permute(0,3,1,2).float() / 127.5 - 1.0  # [T,C,H,W]
        buf.append(x)
        if len(buf) == batch:
            yield torch.stack(buf, 0)  # [B,T,C,H,W]
            buf = []

# ----------------------------
# Diffusion schedules (cosine)
# ----------------------------
def cosine_beta_schedule(n_steps=1000, s=0.008):
    # https://arxiv.org/abs/2102.09672
    steps = n_steps + 1
    xs = torch.linspace(0, n_steps, steps)
    alphas_cumprod = torch.cos(((xs / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)

class DiffusionSchedule:
    def __init__(self, n_steps=1000, device="cpu"):
        betas = cosine_beta_schedule(n_steps).to(device)   # [T]
        alphas = 1.0 - betas                               # [T]
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0) # ᾱ_t, [T]
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # q(x_{t-1} | x_t, x_0) variance
        post_var = torch.empty_like(betas)
        post_var[0] = betas[0]  # or a small eps
        post_var[1:] = betas[1:] * (1.0 - self.alphas_cumprod[:-1]) / (1.0 - self.alphas_cumprod[1:])
        self.posterior_variance = post_var.clamp_(1e-20, 1.0)

    def sample_xt(self, x0, t_idx):
        """
        x0: [B,T,C,H,W], t_idx: [B,T] (each entry in [0..n-1])
        returns x_t and noise eps used
        """
        b, T, C, H, W = x0.shape
        dev = self.alphas_cumprod.device
        t_idx = t_idx.to(dev)
        a = self.sqrt_alphas_cumprod[t_idx]            # [B,T]
        sig = self.sqrt_one_minus_alphas_cumprod[t_idx]# [B,T]
        a = a.view(b, T, 1, 1, 1)
        sig = sig.view(b, T, 1, 1, 1)
        eps = torch.randn_like(x0)
        return a * x0 + sig * eps, eps

    def snr_weight(self, t_idx, clip=5.0):
        # SNR = alpha_bar / (1-alpha_bar)
        dev = self.alphas_cumprod.device
        ab = self.alphas_cumprod[t_idx.to(dev)].clamp_min_(1e-6)
        snr = (ab / (1.0 - ab)).clamp_(0, clip)
        return snr

# ----------------------------
# Rolling local-time mapping (window W, cond n_cln)
# ----------------------------
def per_frame_timesteps(global_t, W=16, n_cln=1, n_steps=1000, mode="standard"):
    """
    global_t: [B] int64 in [0..n_steps-1]
    returns per-frame t_idx: [B, W] (int64)
    - standard: same t for all frames except cond frames forced to 0
    - rolling: offset increases with distance from cond boundary
    """
    b = global_t.shape[0]
    device = global_t.device
    t = global_t.view(b, 1).repeat(1, W)  # [B,W]

    # cond frames set clean (t=0)
    for k in range(n_cln):
        t[:, k] = 0

    if mode == "standard":
        return t.clamp_(0, n_steps - 1)

    # rolling: linear slope of offset across the non-cond frames
    # frames: 0...(n_cln-1) are clean; frames >= n_cln get per-frame offset
    remain = max(W - n_cln, 1)
    # total additional offset from first gen frame to farthest future frame
    total = n_steps - 1
    # slope so that the farthest frame stays noisy longer
    slope = total / remain
    idx = torch.arange(W, device=device).view(1, W).repeat(b, 1)  # [B,W]
    dist = (idx - (n_cln - 1)).clamp(min=0)  # 0 for cond range, 1.. for future frames
    offs = (dist * slope).long()
    t = (t + offs).clamp(0, n_steps - 1)
    # cond frames remain 0
    for k in range(n_cln):
        t[:, k] = 0
    return t

def init_noise_timesteps(W=16, n_cln=1, n_steps=1000):
    """
    Used to enter the rolling state at sampling boundary:
    per-frame starting t_idx: [W], increasing with distance from cond boundary.
    """
    remain = max(W - n_cln, 1)
    total = n_steps - 1
    slope = total / remain
    idx = torch.arange(W)
    dist = (idx - (n_cln - 1)).clamp(min=0)
    return (dist * slope).long().clamp(0, n_steps - 1)  # [W]

# ----------------------------
# Time embeddings (per-frame)
# ----------------------------
def sinusoidal_time_embed(t_idx, dim):
    # t_idx: [B,T] int -> [B,T,dim]
    device = t_idx.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0), math.log(10000.0), steps=half, device=device
        )
        * (-1.0 / (half - 1))
    )
    # normalize t to [0,1]
    t = t_idx.float() / (t_idx.max().float().clamp_min(1.0))
    args = t[..., None] * freqs  # [B,T,half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb  # [B,T,dim]

# ----------------------------
# Small 3D U-Net (time‑conditioned)
#   - keeps temporal length, downsamples spatially (1,2,2)
# ----------------------------
class ResBlock3D(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv3d(c_in, c_out, (1,3,3), padding=(0,1,1))
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv3d(c_out, c_out, (1,3,3), padding=(0,1,1))
        self.emb = nn.Linear(t_dim, c_out)
        self.skip = nn.Conv3d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t_emb):  # x: [B,C,T,H,W], t_emb: [B,T,dim]
        B, C, T, H, W = x.shape
        h = self.conv1(self.act(self.norm1(x)))
        # add per-frame time embedding as bias
        e = self.emb(t_emb).permute(0,2,1)[:, :, :, None, None]  # [B,C',T,1,1]
        h = h + e
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.Conv3d(c, c, (1,2,2), stride=(1,2,2))
    def forward(self, x): return self.pool(x)

class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose3d(c_in, c_out, (1,2,2), stride=(1,2,2))
    def forward(self, x): return self.up(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=3, base=64, t_dim=128):
        super().__init__()
        c1, c2, c3 = base, base*2, base*3
        self.t_dim = t_dim
        self.in_conv = nn.Conv3d(in_ch, c1, 3, padding=1)

        self.rb1a = ResBlock3D(c1, c1, t_dim); self.rb1b = ResBlock3D(c1, c1, t_dim)
        self.down1 = Down(c1)

        self.rb2a = ResBlock3D(c1, c2, t_dim); self.rb2b = ResBlock3D(c2, c2, t_dim)
        self.down2 = Down(c2)

        self.rb3a = ResBlock3D(c2, c3, t_dim); self.rb3b = ResBlock3D(c3, c3, t_dim)

        self.up2 = Up(c3, c2)
        self.rb_up2a = ResBlock3D(c2*2, c2, t_dim)
        self.rb_up2b = ResBlock3D(c2,    c2, t_dim)

        self.up1 = Up(c2, c1)
        self.rb_up1a = ResBlock3D(c1*2, c1, t_dim)
        self.rb_up1b = ResBlock3D(c1,   c1, t_dim)

        self.out = nn.Conv3d(c1, in_ch, 3, padding=1)

    def forward(self, x, t_emb):  # x: [B,C,T,H,W], t_emb: [B,T,dim]
        x = self.in_conv(x)
        x1 = self.rb1a(x, t_emb); x1 = self.rb1b(x1, t_emb)
        x2 = self.down1(x1)
        x2 = self.rb2a(x2, t_emb); x2 = self.rb2b(x2, t_emb)
        x3 = self.down2(x2)
        x3 = self.rb3a(x3, t_emb); x3 = self.rb3b(x3, t_emb)

        u2 = self.up2(x3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.rb_up2a(u2, t_emb); u2 = self.rb_up2b(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.rb_up1a(u1, t_emb); u1 = self.rb_up1b(u1, t_emb)
        return self.out(u1)

# ----------------------------
# EMA
# ----------------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    def load(self, model):
        model.load_state_dict(self.shadow, strict=False)

# ----------------------------
# Training
# ----------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Data iterator
    data_it = iter_bair(batch=args.batch, frames=args.frames, split="train")

    # Model & diffusion
    model = UNet3D(in_ch=3, base=args.base, t_dim=args.t_dim).to(device)
    sched = DiffusionSchedule(n_steps=args.diff_steps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=0.01)
    ema = EMA(model, decay=args.ema)
    ema.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scaler = GradScaler(enabled=args.fp16)

    if args.init_from:
        print(f"Initializing model from {args.init_from}")
        sd = torch.load(args.init_from, map_location=device)
        model.load_state_dict(sd, strict=False)
        # Also initialize the EMA shadow weights to match the loaded model
        ema.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    scaler = GradScaler(enabled=args.fp16)
    
    model.train()
    log_every = 100
    t0 = time.time()
    for step in range(1, args.steps+1):
        x0 = next(data_it).to(device)          # [B,T,C,H,W]
        B, T, C, H, W = x0.shape

        # pick a global t for the batch
        g_t = torch.randint(0, args.diff_steps, (B,), device=device)

        # per-frame t according to mode
        t_idx = per_frame_timesteps(
            g_t, W=T, n_cln=args.cond_frames, n_steps=args.diff_steps, mode=args.mode
        )  # [B,T]

        # sample x_t and eps
        x_t, eps = sched.sample_xt(x0, t_idx)  # [B,T,C,H,W], [B,T,C,H,W]
        # to model layout [B,C,T,H,W]
        x_t_cthw = x_t.permute(0,2,1,3,4).contiguous()

        # time embeddings per-frame
        t_emb = sinusoidal_time_embed(t_idx, args.t_dim)  # [B,T,dim]

        with autocast(enabled=args.fp16):
            pred_eps = model(x_t_cthw, t_emb)                  # [B,C,T,H,W]
            pred_eps = pred_eps.permute(0,2,1,3,4)            # [B,T,C,H,W]
            # ε-loss (SNR-weighted)
            w = sched.snr_weight(t_idx).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,T,1,1,1]
            loss = ((pred_eps - eps)**2 * (1.0 + w)).mean()

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        ema.update(model)

        if step % log_every == 0:
            dt = time.time() - t0
            print(f"[{step:6d}/{args.steps}] loss={loss.item():.4f}  {dt:.1f}s")
            t0 = time.time()
        if step % args.ckpt_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            # ✅ FIX: Create a temporary model with EMA weights and save it
            ema_model = UNet3D(in_ch=3, base=args.base, t_dim=args.t_dim)
            ema.load(ema_model) # Loads the shadow weights into ema_model
            torch.save(ema_model.state_dict(), os.path.join(args.save_dir, f"ema_step{step}.pt"))
            del ema_model # free memory

  # final ema save
    os.makedirs(args.save_dir, exist_ok=True)
    ema_model = UNet3D(in_ch=3, base=args.base, t_dim=args.t_dim)
    ema.load(ema_model)
    torch.save(ema_model.state_dict(), os.path.join(args.save_dir, f"ema_final.pt"))
    del ema_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("TFDS_DATA_DIR", "/data/tfds"))
    ap.add_argument("--save-dir", default="./checkpoints")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--cond-frames", type=int, default=1, dest="cond_frames")
    ap.add_argument("--mode", choices=["standard","rolling"], default="rolling")
    ap.add_argument("--diff-steps", type=int, default=1000, dest="diff_steps")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", type=int, default=64)          # channel base; raise to 96/128 if you have headroom
    ap.add_argument("--t-dim", type=int, default=128, dest="t_dim")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--ckpt-every", type=int, default=5000, dest="ckpt_every")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--init-from", default="", help="Path to an EMA .pt to initialize the model from")
    args = ap.parse_args()

    os.environ["TFDS_DATA_DIR"] = args.data_dir
    train(args)

if __name__ == "__main__":
    main()
