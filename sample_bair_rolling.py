import os, argparse
import torch, torch.nn.functional as F
import numpy as np
import imageio.v2 as imageio

def make_t_init_with_offset(frames, n_cln, n_steps, t_commit_ratio=0.35):
    """
    Leftmost n_cln frames are t=0 (conditioning).
    The commit target (index = n_cln) starts at t = floor((n_steps-1) * t_commit_ratio).
    t then increases linearly toward the right so that the last slot is close to n_steps-1.
    """
    t_commit = int((n_steps - 1) * t_commit_ratio)
    remain = max(frames - n_cln, 1)
    if remain <= 1:
        slope = 0.0
    else:
        slope = ((n_steps - 1) - t_commit) / float(remain - 1)

    idx = torch.arange(frames)
    dist = (idx - n_cln).clamp(min=0)
    t_init = (t_commit + dist * slope).long().clamp(0, n_steps - 1)
    if n_cln > 0:
        t_init[:n_cln] = 0
    return t_init


# === import pieces from your training script (must be in same folder) ===
from train_bair_rolling import (
    UNet3D, DiffusionSchedule, iter_bair,
    sinusoidal_time_embed, init_noise_timesteps
)

@torch.no_grad()
@torch.no_grad()
def roll_sample_sliding(
    model, sched, frames=16, cond_frames=1, diff_steps=1000,
    batch=4, device="cuda", split="test", refine_passes=1
):
    """
    Rolling sampler (carry-over):
      - Denoise all frames in the window simultaneously.
      - Stop when the first generative frame (index = cond_frames) reaches t=0, then commit it.
      - Carry over (shift) the remaining partially denoised future frames left by 1,
        and reinitialize only the last slot with fresh noise.
    Returns:
      out: [B,T,C,H,W] generated sequence (first cond_frames come from GT)
      x_gt: [B,T,C,H,W] ground-truth clip used for conditioning/visualization
    """
    model.eval()
    it = iter_bair(batch=batch, frames=frames, split=split)
    x_gt = next(it).to(device)  # [-1,1], [B,T,C,H,W]
    B, T, C, H, W = x_gt.shape

    # Output buffer
    out = torch.empty_like(x_gt)
    out[:, :cond_frames] = x_gt[:, :cond_frames]

    # Initial window: condition from GT, future initialized with noise
    # (the commit frame starts at a slightly higher initial t)
    x = torch.zeros(B, frames, C, H, W, device=device)
    x[:, :cond_frames] = x_gt[:, :cond_frames]

    t_init = make_t_init_with_offset(frames, cond_frames, diff_steps, t_commit_ratio=0.35).to(device)
    t_idx = t_init.view(1, frames).repeat(B, 1).clone()  # [B,frames]

    sig = sched.sqrt_one_minus_alphas_cumprod[t_idx].view(B, frames, 1, 1, 1)
    x[:, cond_frames:] = torch.randn_like(x[:, cond_frames:]) * sig[:, cond_frames:]

    for t_out in range(cond_frames, T):
        # Reverse diffusion with optional refine passes
        for rp in range(refine_passes):
            # Run reverse steps until the first generative frame reaches t=0
            while int(t_idx[:, cond_frames].max().item()) > 0:
                x_cthw = x.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]
                t_emb = sinusoidal_time_embed(t_idx, model.t_dim)  # per-frame time embeddings
                eps_pred = model(x_cthw, t_emb).permute(0, 2, 1, 3, 4)  # [B,frames,C,H,W]

                # Schedule tensors
                ab_t   = sched.alphas_cumprod[t_idx]                          # \bar{α}_t
                ab_t1  = sched.alphas_cumprod[(t_idx - 1).clamp_min(0)]       # \bar{α}_{t-1}
                sqrt_ab_t = sched.sqrt_alphas_cumprod[t_idx]
                sqrt_1mab = sched.sqrt_one_minus_alphas_cumprod[t_idx]
                alphas_t  = sched.alphas[t_idx]                                # α_t
                betas_t   = (1.0 - alphas_t)                                   # β_t
                var_t     = sched.posterior_variance[t_idx]                    # \tilde{β}_t

                # Estimate x0 and compute posterior mean μ_t
                x0_pred = (x - sqrt_1mab.view(B,frames,1,1,1) * eps_pred) / \
                          (sqrt_ab_t.view(B,frames,1,1,1) + 1e-12)
                x0_pred = x0_pred.clamp(-1, 1)

                coef_x0 = (torch.sqrt(ab_t1) * betas_t) / (1.0 - ab_t + 1e-12)
                coef_xt = (torch.sqrt(alphas_t) * (1.0 - ab_t1)) / (1.0 - ab_t + 1e-12)
                mu_t = coef_x0.view(B,frames,1,1,1) * x0_pred + coef_xt.view(B,frames,1,1,1) * x

                noise = torch.randn_like(x)
                x = torch.where(
                    (t_idx > 0).view(B,frames,1,1,1),
                    mu_t + torch.sqrt(var_t.clamp_min(1e-20)).view(B,frames,1,1,1) * noise,
                    x
                )

                # Decrease t by 1 (pin conditioning frames at t=0 and to GT context)
                t_idx = torch.clamp(t_idx - (t_idx > 0).long(), min=0)
                t_idx[:, :cond_frames] = 0
                x[:, :cond_frames] = out[:, t_out - cond_frames : t_out] if t_out > cond_frames \
                                     else x_gt[:, :cond_frames]

            # If refine_passes > 1, add a light re-noise to tighten the committed frame (optional)
            if rp + 1 < refine_passes:
                sig_ref = sched.sqrt_one_minus_alphas_cumprod[t_idx].view(B, frames, 1, 1, 1)
                x[:, cond_frames:] = x[:, cond_frames:] + \
                                     torch.randn_like(x[:, cond_frames:]) * 0.1 * sig_ref[:, cond_frames:]
                t_idx[:, :cond_frames] = 0
                x[:, :cond_frames] = out[:, t_out - cond_frames : t_out] if t_out > cond_frames \
                                     else x_gt[:, :cond_frames]

        # Commit the first generated frame
        out[:, t_out] = x[:, cond_frames]

        # ====== Carry-over (stateful slide) ======
        if t_out + 1 < T:
            x_next = torch.zeros_like(x)
            t_next = torch.zeros_like(t_idx)

            # New context (cond_frames) from the latest output
            x_next[:, :cond_frames] = out[:, t_out - cond_frames + 1 : t_out + 1]
            t_next[:, :cond_frames] = 0

            # Shift the future part left by 1 to reuse state and t
            if frames - cond_frames - 1 > 0:
                x_next[:, cond_frames : frames-1] = x[:, cond_frames+1 : frames]
                t_next[:, cond_frames : frames-1] = t_idx[:, cond_frames+1 : frames]

            # Initialize the tail slot as the far future (note shape [B,1,1,1])
            last_t = int(t_init[-1].item())
            t_next[:, -1] = last_t
            sig_last = sched.sqrt_one_minus_alphas_cumprod[t_next[:, -1]].view(B, 1, 1, 1)
            x_next[:, -1] = torch.randn_like(x_next[:, -1]) * sig_last

            x, t_idx = x_next, t_next

    return out, x_gt


def to_uint8(x):
    # x: [-1,1] tensor [B,T,C,H,W] → uint8 [B,T,H,W,C]
    x = torch.clamp((x + 1.0) * 127.5, 0, 255).byte()
    return x.permute(0,1,3,4,2).cpu().numpy()

def save_gifs(x_gen, x_gt, out_dir, prefix="sample"):
    os.makedirs(out_dir, exist_ok=True)
    B, T, H, W, C = x_gen.shape
    for b in range(B):
        frames = []
        for t in range(T):
            row = np.concatenate([x_gt[b,t], x_gen[b,t]], axis=1)  # side-by-side
            frames.append(row)
        imageio.mimsave(os.path.join(out_dir, f"{prefix}_{b}.gif"), frames, fps=8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--cond-frames", type=int, default=1, dest="cond_frames")
    ap.add_argument("--diff-steps", type=int, default=1000, dest="diff_steps")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--split", default="test")
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--t-dim", type=int, default=128, dest="t_dim")
    ap.add_argument("--out", default="./samples")
    ap.add_argument("--refine-passes", type=int, default=1,
                    help="Extra short reverse passes per committed frame (1–3 helps sharpen)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model & EMA weights
    model = UNet3D(in_ch=3, base=args.base, t_dim=args.t_dim).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded {args.ckpt}. Missing={len(missing)} Unexpected={len(unexpected)}")
    model.eval()

    # Diffusion schedule
    sched = DiffusionSchedule(n_steps=args.diff_steps, device=device)

    # Generate
    x_gen, x_gt = roll_sample_sliding(
        model, sched,
        frames=args.frames, cond_frames=args.cond_frames,
        diff_steps=args.diff_steps, batch=args.batch,
        device=device, split=args.split, refine_passes=args.refine_passes
    )
    xg = to_uint8(x_gen)
    x0 = to_uint8(x_gt)
    save_gifs(xg, x0, args.out, prefix="bair_roll")

if __name__ == "__main__":
    main()
