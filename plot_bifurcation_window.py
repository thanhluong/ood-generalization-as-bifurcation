"""
Publication-quality visualization of the Semantic Bifurcation Window.

Produces two figures (one per model type) showing MSE and Cosine Similarity
between joint-guided and decomposed-guided scores across the denoising
trajectory. The candidate bifurcation window is highlighted with red shading.

Usage:
    python plot_bifurcation_window.py \
        --diffusion-checkpoint diffusion_checkpoint.npz \
        --flow-checkpoint cfm_checkpoint.npz \
        --left 2 --right 8 --seeds 42 43 44 45 46
"""

import argparse
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from mass_sampling import load_model, NULL_TOKEN


# ---------------------------------------------------------------------------
# Interference measurement (returns t-values on x-axis)
# ---------------------------------------------------------------------------

def measure_interference_diffusion(model, z_init, left, right,
                                   guidance_scale, num_steps):
    """DDIM trajectory: t goes 1 -> 0. Returns (t_vals, mses, coss)."""
    B = z_init.shape[0]
    left_c = mx.full((B,), left, dtype=mx.int32)
    right_c = mx.full((B,), right, dtype=mx.int32)
    null_l = mx.full((B,), NULL_TOKEN, dtype=mx.int32)
    null_r = mx.full((B,), NULL_TOKEN, dtype=mx.int32)

    z = z_init
    t_vals, mses, coss = [], [], []
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_val = 0.999 - step * dt
        t_next = t_val - dt
        t_batch = mx.full((B,), t_val)

        out_uncond = model(z, t_batch, null_l, null_r)
        out_cond = model(z, t_batch, left_c, right_c)
        out_left = model(z, t_batch, left_c, null_r)
        out_right = model(z, t_batch, null_l, right_c)

        guided_joint = out_uncond + guidance_scale * (out_cond - out_uncond)
        guided_decomp = out_uncond + guidance_scale * (
            out_left + out_right - 2 * out_uncond)

        diff = guided_joint - guided_decomp
        mse = mx.mean(diff * diff)
        vj = guided_joint.reshape(B, -1)
        vs = guided_decomp.reshape(B, -1)
        dot = mx.sum(vj * vs, axis=-1)
        nj = mx.sqrt(mx.sum(vj * vj, axis=-1) + 1e-8)
        ns = mx.sqrt(mx.sum(vs * vs, axis=-1) + 1e-8)
        cos = mx.mean(dot / (nj * ns))
        mx.eval(mse, cos)
        t_vals.append(t_val)
        mses.append(mse.item())
        coss.append(cos.item())

        # DDIM step
        ab_t = 1.0 - t_val
        ab_next = 1.0 - t_next
        sqrt_ab = mx.sqrt(mx.maximum(mx.array(ab_t), mx.array(1e-8)))
        sqrt_1mab = mx.sqrt(mx.maximum(mx.array(1.0 - ab_t), mx.array(1e-8)))
        z0 = (z - sqrt_1mab * guided_joint) / sqrt_ab
        sqrt_ab_n = mx.sqrt(mx.maximum(mx.array(ab_next), mx.array(1e-8)))
        sqrt_1mab_n = mx.sqrt(
            mx.maximum(mx.array(1.0 - ab_next), mx.array(1e-8)))
        z = sqrt_ab_n * z0 + sqrt_1mab_n * guided_joint
        mx.eval(z)

    return np.array(t_vals), np.array(mses), np.array(coss)


def measure_interference_flow(model, z_init, left, right,
                              guidance_scale, num_steps):
    """Forward Euler trajectory: t goes 0 -> 1. Returns (t_vals, mses, coss)."""
    B = z_init.shape[0]
    left_c = mx.full((B,), left, dtype=mx.int32)
    right_c = mx.full((B,), right, dtype=mx.int32)
    null_l = mx.full((B,), NULL_TOKEN, dtype=mx.int32)
    null_r = mx.full((B,), NULL_TOKEN, dtype=mx.int32)

    z = z_init
    t_vals, mses, coss = [], [], []
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_val = step * dt
        t_batch = mx.full((B,), t_val)

        out_uncond = model(z, t_batch, null_l, null_r)
        out_cond = model(z, t_batch, left_c, right_c)
        out_left = model(z, t_batch, left_c, null_r)
        out_right = model(z, t_batch, null_l, right_c)

        guided_joint = out_uncond + guidance_scale * (out_cond - out_uncond)
        guided_decomp = out_uncond + guidance_scale * (
            out_left + out_right - 2 * out_uncond)

        diff = guided_joint - guided_decomp
        mse = mx.mean(diff * diff)
        vj = guided_joint.reshape(B, -1)
        vs = guided_decomp.reshape(B, -1)
        dot = mx.sum(vj * vs, axis=-1)
        nj = mx.sqrt(mx.sum(vj * vj, axis=-1) + 1e-8)
        ns = mx.sqrt(mx.sum(vs * vs, axis=-1) + 1e-8)
        cos = mx.mean(dot / (nj * ns))
        mx.eval(mse, cos)
        t_vals.append(t_val)
        mses.append(mse.item())
        coss.append(cos.item())

        z = z + dt * guided_joint
        mx.eval(z)

    return np.array(t_vals), np.array(mses), np.array(coss)


# ---------------------------------------------------------------------------
# Multi-seed averaging
# ---------------------------------------------------------------------------

def collect_seeds(model, model_type, left, right, guidance_scale,
                  num_steps, seeds):
    measure_fn = (measure_interference_diffusion if model_type == "diffusion"
                  else measure_interference_flow)
    all_mses, all_coss = [], []
    t_vals = None
    for seed in seeds:
        mx.random.seed(seed)
        z_init = mx.random.normal((1, 7, 14, 2))
        t, mses, coss = measure_fn(model, z_init, left, right,
                                   guidance_scale, num_steps)
        all_mses.append(mses)
        all_coss.append(coss)
        if t_vals is None:
            t_vals = t
    return t_vals, np.array(all_mses), np.array(all_coss)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(t_vals, all_mses, all_coss, model_type, left, right,
                guidance_scale, bif_lo, bif_hi, title_pad, output_path):
    """Create a publication-quality dual-panel figure."""

    # Compute mean and std across seeds
    mse_mean = all_mses.mean(axis=0)
    mse_std = all_mses.std(axis=0)
    cos_mean = all_coss.mean(axis=0)
    cos_std = all_coss.std(axis=0)

    # For diffusion, t goes high->low; reverse so x-axis is ascending
    if model_type == "diffusion":
        t_vals = t_vals[::-1].copy()
        mse_mean = mse_mean[::-1].copy()
        mse_std = mse_std[::-1].copy()
        cos_mean = cos_mean[::-1].copy()
        cos_std = cos_std[::-1].copy()

    model_label = ("CFDG Diffusion (DDIM)" if model_type == "diffusion"
                   else "Conditional Flow Matching (CFM)")

    # ---- Style ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "lines.linewidth": 2.0,
    })

    fig, (ax_mse, ax_cos) = plt.subplots(
        2, 1, figsize=(7, 5.5), sharex=True,
        gridspec_kw={"hspace": 0.08, "height_ratios": [1, 1]})

    # ---- Bifurcation shading (integral-style fill) ----
    for ax in (ax_mse, ax_cos):
        ax.axvspan(bif_lo, bif_hi, color="#e74c3c", alpha=0.12,
                   zorder=0, label="_nolegend_")

    # ---- MSE panel ----
    ax_mse.plot(t_vals, mse_mean, color="#2c3e50", zorder=3)
    ax_mse.fill_between(t_vals, mse_mean - mse_std, mse_mean + mse_std,
                        color="#2c3e50", alpha=0.18, zorder=2)
    # Fill MSE inside bifurcation window with red
    bif_mask = (t_vals >= bif_lo) & (t_vals <= bif_hi)
    ax_mse.fill_between(t_vals, 0, mse_mean,
                        where=bif_mask, color="#e74c3c", alpha=0.30,
                        zorder=2, label="Bifurcation window")
    ax_mse.set_ylabel("MSE  (joint vs. decomposed)")
    ax_mse.set_xlim(t_vals.min(), t_vals.max())
    ax_mse.set_ylim(bottom=0)
    ax_mse.grid(True, alpha=0.25)
    ax_mse.legend(loc="upper right", framealpha=0.85)
    ax_mse.set_title(
        f"{model_label}  —  Superposition Interference  "
        f"(pair {left},{right},  $w$={guidance_scale})",
        pad=title_pad)

    # ---- Cosine Similarity panel ----
    ax_cos.plot(t_vals, cos_mean, color="#2980b9", zorder=3)
    ax_cos.fill_between(t_vals, cos_mean - cos_std, cos_mean + cos_std,
                        color="#2980b9", alpha=0.18, zorder=2)
    # Fill cos inside bifurcation window with red (down from 1.0)
    ax_cos.fill_between(t_vals, cos_mean, 1.0,
                        where=bif_mask, color="#e74c3c", alpha=0.30,
                        zorder=2, label="Bifurcation window")
    ax_cos.set_ylabel("Cosine Similarity")
    ax_cos.set_xlabel("Denoising time  $t$")
    ax_cos.set_ylim(0.8, 1.05)
    ax_cos.grid(True, alpha=0.25)
    ax_cos.legend(loc="lower right", framealpha=0.85)

    # ---- Dashed vertical lines at window edges ----
    for ax in (ax_mse, ax_cos):
        for tb in (bif_lo, bif_hi):
            ax.axvline(tb, color="#e74c3c", ls="--", lw=1.0, alpha=0.6,
                       zorder=4)

    # Minor ticks
    ax_cos.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    for ax in (ax_mse, ax_cos):
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(which="both", direction="in")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot semantic bifurcation window for paper figures")
    parser.add_argument("--diffusion-checkpoint", type=str,
                        default="diffusion_checkpoint.npz")
    parser.add_argument("--flow-checkpoint", type=str,
                        default="cfm_checkpoint.npz")
    parser.add_argument("--left", type=int, default=2,
                        help="Left digit label (default: 2)")
    parser.add_argument("--right", type=int, default=8,
                        help="Right digit label (default: 8)")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=100,
                        help="Sampling steps (more = smoother curves)")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46],
                        help="Random seeds to average over")
    parser.add_argument("--bif-lo", type=float, default=0.1,
                        help="Bifurcation window lower bound (t)")
    parser.add_argument("--bif-hi", type=float, default=0.3,
                        help="Bifurcation window upper bound (t)")
    parser.add_argument("--title-pad", type=float, default=30.0,
                        help="Padding between title and plot (y-axis distance)")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Directory for output PNGs")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for model_type, ckpt in [("diffusion", args.diffusion_checkpoint),
                              ("flow", args.flow_checkpoint)]:
        print(f"\n{'='*60}")
        print(f"  {model_type.upper()} — loading {ckpt}")
        print(f"{'='*60}")
        model = load_model(ckpt)

        print(f"  Collecting interference over {len(args.seeds)} seeds "
              f"({args.steps} steps each) ...")
        t_vals, all_mses, all_coss = collect_seeds(
            model, model_type, args.left, args.right,
            args.guidance_scale, args.steps, args.seeds)

        out_path = str(
            Path(args.output_dir) /
            f"bifurcation_{model_type}_{args.left}{args.right}.png")
        make_figure(t_vals, all_mses, all_coss, model_type,
                    args.left, args.right, args.guidance_scale,
                    args.bif_lo, args.bif_hi, args.title_pad, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
