"""Visualize joint vs decomposed sampling trajectories for flow matching.

Produces a grid:
  Rows:    t = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
  Columns: v_joint | v_decomposed | v(L,∅) | v(∅,R)

Rows inside the bifurcation window (t=0.2, t=0.4) are tinted red.

Usage:
  python visualize_flow_strategies.py \
      --checkpoint cfm_checkpoint.npz \
      --vae-checkpoint vae_checkpoint.npz \
      --left 2 --right 8 --seed 42 \
      --output figures/flow_strategies_28.png
"""
import argparse
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt

from mass_sampling import load_model, load_vae, latent_to_image, NULL_TOKEN


def _make_labels(B, val):
    return mx.full((B,), val, dtype=mx.int32)


def _run_flow_trajectory(model, z_init, score_fn, num_steps, capture_ts):
    """Run forward Euler and capture z at specified t values.

    Args:
        score_fn: callable(model, z, t_batch) -> guided velocity
        capture_ts: set of float t values to capture (e.g. {0.0, 0.2, ...})

    Returns:
        dict mapping t_float -> z tensor
    """
    z = z_init
    dt = 1.0 / num_steps
    snaps = {}

    # Capture initial state at t=0
    if 0.0 in capture_ts:
        mx.eval(z)
        snaps[0.0] = z

    for step in range(num_steps):
        t_val = step * dt
        B = z.shape[0]
        t_batch = mx.full((B,), t_val)

        v = score_fn(model, z, t_batch)
        z = z + dt * v
        mx.eval(z)

        # t after this step
        t_after = round((step + 1) * dt, 4)
        if t_after in capture_ts:
            snaps[t_after] = z

    return snaps


def main():
    parser = argparse.ArgumentParser(
        description="Visualize joint vs decomposed flow sampling trajectories")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to CFM model checkpoint")
    parser.add_argument("--vae-checkpoint", type=str,
                        default="vae_checkpoint.npz")
    parser.add_argument("--left", type=int, required=True,
                        help="Left digit label")
    parser.add_argument("--right", type=int, required=True,
                        help="Right digit label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--output", type=str, required=True,
                        help="Output image path (.png)")
    parser.add_argument("--bif-start", type=float, default=0.2,
                        help="Bifurcation window start t (for red tint)")
    parser.add_argument("--bif-end", type=float, default=0.4,
                        help="Bifurcation window end t (for red tint)")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    vae = load_vae(args.vae_checkpoint)

    mx.random.seed(args.seed)
    z_init = mx.random.normal((1, 7, 14, 2))

    left, right, w = args.left, args.right, args.guidance_scale
    null = NULL_TOKEN

    null_l = _make_labels(1, null)
    null_r = _make_labels(1, null)
    left_c = _make_labels(1, left)
    right_c = _make_labels(1, right)

    # --- Score functions (CFG-guided) ---
    def score_joint(m, z, t):
        out_u = m(z, t, null_l, null_r)
        out_c = m(z, t, left_c, right_c)
        return out_u + w * (out_c - out_u)

    def score_decomposed(m, z, t):
        out_u = m(z, t, null_l, null_r)
        out_l = m(z, t, left_c, null_r)
        out_r = m(z, t, null_l, right_c)
        return out_u + w * (out_l + out_r - 2 * out_u)

    def score_left_only(m, z, t):
        out_u = m(z, t, null_l, null_r)
        out_l = m(z, t, left_c, null_r)
        return out_u + w * (out_l - out_u)

    def score_right_only(m, z, t):
        out_u = m(z, t, null_l, null_r)
        out_r = m(z, t, null_l, right_c)
        return out_u + w * (out_r - out_u)

    capture_ts = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
    t_list = sorted(capture_ts)

    strategies = [
        (f"v_joint({left},{right})", score_joint),
        (f"v_decomp({left},{right})", score_decomposed),
        (f"v({left},∅)", score_left_only),
        (f"v(∅,{right})", score_right_only),
    ]

    # Run all 4 trajectories from the same z_init
    all_snaps = {}
    for name, sfn in strategies:
        print(f"  Running: {name}")
        all_snaps[name] = _run_flow_trajectory(
            model, z_init, sfn, args.steps, capture_ts)

    # --- Build figure ---
    n_rows = len(t_list)
    n_cols = len(strategies)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 2.2 * n_rows))

    bif_ts = {round(t, 4) for t in t_list
              if args.bif_start <= t <= args.bif_end and t > 0.0}

    for col, (name, _) in enumerate(strategies):
        for row, t in enumerate(t_list):
            z_snap = all_snaps[name].get(t)
            if z_snap is None:
                axes[row, col].axis("off")
                continue

            img_gray = latent_to_image(vae, z_snap)[0]  # (28, 56) uint8

            in_bif = round(t, 4) in bif_ts

            if in_bif:
                # Convert to RGB and apply red tint
                img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
                red_overlay = np.zeros_like(img_rgb)
                red_overlay[..., 0] = 255.0
                alpha = 0.3
                img_rgb = (1 - alpha) * img_rgb + alpha * red_overlay
                img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
                axes[row, col].imshow(img_rgb)
            else:
                axes[row, col].imshow(img_gray, cmap="gray", vmin=0, vmax=255)

            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(name, fontsize=9, fontweight="bold")
            if col == 0:
                axes[row, col].set_ylabel(f"t = {t:.1f}", fontsize=10,
                                           rotation=0, labelpad=40, va="center")

    fig.suptitle(
        f"Flow Matching Sampling Strategies — pair ({left}, {right}), "
        f"w={w}, seed={args.seed}\n"
        f"Red tint = bifurcation window [{args.bif_start}, {args.bif_end}]",
        fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
