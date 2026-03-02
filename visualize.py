import argparse
import pickle
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mass_sampling import (load_model, load_vae, latent_to_image,
                           cfg_combine, NULL_TOKEN, BLANK_LABEL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_labels(B, left, right):
    return (mx.full((B,), left, dtype=mx.int32),
            mx.full((B,), right, dtype=mx.int32))


def _milestone_indices(num_steps, n_milestones=5):
    """Return step indices for evenly-spaced milestones (inclusive of last)."""
    return np.linspace(0, num_steps - 1, n_milestones).astype(int).tolist()


def _score_at(model, z, t_batch, left, right):
    """Raw model output for a single conditioning pair."""
    B = z.shape[0]
    l, r = _make_labels(B, left, right)
    return model(z, t_batch, l, r)


def _score_norm_heatmap(score_np):
    """L2 norm across channel dim -> (H, W) heatmap. Input: (1, H, W, C)."""
    return np.sqrt((score_np[0] ** 2).sum(axis=-1))  # (H, W)


# ---------------------------------------------------------------------------
# Sampling loops with intermediate capture
# ---------------------------------------------------------------------------

def _run_diffusion_loop(model, z_init, cond_fn, guidance_scale, num_steps,
                        milestones):
    """DDIM loop returning (final_z, {step: z_after}, {step: guided_score})."""
    B = z_init.shape[0]
    null_l, null_r = _make_labels(B, NULL_TOKEN, NULL_TOKEN)
    z = z_init
    z_snaps, score_snaps = {}, {}
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t_next = t_val - dt
        t_batch = mx.full((B,), t_val)

        out_cond = cond_fn(model, z, t_batch)
        out_uncond = model(z, t_batch, null_l, null_r)
        guided = cfg_combine(out_cond, out_uncond, guidance_scale)

        if step in milestones:
            mx.eval(guided)
            score_snaps[step] = np.array(guided)

        ab_t = 1.0 - t_val
        ab_next = 1.0 - t_next
        sqrt_ab = mx.sqrt(mx.maximum(mx.array(ab_t), mx.array(1e-8)))
        sqrt_1mab = mx.sqrt(mx.maximum(mx.array(1.0 - ab_t), mx.array(1e-8)))
        z0 = (z - sqrt_1mab * guided) / sqrt_ab
        sqrt_ab_n = mx.sqrt(mx.maximum(mx.array(ab_next), mx.array(1e-8)))
        sqrt_1mab_n = mx.sqrt(mx.maximum(mx.array(1.0 - ab_next), mx.array(1e-8)))
        z = sqrt_ab_n * z0 + sqrt_1mab_n * guided
        mx.eval(z)

        if step in milestones:
            z_snaps[step] = z

    return z, z_snaps, score_snaps


def _run_flow_loop(model, z_init, cond_fn, guidance_scale, num_steps,
                   milestones):
    """Forward Euler loop returning (final_z, {step: z_after}, {step: score})."""
    B = z_init.shape[0]
    null_l, null_r = _make_labels(B, NULL_TOKEN, NULL_TOKEN)
    z = z_init
    z_snaps, score_snaps = {}, {}
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_val = step * dt
        t_batch = mx.full((B,), t_val)

        out_cond = cond_fn(model, z, t_batch)
        out_uncond = model(z, t_batch, null_l, null_r)
        guided = cfg_combine(out_cond, out_uncond, guidance_scale)

        if step in milestones:
            mx.eval(guided)
            score_snaps[step] = np.array(guided)

        z = z + dt * guided
        mx.eval(z)

        if step in milestones:
            z_snaps[step] = z

    return z, z_snaps, score_snaps


def _run_loop(model, model_type, z_init, cond_fn, guidance_scale,
              num_steps, milestones):
    if model_type == "diffusion":
        return _run_diffusion_loop(model, z_init, cond_fn, guidance_scale,
                                   num_steps, milestones)
    return _run_flow_loop(model, z_init, cond_fn, guidance_scale,
                          num_steps, milestones)


# ---------------------------------------------------------------------------
# Interference measurement along joint trajectory
# ---------------------------------------------------------------------------

def _measure_interference_diffusion(model, z_init, left, right,
                                    guidance_scale, num_steps):
    B = z_init.shape[0]
    left_c, right_c = _make_labels(B, left, right)
    null_l, null_r = _make_labels(B, NULL_TOKEN, NULL_TOKEN)
    blank = mx.full((B,), BLANK_LABEL, dtype=mx.int32)

    z = z_init
    mses, coss, steps_list = [], [], []
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t_next = t_val - dt
        t_batch = mx.full((B,), t_val)

        # Joint score (conditional only, before CFG)
        v_joint = model(z, t_batch, left_c, right_c)
        # Sum score
        v_lb = model(z, t_batch, left_c, blank)
        v_br = model(z, t_batch, blank, right_c)
        v_bb = model(z, t_batch, blank, blank)
        v_sum = v_lb + v_br - v_bb

        # MSE
        diff = v_joint - v_sum
        mse = mx.mean(diff * diff)
        # Cosine
        vj = v_joint.reshape(B, -1)
        vs = v_sum.reshape(B, -1)
        dot = mx.sum(vj * vs, axis=-1)
        nj = mx.sqrt(mx.sum(vj * vj, axis=-1) + 1e-8)
        ns = mx.sqrt(mx.sum(vs * vs, axis=-1) + 1e-8)
        cos = mx.mean(dot / (nj * ns))
        mx.eval(mse, cos)
        mses.append(mse.item())
        coss.append(cos.item())
        steps_list.append(step)

        # Advance z using joint + CFG (standard trajectory)
        v_uncond = model(z, t_batch, null_l, null_r)
        guided = cfg_combine(v_joint, v_uncond, guidance_scale)

        ab_t = 1.0 - t_val
        ab_next = 1.0 - t_next
        sqrt_ab = mx.sqrt(mx.maximum(mx.array(ab_t), mx.array(1e-8)))
        sqrt_1mab = mx.sqrt(mx.maximum(mx.array(1.0 - ab_t), mx.array(1e-8)))
        z0 = (z - sqrt_1mab * guided) / sqrt_ab
        sqrt_ab_n = mx.sqrt(mx.maximum(mx.array(ab_next), mx.array(1e-8)))
        sqrt_1mab_n = mx.sqrt(mx.maximum(mx.array(1.0 - ab_next), mx.array(1e-8)))
        z = sqrt_ab_n * z0 + sqrt_1mab_n * guided
        mx.eval(z)

    return steps_list, mses, coss


def _measure_interference_flow(model, z_init, left, right,
                                guidance_scale, num_steps):
    B = z_init.shape[0]
    left_c, right_c = _make_labels(B, left, right)
    null_l, null_r = _make_labels(B, NULL_TOKEN, NULL_TOKEN)
    blank = mx.full((B,), BLANK_LABEL, dtype=mx.int32)

    z = z_init
    mses, coss, steps_list = [], [], []
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_val = step * dt
        t_batch = mx.full((B,), t_val)

        v_joint = model(z, t_batch, left_c, right_c)
        v_lb = model(z, t_batch, left_c, blank)
        v_br = model(z, t_batch, blank, right_c)
        v_bb = model(z, t_batch, blank, blank)
        v_sum = v_lb + v_br - v_bb

        diff = v_joint - v_sum
        mse = mx.mean(diff * diff)
        vj = v_joint.reshape(B, -1)
        vs = v_sum.reshape(B, -1)
        dot = mx.sum(vj * vs, axis=-1)
        nj = mx.sqrt(mx.sum(vj * vj, axis=-1) + 1e-8)
        ns = mx.sqrt(mx.sum(vs * vs, axis=-1) + 1e-8)
        cos = mx.mean(dot / (nj * ns))
        mx.eval(mse, cos)
        mses.append(mse.item())
        coss.append(cos.item())
        steps_list.append(step)

        v_uncond = model(z, t_batch, null_l, null_r)
        guided = cfg_combine(v_joint, v_uncond, guidance_scale)
        z = z + dt * guided
        mx.eval(z)

    return steps_list, mses, coss


# ---------------------------------------------------------------------------
# Subcommand: recon
# ---------------------------------------------------------------------------

def cmd_recon(args):
    """Visualize VAE reconstruction on random samples."""
    vae = load_vae(args.vae_checkpoint)

    with open(args.data, "rb") as f:
        data = pickle.load(f)
    images = data["images"].astype(np.float32).reshape(-1, 28, 56, 1)
    labels = data["labels"]
    images_norm = images / 127.5 - 1.0

    np.random.seed(args.seed)
    idx = np.random.choice(len(images_norm), args.num_samples, replace=False)

    fig, axes = plt.subplots(2, args.num_samples,
                             figsize=(2.5 * args.num_samples, 5))
    if args.num_samples == 1:
        axes = axes.reshape(2, 1)

    for i, j in enumerate(idx):
        x = mx.array(images_norm[j:j+1])
        mu, _ = vae.encoder(x)
        x_hat = vae.decode(mu)
        mx.eval(x_hat)
        x_hat_np = np.array(x_hat)
        x_hat_np = ((x_hat_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        orig = images[j].reshape(28, 56).clip(0, 255).astype(np.uint8)

        lbl = labels[j]
        axes[0, i].imshow(orig, cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(f"Original [{lbl[0]},{lbl[1]}]", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(x_hat_np[0, :, :, 0], cmap="gray", vmin=0, vmax=255)
        axes[1, i].set_title("Reconstruction", fontsize=9)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved reconstruction to {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: interference
# ---------------------------------------------------------------------------

def cmd_interference(args):
    """Plot MSE and cosine similarity between joint and sum scores."""
    model = load_model(args.checkpoint)

    mx.random.seed(args.seed)
    z_init = mx.random.normal((1, 7, 14, 2))

    print(f"Measuring interference for ({args.left}, {args.right}) "
          f"over {args.steps} steps...")

    if args.model_type == "diffusion":
        steps, mses, coss = _measure_interference_diffusion(
            model, z_init, args.left, args.right,
            args.guidance_scale, args.steps)
    else:
        steps, mses, coss = _measure_interference_flow(
            model, z_init, args.left, args.right,
            args.guidance_scale, args.steps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(steps, mses, color="tab:red", linewidth=1.5)
    ax1.set_ylabel("MSE", fontsize=11)
    ax1.set_title(f"Interference: e({args.left},{args.right}) vs "
                  f"e({args.left},0)+e(0,{args.right})-e(0,0)  "
                  f"[{args.model_type}, w={args.guidance_scale}]", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, coss, color="tab:blue", linewidth=1.5)
    ax2.set_ylabel("Cosine Similarity", fontsize=11)
    ax2.set_xlabel("Sampling step", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved interference plot to {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: progress
# ---------------------------------------------------------------------------

def cmd_progress(args):
    """Visualize generation progress (decoded images) for 5 score variants."""
    model = load_model(args.checkpoint)
    vae = load_vae(args.vae_checkpoint)

    mx.random.seed(args.seed)
    z_init = mx.random.normal((1, 7, 14, 2))

    milestones = _milestone_indices(args.steps)
    B = 1
    left, right = args.left, args.right
    blank_l = BLANK_LABEL

    # Define 5 conditioning functions
    def cond_joint(m, z, t):
        return _score_at(m, z, t, left, right)

    def cond_sum(m, z, t):
        return (_score_at(m, z, t, left, blank_l) +
                _score_at(m, z, t, blank_l, right) -
                _score_at(m, z, t, blank_l, blank_l))

    def cond_left_blank(m, z, t):
        return _score_at(m, z, t, left, blank_l)

    def cond_blank_right(m, z, t):
        return _score_at(m, z, t, blank_l, right)

    def cond_blank_blank(m, z, t):
        return _score_at(m, z, t, blank_l, blank_l)

    variants = [
        (f"e({left},{right})", cond_joint),
        (f"e({left},0)+e(0,{right})-e(0,0)", cond_sum),
        (f"e({left},0)", cond_left_blank),
        (f"e(0,{right})", cond_blank_right),
        ("e(0,0)", cond_blank_blank),
    ]

    n_rows = len(milestones)
    n_cols = len(variants)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 2 * n_rows))

    for col, (title, cond_fn) in enumerate(variants):
        print(f"  Running trajectory: {title}")
        _, z_snaps, _ = _run_loop(model, args.model_type,
                                  z_init, cond_fn,
                                  args.guidance_scale, args.steps,
                                  milestones)
        for row, ms in enumerate(milestones):
            img = latent_to_image(vae, z_snaps[ms])  # (1, 28, 56)
            axes[row, col].imshow(img[0], cmap="gray", vmin=0, vmax=255)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=8)
            if col == 0:
                axes[row, col].set_ylabel(f"step {ms}", fontsize=9)

    fig.suptitle(f"Generation progress ({args.model_type}, "
                 f"pair={args.left}{args.right}, w={args.guidance_scale})",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved progress visualization to {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: heatmap
# ---------------------------------------------------------------------------

def cmd_heatmap(args):
    """Visualize score magnitude heatmaps at milestone steps."""
    model = load_model(args.checkpoint)

    mx.random.seed(args.seed)
    z_init = mx.random.normal((1, 7, 14, 2))

    milestones = _milestone_indices(args.steps)
    left, right = args.left, args.right
    blank_l = BLANK_LABEL

    def cond_joint(m, z, t):
        return _score_at(m, z, t, left, right)

    def cond_sum(m, z, t):
        return (_score_at(m, z, t, left, blank_l) +
                _score_at(m, z, t, blank_l, right) -
                _score_at(m, z, t, blank_l, blank_l))

    def cond_left_blank(m, z, t):
        return _score_at(m, z, t, left, blank_l)

    def cond_blank_right(m, z, t):
        return _score_at(m, z, t, blank_l, right)

    def cond_blank_blank(m, z, t):
        return _score_at(m, z, t, blank_l, blank_l)

    variants = [
        (f"e({left},{right})", cond_joint),
        (f"e({left},0)+e(0,{right})-e(0,0)", cond_sum),
        (f"e({left},0)", cond_left_blank),
        (f"e(0,{right})", cond_blank_right),
        ("e(0,0)", cond_blank_blank),
    ]

    n_rows = len(milestones)
    n_cols = len(variants)

    # Collect all heatmaps first to determine global vmin/vmax
    all_hmaps = {}
    for col, (title, cond_fn) in enumerate(variants):
        print(f"  Running trajectory: {title}")
        _, _, score_snaps = _run_loop(model, args.model_type,
                                      z_init, cond_fn,
                                      args.guidance_scale, args.steps,
                                      milestones)
        for ms in milestones:
            all_hmaps[(col, ms)] = _score_norm_heatmap(score_snaps[ms])

    vmin = min(h.min() for h in all_hmaps.values())
    vmax = max(h.max() for h in all_hmaps.values())

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 2 * n_rows))

    for col, (title, _) in enumerate(variants):
        for row, ms in enumerate(milestones):
            im = axes[row, col].imshow(all_hmaps[(col, ms)],
                                       cmap="inferno",
                                       vmin=vmin, vmax=vmax,
                                       aspect="auto",
                                       interpolation="nearest")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=8)
            if col == 0:
                axes[row, col].set_ylabel(f"step {ms}", fontsize=9)

    fig.colorbar(im, ax=axes, shrink=0.6, label="||score||")
    fig.suptitle(f"Score magnitude heatmap ({args.model_type}, "
                 f"pair={args.left}{args.right}, w={args.guidance_scale})",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved heatmap to {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_model_args(parser):
    """Add common generative-model arguments to a subparser."""
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["diffusion", "flow"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str,
                        default="vae_checkpoint.npz")
    parser.add_argument("--left", type=int, required=True,
                        help="Left digit label")
    parser.add_argument("--right", type=int, required=True,
                        help="Right digit label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=3.0,
                        help="CFG guidance scale w")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image path (.png)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualization tools for OOD generalization experiments")
    subparsers = parser.add_subparsers(dest="command")

    # --- recon ---
    p_recon = subparsers.add_parser("recon",
                                    help="VAE reconstruction visualization")
    p_recon.add_argument("--vae-checkpoint", type=str,
                         default="vae_checkpoint.npz")
    p_recon.add_argument("--data", type=str, required=True,
                         help="Path to bi-digit .pkl dataset")
    p_recon.add_argument("--num-samples", type=int, default=8,
                         help="Number of random samples to show")
    p_recon.add_argument("--seed", type=int, default=42)
    p_recon.add_argument("--output", type=str, required=True,
                         help="Output image path (.png)")

    # --- interference ---
    p_inter = subparsers.add_parser(
        "interference",
        help="Plot MSE/cosine between joint and sum scores across steps")
    _add_model_args(p_inter)

    # --- progress ---
    p_prog = subparsers.add_parser(
        "progress",
        help="Decoded image progress at milestone steps for 5 score variants")
    _add_model_args(p_prog)

    # --- heatmap ---
    p_heat = subparsers.add_parser(
        "heatmap",
        help="Score magnitude heatmap at milestone steps for 5 score variants")
    _add_model_args(p_heat)

    args = parser.parse_args()

    if args.command == "recon":
        cmd_recon(args)
    elif args.command == "interference":
        cmd_interference(args)
    elif args.command == "progress":
        cmd_progress(args)
    elif args.command == "heatmap":
        cmd_heatmap(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
