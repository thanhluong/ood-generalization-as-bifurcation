import argparse
import os
import numpy as np
import mlx.core as mx

from vae import VAE
from unet import UNet


NULL_TOKEN = 9
SIGMA_MIN = 1e-5
BLANK_LABEL = 0  # label 0 = blank (all-black image half)


def load_model(checkpoint_path):
    """Load a UNet model from checkpoint (weights only, no optimizer)."""
    model = UNet()
    data = dict(np.load(checkpoint_path, allow_pickle=True))
    weights = [(k[6:], mx.array(v))
               for k, v in data.items() if k.startswith("model.")]
    model.load_weights(weights)
    return model


def load_vae(checkpoint_path):
    """Load a frozen VAE from checkpoint."""
    vae = VAE()
    data = dict(np.load(checkpoint_path, allow_pickle=True))
    weights = [(k[6:], mx.array(v))
               for k, v in data.items() if k.startswith("model.")]
    vae.load_weights(weights)
    vae.freeze()
    return vae


def cfg_combine(output_cond, output_uncond, guidance_scale):
    """Classifier-free guidance: (1+w)*cond - w*uncond."""
    return (1.0 + guidance_scale) * output_cond - guidance_scale * output_uncond


def compute_guided_score(model, z, t_batch, left_cond, right_cond,
                         null_left, null_right, guidance_scale,
                         strategy, progress, bif_start, bif_end):
    """Compute CFG-guided score based on sampling strategy.

    Args:
        strategy: "joint", "decomposed", or "hybrid"
        progress: float in [0, 1], fraction of sampling steps completed
        bif_start/bif_end: bifurcation window bounds (used only for hybrid)
    """
    B = z.shape[0]
    blank = mx.full((B,), BLANK_LABEL, dtype=mx.int32)

    use_decomposed = (strategy == "decomposed" or
                      (strategy == "hybrid" and
                       bif_start <= progress < bif_end))

    out_uncond = model(z, t_batch, null_left, null_right)
    if use_decomposed:
        out_left = model(z, t_batch, left_cond, null_right)
        out_right = model(z, t_batch, null_left, right_cond)
        out_decomposed = out_uncond + guidance_scale * (out_left + out_right - 2 * out_uncond)
        return out_decomposed
    else:
        out_cond = model(z, t_batch, left_cond, right_cond)
        return out_uncond + guidance_scale * (out_cond - out_uncond)


def sample_diffusion(model, left_digit, right_digit, num_samples=3,
                     num_steps=50, guidance_scale=3.0,
                     strategy="joint", bif_start=0.0, bif_end=1.0):
    """DDIM sampling for classifier-free diffusion guidance.

    Backward Euler from t=1 (noise) to t=0 (clean).
    Noise schedule: alpha_bar(t) = 1 - t.
    """
    latent_shape = (num_samples, 7, 14, 2)
    z = mx.random.normal(latent_shape)

    left_cond = mx.full((num_samples,), left_digit, dtype=mx.int32)
    right_cond = mx.full((num_samples,), right_digit, dtype=mx.int32)
    left_null = mx.full((num_samples,), NULL_TOKEN, dtype=mx.int32)
    right_null = mx.full((num_samples,), NULL_TOKEN, dtype=mx.int32)

    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_val = 0.999 - step * dt
        t_next = t_val - dt
        t_batch = mx.full((num_samples,), t_val)
        progress = step / num_steps

        # alpha_bar(t) = 1 - t
        ab_t = 1.0 - t_val
        ab_next = 1.0 - t_next

        eps_guided = compute_guided_score(
            model, z, t_batch, left_cond, right_cond,
            left_null, right_null, guidance_scale,
            strategy, progress, bif_start, bif_end)

        # DDIM deterministic step
        sqrt_ab = mx.sqrt(mx.maximum(mx.array(ab_t), mx.array(1e-8)))
        sqrt_one_minus_ab = mx.sqrt(mx.maximum(mx.array(1.0 - ab_t), mx.array(1e-8)))
        z_0_pred = (z - sqrt_one_minus_ab * eps_guided) / sqrt_ab

        sqrt_ab_next = mx.sqrt(mx.maximum(mx.array(ab_next), mx.array(1e-8)))
        sqrt_one_minus_ab_next = mx.sqrt(
            mx.maximum(mx.array(1.0 - ab_next), mx.array(1e-8)))
        z = sqrt_ab_next * z_0_pred + sqrt_one_minus_ab_next * eps_guided
        mx.eval(z)

    return z


def sample_flow(model, left_digit, right_digit, num_samples=3,
                num_steps=50, guidance_scale=3.0,
                strategy="joint", bif_start=0.0, bif_end=1.0):
    """Forward Euler ODE sampling for conditional flow matching.

    Forward from t=0 (noise) to t=1 (data).
    """
    latent_shape = (num_samples, 7, 14, 2)
    z = mx.random.normal(latent_shape)

    left_cond = mx.full((num_samples,), left_digit, dtype=mx.int32)
    right_cond = mx.full((num_samples,), right_digit, dtype=mx.int32)
    left_null = mx.full((num_samples,), NULL_TOKEN, dtype=mx.int32)
    right_null = mx.full((num_samples,), NULL_TOKEN, dtype=mx.int32)

    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_val = step * dt
        t_batch = mx.full((num_samples,), t_val)
        progress = step / num_steps

        v_guided = compute_guided_score(
            model, z, t_batch, left_cond, right_cond,
            left_null, right_null, guidance_scale,
            strategy, progress, bif_start, bif_end)

        z = z + dt * v_guided
        mx.eval(z)

    return z


def latent_to_image(vae, z):
    """Decode latents through VAE decoder and convert to uint8 numpy array.

    Args:
        vae: Frozen VAE model
        z: (B, 7, 14, 2) latent tensor

    Returns:
        (B, 28, 56) uint8 numpy array
    """
    x = vae.decode(z)                   # (B, 28, 56, 1)
    mx.eval(x)
    x = np.array(x)
    x = (x + 1.0) * 127.5              # [-1, 1] -> [0, 255]
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x.squeeze(-1)                # (B, 28, 56)


def main():
    parser = argparse.ArgumentParser(
        description="Mass sampling from trained diffusion or flow matching models")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["diffusion", "flow"],
                        help="Which model to sample from")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--vae-checkpoint", type=str, default="vae_checkpoint.npz",
                        help="Path to VAE checkpoint")
    parser.add_argument("--seed-lo", type=int, required=True,
                        help="Start of seed range (inclusive)")
    parser.add_argument("--seed-hi", type=int, required=True,
                        help="End of seed range (inclusive)")
    parser.add_argument("--pairs", type=str, default="28,32,53,85,22,33,55,88",
                        help="Comma-separated digit pairs to generate")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for generated images")
    parser.add_argument("--guidance-scale", type=float, default=3.0,
                        help="Classifier-free guidance scale w")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of sampling steps")
    parser.add_argument("--strategy", type=str, default="joint",
                        choices=["joint", "decomposed", "hybrid"],
                        help="Scoring strategy: joint, decomposed, or hybrid")
    parser.add_argument("--bif-start", type=float, default=0.3,
                        help="Bifurcation window start (fraction 0-1, for hybrid)")
    parser.add_argument("--bif-end", type=float, default=0.7,
                        help="Bifurcation window end (fraction 0-1, for hybrid)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = [(int(p[0]), int(p[1])) for p in args.pairs.split(",") if len(p) == 2]
    if not pairs:
        print("Error: no valid digit pairs specified.")
        return

    print(f"Loading {args.model_type} model from {args.checkpoint}")
    model = load_model(args.checkpoint)

    print(f"Loading VAE from {args.vae_checkpoint}")
    vae = load_vae(args.vae_checkpoint)

    sample_fn = sample_diffusion if args.model_type == "diffusion" else sample_flow
    total_images = (args.seed_hi - args.seed_lo + 1) * len(pairs)
    print(f"Generating {total_images} images "
          f"(seeds {args.seed_lo}-{args.seed_hi}, pairs {args.pairs})")
    print(f"Strategy: {args.strategy}", end="")
    if args.strategy == "hybrid":
        print(f" (bifurcation window: {args.bif_start:.2f}-{args.bif_end:.2f})")
    else:
        print()

    from PIL import Image

    for seed in range(args.seed_lo, args.seed_hi + 1):
        print(f"Generating images for seed {seed}")
        for left_d, right_d in pairs:
            mx.random.seed(seed)

            z = sample_fn(model, left_d, right_d, num_samples=3,
                          num_steps=args.steps, guidance_scale=args.guidance_scale,
                          strategy=args.strategy,
                          bif_start=args.bif_start, bif_end=args.bif_end)
            imgs = latent_to_image(vae, z)  # (3, 28, 56)

            stacked = np.concatenate(list(imgs), axis=0)  # (84, 56)

            fname = f"seed{seed:04d}_{left_d}{right_d}.png"
            fpath = os.path.join(args.output_dir, fname)
            Image.fromarray(stacked, mode="L").save(fpath)

    print(f"Done. {total_images} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
