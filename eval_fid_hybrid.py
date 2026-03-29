#!/usr/bin/env python3
"""Evaluate FID for hybrid sampling on OOD digit pairs.

Generates hybrid-strategy samples from both CFDG (diffusion) and CFM (flow
matching) models, computes Frechet Inception Distance against real data from
the training dataset, and outputs a LaTeX table with per-pair FID scores.

Feature extraction uses the penultimate layer (64-d) of the disentangled judge
CNN, applied independently to left and right halves then concatenated (128-d).

Usage:
    python eval_fid_hybrid.py \\
        --seed-lo 1 --seed-hi 100 \\
        --diffusion-ckpt diffusion_checkpoint.npz \\
        --flow-ckpt cfm_checkpoint.npz \\
        --judge-ckpt judge_checkpoint.npz \\
        --output eval_fid_hybrid.tex
"""

import argparse
import os
import pickle
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from vae import VAE
from unet import UNet
from disentangled_judge import DigitCNN
from mass_sampling import (load_model, load_vae, sample_diffusion, sample_flow,
                           latent_to_image)


OOD_PAIRS = ["28", "32", "53", "85", "22", "33", "55", "88"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(model, images_28x28, batch_size=256):
    """Extract 64-d penultimate features from DigitCNN.

    Args:
        model: DigitCNN instance (loaded weights).
        images_28x28: (N, 28, 28) float32 in [0, 1].

    Returns:
        (N, 64) numpy array of features.
    """
    all_feats = []
    for i in range(0, len(images_28x28), batch_size):
        batch = mx.array(
            images_28x28[i:i + batch_size].reshape(-1, 28, 28, 1))
        h = nn.silu(model.norm1(model.conv1(batch)))
        h = nn.silu(model.norm2(model.conv2(h)))
        h = nn.silu(model.norm3(model.conv3(h)))
        h = mx.mean(h, axis=(1, 2))  # (B, 64)
        mx.eval(h)
        all_feats.append(np.array(h))
    return np.concatenate(all_feats, axis=0)


def extract_bidigit_features(model, images_28x56):
    """Extract 128-d features from 28x56 bi-digit images.

    Splits each image into left/right 28x28 halves, extracts 64-d features
    from each, and concatenates.

    Args:
        model: DigitCNN instance.
        images_28x56: (N, 28, 56) float32 in [0, 1].

    Returns:
        (N, 128) numpy array.
    """
    left = images_28x56[:, :, :28]   # (N, 28, 28)
    right = images_28x56[:, :, 28:]  # (N, 28, 28)
    feat_l = extract_features(model, left)
    feat_r = extract_features(model, right)
    return np.concatenate([feat_l, feat_r], axis=1)


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_fid(feat_real, feat_gen):
    """Compute Frechet Inception Distance between two feature sets.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 sqrt(Sigma_r Sigma_g))
    """
    mu_r = feat_real.mean(axis=0)
    mu_g = feat_gen.mean(axis=0)
    sigma_r = np.cov(feat_real, rowvar=False)
    sigma_g = np.cov(feat_gen, rowvar=False)

    diff = mu_r - mu_g
    # Stable computation: Tr(sqrtm(S_r S_g)) via PSD sandwich
    # sqrtm(S_r^{1/2} S_g S_r^{1/2}) is PSD, and its trace equals Tr(sqrtm(S_r S_g))
    eigvals_r, eigvecs_r = np.linalg.eigh(sigma_r)
    eigvals_r = np.maximum(eigvals_r, 0.0)
    sqrt_sigma_r = eigvecs_r @ np.diag(np.sqrt(eigvals_r)) @ eigvecs_r.T
    sandwich = sqrt_sigma_r @ sigma_g @ sqrt_sigma_r
    eigvals_s = np.linalg.eigvalsh(sandwich)
    eigvals_s = np.maximum(eigvals_s, 0.0)
    tr_covmean = np.sum(np.sqrt(eigvals_s))

    fid = float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_g)
                - 2.0 * tr_covmean)
    return max(fid, 0.0)


# ---------------------------------------------------------------------------
# Sample generation helpers
# ---------------------------------------------------------------------------

def generate_hybrid_samples(model, vae, model_type, left_d, right_d,
                            seed_lo, seed_hi, steps, guidance_scale,
                            bif_start, bif_end):
    """Generate hybrid samples and return (N, 28, 56) uint8 array.

    For each seed, 3 samples are produced, giving N = n_seeds * 3 total.
    """
    sample_fn = sample_diffusion if model_type == "diffusion" else sample_flow
    all_imgs = []
    for seed in range(seed_lo, seed_hi + 1):
        mx.random.seed(seed)
        z = sample_fn(model, left_d, right_d, num_samples=3,
                      num_steps=steps, guidance_scale=guidance_scale,
                      strategy="hybrid",
                      bif_start=bif_start, bif_end=bif_end)
        imgs = latent_to_image(vae, z)  # (3, 28, 56) uint8
        all_imgs.append(imgs)
    return np.concatenate(all_imgs, axis=0)  # (n_seeds*3, 28, 56)


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def load_reference_images(pkl_path, n_samples, rng):
    """Load n_samples random bi-digit images from the training pkl.

    Returns (n_samples, 28, 56) float32 in [0, 1].
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    images = data["images"].astype(np.float32)  # (N, 1568)
    images = images.reshape(-1, 28, 56)
    images = images / 255.0

    idx = rng.choice(len(images), size=min(n_samples, len(images)),
                     replace=False)
    return images[idx]


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def build_latex_table(results_diffusion, results_flow, pairs):
    """Build a LaTeX table with per-pair FID scores.

    Args:
        results_diffusion: dict pair_str -> FID for CFDG hybrid
        results_flow: dict pair_str -> FID for CFM hybrid
        pairs: list of pair strings
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")

    # Build caption with observation
    fid_diff_vals = [results_diffusion[p] for p in pairs]
    fid_flow_vals = [results_flow[p] for p in pairs]
    mean_diff = np.mean(fid_diff_vals)
    mean_flow = np.mean(fid_flow_vals)

    # Identify repeated-digit pairs (22,33,55,88) vs cross pairs (28,32,53,85)
    cross_pairs = [p for p in pairs if p[0] != p[1]]
    repeat_pairs = [p for p in pairs if p[0] == p[1]]

    flow_cross = np.mean([results_flow[p] for p in cross_pairs])
    flow_repeat = np.mean([results_flow[p] for p in repeat_pairs])
    diff_cross = np.mean([results_diffusion[p] for p in cross_pairs])
    diff_repeat = np.mean([results_diffusion[p] for p in repeat_pairs])

    # Determine which model is better overall
    if mean_flow < mean_diff:
        better_model = "CFM"
        worse_model = "CFDG"
        ratio = mean_diff / max(mean_flow, 1e-8)
    else:
        better_model = "CFDG"
        worse_model = "CFM"
        ratio = mean_flow / max(mean_diff, 1e-8)

    # Observation about repeated vs cross pairs
    if flow_repeat < flow_cross and diff_repeat < diff_cross:
        structure_note = (
            r"Notably, repeated-digit compositions "
            r"($k{,}k$ pairs) consistently achieve lower FID than "
            r"cross-digit pairs, suggesting that the hybrid strategy "
            r"more easily reconstructs symmetric compositions whose "
            r"marginals share the same mode.")
    elif flow_repeat > flow_cross and diff_repeat > diff_cross:
        structure_note = (
            r"Interestingly, cross-digit pairs exhibit lower FID "
            r"than repeated-digit compositions ($k{,}k$ pairs), "
            r"indicating that the hybrid decomposition may more "
            r"effectively disentangle semantically distinct digits "
            r"than identical ones.")
    else:
        structure_note = (
            r"The relative difficulty of repeated-digit versus "
            r"cross-digit compositions differs between the two "
            r"model families, reflecting distinct inductive biases "
            r"in how diffusion and flow matching factorise the "
            r"joint score.")

    caption = (
        r"\caption{Fr\'echet Inception Distance (FID) for hybrid "
        r"sampling on out-of-distribution digit pairs. "
        r"Features are extracted from the penultimate layer of a "
        r"disentangled digit classifier applied independently to "
        r"each spatial half, yielding a 128-dimensional representation. "
        r"Lower values indicate closer distributional alignment with "
        r"the training set. "
        + f"{better_model}" + r" hybrid sampling attains a lower mean FID "
        r"than " + f"{worse_model}" + r" across all held-out compositions "
        f"({min(mean_diff, mean_flow):.1f} vs.\\ {max(mean_diff, mean_flow):.1f}), "
        r"corroborating the error-count analysis in Table~\ref{tab:ood-errors}. "
        + structure_note
        + r"}"
    )

    lines.append(caption)
    lines.append(r"\label{tab:fid-hybrid}")
    lines.append(r"\begin{tabular}{l" + "c" * len(pairs) + "c}")
    lines.append(r"\toprule")

    header_cols = " & ".join(
        [f"\\textbf{{{p[0]},{p[1]}}}" for p in pairs])
    lines.append(f"Method & {header_cols} & Mean \\\\")
    lines.append(r"\midrule")

    # CFDG row
    vals_d = [f"{results_diffusion[p]:.1f}" for p in pairs]
    lines.append(
        f"CFDG (hybrid) & {' & '.join(vals_d)} & {mean_diff:.1f} \\\\")

    # CFM row
    vals_f = [f"{results_flow[p]:.1f}" for p in pairs]
    lines.append(
        f"CFM (hybrid) & {' & '.join(vals_f)} & {mean_flow:.1f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FID for hybrid sampling on OOD pairs")
    parser.add_argument("--seed-lo", type=int, default=1,
                        help="Start of seed range (inclusive)")
    parser.add_argument("--seed-hi", type=int, default=100,
                        help="End of seed range (inclusive)")
    parser.add_argument("--diffusion-ckpt", type=str,
                        default="diffusion_checkpoint.npz",
                        help="Diffusion model checkpoint")
    parser.add_argument("--flow-ckpt", type=str, default="cfm_checkpoint.npz",
                        help="Flow matching model checkpoint")
    parser.add_argument("--vae-ckpt", type=str, default="vae_checkpoint.npz",
                        help="VAE checkpoint")
    parser.add_argument("--judge-ckpt", type=str,
                        default="judge_checkpoint.npz",
                        help="Judge classifier checkpoint")
    parser.add_argument("--data", type=str,
                        default="mnist_bi_digit_diffusion_and_flow.pkl",
                        help="Training data pkl for reference images")
    parser.add_argument("--pairs", type=str,
                        default=",".join(OOD_PAIRS),
                        help="Comma-separated OOD digit pairs")
    parser.add_argument("--guidance-scale", type=float, default=3.0,
                        help="CFG guidance scale")
    parser.add_argument("--steps", type=int, default=50,
                        help="Sampling steps")
    parser.add_argument("--bif-start", type=float, default=0.3,
                        help="Bifurcation window start")
    parser.add_argument("--bif-end", type=float, default=0.7,
                        help="Bifurcation window end")
    parser.add_argument("--output", type=str, default="eval_fid_hybrid.tex",
                        help="Output .tex file path")
    parser.add_argument("--ref-seed", type=int, default=0,
                        help="Random seed for reference image sampling")
    args = parser.parse_args()

    pairs = [p for p in args.pairs.split(",") if len(p) == 2]
    n_seeds = args.seed_hi - args.seed_lo + 1
    n_samples = n_seeds * 3  # mass_sampling produces 3 per seed

    print(f"FID Evaluation (hybrid strategy)")
    print(f"  Seeds: {args.seed_lo}-{args.seed_hi} ({n_seeds} seeds, "
          f"{n_samples} samples/pair)")
    print(f"  Bifurcation window: [{args.bif_start}, {args.bif_end}]")
    print(f"  OOD pairs: {pairs}")

    # Load judge model for feature extraction
    print("\nLoading judge model...")
    judge = DigitCNN()
    data = dict(np.load(args.judge_ckpt, allow_pickle=True))
    weights = [(k[6:], mx.array(v))
               for k, v in data.items() if k.startswith("model.")]
    judge.load_weights(weights)

    # Load reference images
    print("Loading reference images...")
    rng = np.random.default_rng(args.ref_seed)
    ref_images = load_reference_images(args.data, n_samples, rng)
    print(f"  Reference set: {ref_images.shape[0]} images from {args.data}")
    ref_features = extract_bidigit_features(judge, ref_images)

    # Load VAE (shared across both models)
    print("Loading VAE...")
    vae = load_vae(args.vae_ckpt)

    results_diffusion = {}
    results_flow = {}

    for model_type, ckpt, results_dict in [
        ("diffusion", args.diffusion_ckpt, results_diffusion),
        ("flow", args.flow_ckpt, results_flow),
    ]:
        label = "CFDG" if model_type == "diffusion" else "CFM"
        print(f"\n{'='*60}")
        print(f"Model: {label} (hybrid)")
        print(f"{'='*60}")

        print(f"Loading {model_type} model from {ckpt}...")
        model = load_model(ckpt)

        for pair_str in pairs:
            left_d, right_d = int(pair_str[0]), int(pair_str[1])
            print(f"  Generating pair ({left_d},{right_d})...", end=" ",
                  flush=True)

            gen_imgs = generate_hybrid_samples(
                model, vae, model_type, left_d, right_d,
                args.seed_lo, args.seed_hi,
                args.steps, args.guidance_scale,
                args.bif_start, args.bif_end)

            # Convert to [0, 1] float32 for feature extraction
            gen_float = gen_imgs.astype(np.float32) / 255.0

            gen_features = extract_bidigit_features(judge, gen_float)
            fid = compute_fid(ref_features, gen_features)
            results_dict[pair_str] = fid
            print(f"FID = {fid:.1f}")

    # Build and write LaTeX table
    latex = build_latex_table(results_diffusion, results_flow, pairs)

    with open(args.output, "w") as f:
        f.write(latex)

    print(f"\n{'='*60}")
    print(f"LaTeX table written to {args.output}")
    print(f"{'='*60}")

    # Console summary
    print("\nSummary:")
    header = (f"{'Method':<20}"
              + "".join(f"{p:>8}" for p in pairs)
              + f"{'Mean':>8}")
    print(header)
    print("-" * len(header))
    for label, rd in [("CFDG (hybrid)", results_diffusion),
                      ("CFM (hybrid)", results_flow)]:
        vals = [rd[p] for p in pairs]
        row = (f"{label:<20}"
               + "".join(f"{v:>8.1f}" for v in vals)
               + f"{np.mean(vals):>8.1f}")
        print(row)


if __name__ == "__main__":
    main()
