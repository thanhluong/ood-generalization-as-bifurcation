#!/usr/bin/env python3
"""Evaluate hybrid sampling across sliding bifurcation windows.

For a given window length l, tests windows [0, l], [0.05, 0.05+l], ..., [1-l, 1].
Runs joint once, then hybrid for each window. Uses disentangled_judge.py to score.
Plots improvement (%) of hybrid over joint for Left, Right, and Pair accuracy.
"""

import argparse
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


def run_cmd(cmd, desc=""):
    """Run a shell command, printing it and checking for errors."""
    print(f"  [{desc}] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}")
        sys.exit(1)
    if result.stdout.strip():
        print(f"  {result.stdout.strip()[-200:]}")
    return result.stdout


def parse_judge_summary(stdout):
    """Parse 'Summary: Left X% | Right Y% | Pair Z%' from judge stdout."""
    m = re.search(
        r"Summary:\s*Left\s+([\d.]+)%\s*\|\s*Right\s+([\d.]+)%\s*\|\s*Pair\s+([\d.]+)%",
        stdout,
    )
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    raise ValueError(f"Could not parse judge summary from:\n{stdout}")


def sample_and_judge(
    model_type, checkpoint, vae_checkpoint, judge_checkpoint,
    seed_lo, seed_hi, pairs, guidance_scale, steps,
    strategy, bif_start, bif_end, output_dir, report_path,
):
    """Run mass_sampling.py then disentangled_judge.py. Return (left%, right%, pair%)."""
    # Sample
    sample_cmd = [
        sys.executable, "mass_sampling.py",
        "--model-type", model_type,
        "--checkpoint", checkpoint,
        "--vae-checkpoint", vae_checkpoint,
        "--seed-lo", str(seed_lo),
        "--seed-hi", str(seed_hi),
        "--pairs", pairs,
        "--output-dir", output_dir,
        "--guidance-scale", str(guidance_scale),
        "--steps", str(steps),
        "--strategy", strategy,
    ]
    if strategy == "hybrid":
        sample_cmd += ["--bif-start", str(bif_start), "--bif-end", str(bif_end)]
    run_cmd(sample_cmd, desc=f"sample {strategy}")

    # Judge
    judge_cmd = [
        sys.executable, "disentangled_judge.py",
        "--mode", "judge",
        "--checkpoint", judge_checkpoint,
        "--samples-dir", output_dir,
        "--output", report_path,
    ]
    stdout = run_cmd(judge_cmd, desc=f"judge {strategy}")
    return parse_judge_summary(stdout)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid sampling across sliding bifurcation windows")
    parser.add_argument("--model-type", type=str, default="flow",
                        choices=["diffusion", "flow"])
    parser.add_argument("--checkpoint", type=str, default="cfm_checkpoint.npz")
    parser.add_argument("--vae-checkpoint", type=str, default="vae_checkpoint.npz")
    parser.add_argument("--judge-checkpoint", type=str, default="judge_checkpoint.npz")
    parser.add_argument("--seed-lo", type=int, default=1)
    parser.add_argument("--seed-hi", type=int, default=50)
    parser.add_argument("--pairs", type=str, default="28,32,53,85,22,33,55,88")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--window-length", type=float, default=0.2,
                        help="Bifurcation window length (must be divisible by 0.05)")
    parser.add_argument("--output", type=str, default="eval_bifurcation_windows.png",
                        help="Output plot filename")
    args = parser.parse_args()

    l = args.window_length
    # Validate divisibility by 0.05
    if round(l / 0.05) != l / 0.05 or l <= 0 or l > 1:
        print("Error: --window-length must be > 0, <= 1, and divisible by 0.05")
        sys.exit(1)

    # Build sliding windows
    step = 0.05
    windows = []
    start = 0.0
    while start + l <= 1.0 + 1e-9:
        end = min(start + l, 1.0)
        windows.append((round(start, 2), round(end, 2)))
        start = round(start + step, 2)

    print(f"Window length: {l}, Windows to evaluate: {len(windows)}")
    print(f"Seeds: {args.seed_lo}-{args.seed_hi}")

    base_dir = "generated_eval_bif"
    os.makedirs(base_dir, exist_ok=True)

    # --- Run joint once ---
    joint_dir = os.path.join(base_dir, "generated_joint")
    joint_report = os.path.join(base_dir, "report_joint.md")
    print("\n=== Joint sampling ===")
    joint_acc = sample_and_judge(
        args.model_type, args.checkpoint, args.vae_checkpoint,
        args.judge_checkpoint, args.seed_lo, args.seed_hi, args.pairs,
        args.guidance_scale, args.steps,
        "joint", 0.0, 1.0, joint_dir, joint_report,
    )
    print(f"  Joint => Left {joint_acc[0]:.1f}% | Right {joint_acc[1]:.1f}% | Pair {joint_acc[2]:.1f}%")

    # --- Run hybrid for each window ---
    hybrid_results = []
    for i, (ws, we) in enumerate(windows):
        tag = f"{ws:.2f}_{we:.2f}"
        hybrid_dir = os.path.join(base_dir, f"generated_hybrid_{tag}")
        hybrid_report = os.path.join(base_dir, f"report_hybrid_{tag}.md")
        print(f"\n=== Hybrid [{ws:.2f}, {we:.2f}] ({i+1}/{len(windows)}) ===")
        acc = sample_and_judge(
            args.model_type, args.checkpoint, args.vae_checkpoint,
            args.judge_checkpoint, args.seed_lo, args.seed_hi, args.pairs,
            args.guidance_scale, args.steps,
            "hybrid", ws, we, hybrid_dir, hybrid_report,
        )
        print(f"  Hybrid => Left {acc[0]:.1f}% | Right {acc[1]:.1f}% | Pair {acc[2]:.1f}%")
        hybrid_results.append((ws, we, acc))

    # --- Compute improvements over joint ---
    x_labels = [f"[{ws:.2f},{we:.2f}]" for ws, we, _ in hybrid_results]
    x_pos = np.arange(len(hybrid_results))

    left_imp = [acc[0] - joint_acc[0] for _, _, acc in hybrid_results]
    right_imp = [acc[1] - joint_acc[1] for _, _, acc in hybrid_results]
    pair_imp = [acc[2] - joint_acc[2] for _, _, acc in hybrid_results]

    metrics = [
        ("Left Accuracy", left_imp),
        ("Right Accuracy", right_imp),
        ("Pair Accuracy", pair_imp),
    ]

    suptitle = (
        f"Hybrid vs Joint Improvement  |  "
        f"window-length={l}  |  seeds {args.seed_lo}-{args.seed_hi}"
    )

    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(windows) * 0.8), 12),
                             sharex=True)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for idx, (ax, (metric_name, improvements), color) in enumerate(
        zip(axes, metrics, colors)
    ):
        bars = ax.bar(x_pos, improvements, color=color, alpha=0.85, edgecolor="white")

        # Find optimal window
        best_idx = int(np.argmax(improvements))
        best_val = improvements[best_idx]

        # Highlight optimal bar
        bars[best_idx].set_edgecolor("red")
        bars[best_idx].set_linewidth(2.5)

        # Star marker above optimal bar
        ax.annotate(
            f"★ {best_val:+.1f}%",
            xy=(best_idx, best_val),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="red",
        )

        # Value labels on bars
        for j, v in enumerate(improvements):
            if j == best_idx:
                continue
            ax.annotate(
                f"{v:+.1f}%",
                xy=(j, v),
                xytext=(0, 3 if v >= 0 else -12),
                textcoords="offset points",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, color="gray",
            )

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel(f"Δ {metric_name} (%)", fontsize=11)
        ax.set_title(
            f"{metric_name}  (joint baseline: {joint_acc[idx]:.1f}%)",
            fontsize=11,
        )
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(x_pos)
    axes[-1].set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel("Bifurcation Window [start, end]", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Window':<18} {'Left Δ':>8} {'Right Δ':>8} {'Pair Δ':>8}")
    print("-" * 70)
    for (ws, we, acc), li, ri, pi in zip(hybrid_results, left_imp, right_imp, pair_imp):
        print(f"[{ws:.2f}, {we:.2f}]      {li:>+7.1f}% {ri:>+7.1f}% {pi:>+7.1f}%")
    print("=" * 70)

    best_pair = int(np.argmax(pair_imp))
    bw = hybrid_results[best_pair]
    print(f"\n★ Best window (Pair): [{bw[0]:.2f}, {bw[1]:.2f}] "
          f"with Δ={pair_imp[best_pair]:+.1f}%")


if __name__ == "__main__":
    main()
