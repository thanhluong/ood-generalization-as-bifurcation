#!/usr/bin/env python3
"""Evaluate OOD generalization: joint vs decomposed × diffusion vs flow.

Generates samples with mass_sampling.py, judges them with disentangled_judge.py,
and outputs a LaTeX table (.tex) summarising mismatch counts per label pair.

Usage:
    python eval_ood_table.py \
        --seed-lo 1 --seed-hi 100 \
        --diffusion-ckpt diffusion_checkpoint.npz \
        --flow-ckpt cfm_checkpoint.npz \
        --judge-ckpt judge_checkpoint.npz \
        --output eval_ood_table.tex
"""

import argparse
import os
import sys
import subprocess
import tempfile
import re


OOD_PAIRS = ["28", "32", "53", "85", "22", "33", "55", "88"]

CONFIGS = [
    {"model_type": "diffusion", "strategy": "joint",      "label": "CFDG (joint)"},
    {"model_type": "diffusion", "strategy": "decomposed", "label": "CFDG (decomposed)"},
    {"model_type": "flow",      "strategy": "joint",      "label": "CFM (joint)"},
    {"model_type": "flow",      "strategy": "decomposed", "label": "CFM (decomposed)"},
]


def run_cmd(cmd, description=""):
    """Run a command, printing a summary, and return stdout."""
    print(f"  >> {description or ' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def parse_judge_report(report_path):
    """Parse judge_report.md and return per-label mismatch counts.

    Returns dict: label_str -> number of mismatched samples.
    """
    with open(report_path, "r") as f:
        text = f.read()

    mismatches = {}

    # Parse per-label breakdown table
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Label"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if len(cols) >= 5:
                label = cols[0]
                total = int(cols[1])
                # Pair Acc is the 4th data column (index 4)
                pair_acc_str = cols[4]
                # Parse percentage like "85.0%"
                pct_match = re.search(r"([\d.]+)%", pair_acc_str)
                if pct_match:
                    pair_acc = float(pct_match.group(1)) / 100.0
                    mismatches[label] = total - round(pair_acc * total)
                else:
                    mismatches[label] = total
        elif in_table and not line.startswith("|"):
            in_table = False

    return mismatches


def build_latex_table(all_results, seed_lo, seed_hi, pairs):
    """Build a complete LaTeX table string.

    all_results: list of (row_label, {pair_str: mismatch_count})
    """
    n_seeds = seed_hi - seed_lo + 1
    samples_per_seed = 3  # mass_sampling generates 3 samples per seed per pair
    total_per_cell = n_seeds * samples_per_seed

    num_cols = len(pairs)
    col_spec = "l" + "c" * num_cols + "c"
    header_cols = " & ".join([f"\\textbf{{{p[0]},{p[1]}}}" for p in pairs])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Out-of-distribution compositional generalization errors. "
                 r"Each cell reports the number of misclassified samples (out of "
                 f"{total_per_cell}) "
                 r"for a given model--strategy combination on held-out digit pairs "
                 r"never seen during training. Lower is better. "
                 r"A disentangled classifier independently predicts left and right digits; "
                 r"a sample is counted as an error if either prediction disagrees with the "
                 r"conditioning label.}")
    lines.append(r"\label{tab:ood-errors}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(f"Method & {header_cols} & Total \\\\")
    lines.append(r"\midrule")

    for row_label, mismatch_dict in all_results:
        vals = []
        row_total = 0
        for p in pairs:
            v = mismatch_dict.get(p, total_per_cell)
            vals.append(str(v))
            row_total += v
        lines.append(f"{row_label} & {' & '.join(vals)} & {row_total} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OOD generalization and produce a LaTeX error table")
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
    parser.add_argument("--pairs", type=str,
                        default=",".join(OOD_PAIRS),
                        help="Comma-separated digit pairs")
    parser.add_argument("--guidance-scale", type=float, default=3.0,
                        help="CFG guidance scale")
    parser.add_argument("--steps", type=int, default=50,
                        help="Sampling steps")
    parser.add_argument("--output", type=str, default="eval_ood_table.tex",
                        help="Output .tex file path")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Working directory for intermediate files "
                             "(default: auto temp dir)")
    parser.add_argument("--keep-samples", action="store_true",
                        help="Keep generated sample directories after evaluation")
    args = parser.parse_args()

    pairs = [p for p in args.pairs.split(",") if len(p) == 2]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.work_dir:
        work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)
        cleanup = False
    else:
        work_dir = tempfile.mkdtemp(prefix="eval_ood_")
        cleanup = not args.keep_samples

    print(f"OOD Evaluation: seeds {args.seed_lo}-{args.seed_hi}, "
          f"pairs {pairs}")
    print(f"Work directory: {work_dir}")

    all_results = []

    for cfg in CONFIGS:
        model_type = cfg["model_type"]
        strategy = cfg["strategy"]
        row_label = cfg["label"]

        ckpt = (args.diffusion_ckpt if model_type == "diffusion"
                else args.flow_ckpt)

        samples_dir = os.path.join(
            work_dir, f"samples_{model_type}_{strategy}")
        report_path = os.path.join(
            work_dir, f"report_{model_type}_{strategy}.md")

        print(f"\n{'='*60}")
        print(f"Config: {row_label}")
        print(f"{'='*60}")

        # Step 1: Generate samples
        print("Step 1: Generating samples...")
        run_cmd([
            sys.executable, os.path.join(script_dir, "mass_sampling.py"),
            "--model-type", model_type,
            "--checkpoint", ckpt,
            "--vae-checkpoint", args.vae_ckpt,
            "--seed-lo", str(args.seed_lo),
            "--seed-hi", str(args.seed_hi),
            "--pairs", args.pairs,
            "--output-dir", samples_dir,
            "--guidance-scale", str(args.guidance_scale),
            "--steps", str(args.steps),
            "--strategy", strategy,
        ], description=f"mass_sampling ({model_type}, {strategy})")

        # Step 2: Judge samples
        print("Step 2: Judging samples...")
        run_cmd([
            sys.executable, os.path.join(script_dir, "disentangled_judge.py"),
            "--mode", "judge",
            "--samples-dir", samples_dir,
            "--checkpoint", args.judge_ckpt,
            "--output", report_path,
        ], description=f"disentangled_judge ({model_type}, {strategy})")

        # Step 3: Parse report
        mismatches = parse_judge_report(report_path)
        print(f"  Mismatches: {mismatches}")
        all_results.append((row_label, mismatches))

        # Cleanup samples if requested
        if cleanup:
            import shutil
            shutil.rmtree(samples_dir, ignore_errors=True)

    # Build and write LaTeX table
    latex = build_latex_table(all_results, args.seed_lo, args.seed_hi, pairs)

    with open(args.output, "w") as f:
        f.write(latex)

    print(f"\n{'='*60}")
    print(f"LaTeX table written to {args.output}")
    print(f"{'='*60}")

    # Print summary to console
    print("\nSummary:")
    header = f"{'Method':<25}" + "".join(f"{p:>6}" for p in pairs) + f"{'Total':>8}"
    print(header)
    print("-" * len(header))
    for row_label, mismatch_dict in all_results:
        vals = [mismatch_dict.get(p, 0) for p in pairs]
        row = f"{row_label:<25}" + "".join(f"{v:>6}" for v in vals) + f"{sum(vals):>8}"
        print(row)

    if cleanup:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
