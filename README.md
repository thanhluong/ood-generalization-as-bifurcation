# OOD Generalization as Bifurcation

Comparing classifier-free diffusion guidance (CFDG) and conditional flow matching (CFM) on out-of-distribution bi-digit MNIST composition. Both models learn to generate digit pairs from a restricted set of observed combinations, then must generalize to unseen compositions.

## Setup

```bash
source ../venv/bin/activate
```

## Pipeline

### Stage 0: Create bi-digit dataset (if needed)

```bash
python create_bidgit_data.py
python create_bidgit_data.py --reverse  # for CFM variant
```

### Stage 1: Train VAE

```bash
python train_vae.py --data mnist_bi_digit_vae.pkl --epochs 100
```

### Stage 2a: Train Diffusion (CFDG)

```bash
python train_diffusion.py \
    --data mnist_bi_digit_diffusion_and_flow.pkl \
    --vae-checkpoint vae_checkpoint.npz \
    --checkpoint diffusion_checkpoint.npz \
    --epochs 200 \
    --seed 42
```

Resume from checkpoint:

```bash
python train_diffusion.py \
    --data mnist_bi_digit_diffusion_and_flow.pkl \
    --checkpoint diffusion_checkpoint.npz \
    --resume \
    --epochs 100
```

### Stage 2b: Train Flow Matching (CFM)

```bash
python train_flow.py \
    --data mnist_bi_digit_diffusion_and_flow.pkl \
    --vae-checkpoint vae_checkpoint.npz \
    --checkpoint cfm_checkpoint.npz \
    --epochs 200 \
    --seed 42
```

### Stage 3: Mass Sampling

Generate OOD digit pairs with different scoring strategies:

```bash
# Joint scoring (default, standard CFG)
python mass_sampling.py \
    --model-type flow \
    --checkpoint cfm_checkpoint.npz \
    --seed-lo 0 --seed-hi 99 \
    --output-dir samples_flow_joint/ \
    --strategy joint

# Decomposed scoring: e(L,0) + e(0,R) - e(0,0)
python mass_sampling.py \
    --model-type flow \
    --checkpoint cfm_checkpoint.npz \
    --seed-lo 0 --seed-hi 99 \
    --output-dir samples_flow_decomposed/ \
    --strategy decomposed

# Hybrid: decomposed in bifurcation window, joint outside
python mass_sampling.py \
    --model-type flow \
    --checkpoint cfm_checkpoint.npz \
    --seed-lo 0 --seed-hi 99 \
    --output-dir samples_flow_hybrid/ \
    --strategy hybrid --bif-start 0.3 --bif-end 0.7
```

Each seed produces a 84x56 PNG image (3 rows of 28x56 bi-digit images stacked vertically).

### Stage 4: Evaluate with Judge

```bash
python disentangled_judge.py \
    --mode judge \
    --checkpoint judge_checkpoint.npz \
    --samples-dir samples_flow_joint/ \
    --output judge_report.md
```

## Visualization

```bash
# VAE reconstruction quality
python visualize.py recon \
    --data mnist_bi_digit_vae.pkl \
    --output recon.png

# Interference: MSE & cosine similarity between joint and sum scores
python visualize.py interference \
    --model-type flow --checkpoint cfm_checkpoint.npz \
    --left 2 --right 8 \
    --output interference.png

# Generation progress (decoded images at 5 milestone steps)
python visualize.py progress \
    --model-type flow --checkpoint cfm_checkpoint.npz \
    --left 2 --right 8 \
    --output progress.png

# Score magnitude heatmaps in latent space
python visualize.py heatmap \
    --model-type flow --checkpoint cfm_checkpoint.npz \
    --left 2 --right 8 \
    --output heatmap.png
```

## Sampling Strategies

The project tests the **superposition principle**: `e(L, R) ≈ e(L, 0) + e(0, R) - e(0, 0)`, where label `0` = blank (black image half).

| Strategy | Conditional Score | Cost per Step |
|----------|------------------|---------------|
| `joint` | `e(L, R)` | 2 forward passes (cond + uncond) |
| `decomposed` | `e(L,0) + e(0,R) - e(0,0)` | 4 forward passes |
| `hybrid` | decomposed in bifurcation window, joint outside | 2-4 depending on step |

A **semantic bifurcation window** exists mid-sampling where joint and sum scores diverge. The hybrid strategy uses decomposed scoring only inside this window (`--bif-start`/`--bif-end`), aiming for better OOD accuracy than joint at lower cost than full decomposed.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--guidance-scale` | 3.0 | CFG strength during sampling |
| `--steps` | 50 | Number of ODE solver steps |
| `--strategy` | joint | Scoring strategy: joint, decomposed, hybrid |
| `--bif-start` | 0.3 | Bifurcation window start (hybrid only) |
| `--bif-end` | 0.7 | Bifurcation window end (hybrid only) |
| `--eval-interval` | 10 | Interference metric logging frequency |
| `--lr` | 1e-4 | Learning rate for stage-2 training |
| `--batch-size` | 64 | Training batch size |

## Observed vs OOD Pairs

- **Observed (training):** 23, 25, 35, 38, 58, 52, 82, 83
- **OOD (test):** 28, 32, 53, 85, 22, 33, 55, 88

The models never see OOD combinations during training. Generalization relies on disentangled left/right digit embeddings learned through structured label dropout.
