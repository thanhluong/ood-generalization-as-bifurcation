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

Generate OOD digit pairs from a trained model:

```bash
# Diffusion model
python mass_sampling.py \
    --model-type diffusion \
    --checkpoint diffusion_checkpoint.npz \
    --vae-checkpoint vae_checkpoint.npz \
    --seed-lo 0 --seed-hi 99 \
    --pairs 28,32,53,85,22,33,55,88 \
    --output-dir samples_diffusion/

# Flow matching model
python mass_sampling.py \
    --model-type flow \
    --checkpoint cfm_checkpoint.npz \
    --vae-checkpoint vae_checkpoint.npz \
    --seed-lo 0 --seed-hi 99 \
    --pairs 28,32,53,85,22,33,55,88 \
    --output-dir samples_flow/
```

Each seed produces a 84x56 PNG image (3 rows of 28x56 bi-digit images stacked vertically).

### Stage 4: Evaluate with Judge

```bash
# Judge diffusion samples
python disentangled_judge.py \
    --mode judge \
    --checkpoint judge_checkpoint.npz \
    --samples-dir samples_diffusion/ \
    --output judge_report_diffusion.md

# Judge flow samples
python disentangled_judge.py \
    --mode judge \
    --checkpoint judge_checkpoint.npz \
    --samples-dir samples_flow/ \
    --output judge_report_flow.md
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--guidance-scale` | 3.0 | CFG strength during sampling |
| `--steps` | 50 | Number of ODE solver steps |
| `--eval-interval` | 10 | Interference metric logging frequency |
| `--lr` | 1e-4 | Learning rate for stage-2 training |
| `--batch-size` | 64 | Training batch size |

## Observed vs OOD Pairs

- **Observed (training):** 23, 25, 35, 38, 58, 52, 82, 83
- **OOD (test):** 28, 32, 53, 85, 22, 33, 55, 88

The models never see OOD combinations during training. Generalization relies on disentangled left/right digit embeddings learned through structured label dropout.
