# OOD Generalization as Bifurcation

## Workflow Rules
- **Auto-update docs**: After completing any task that changes the codebase (new files, modified APIs, new CLI args, etc.), update CLAUDE.md and README.md immediately without waiting for an explicit prompt.

## Project Overview
Comparing two independent generative models (CFDG diffusion vs CFM flow matching) on out-of-distribution bi-digit MNIST composition. Both use a shared U-Net backbone (`unet.py`) but with **separate weight instances**.

## Key Conventions

### Label System
- Digits 0-8 are valid digit classes. Label `0` represents a blank (black) image half.
- Label `9` is the **null token** used for classifier-free guidance dropout during training.
- Label dropout: 10% drop right, 10% drop left, 10% drop both, 70% keep both.

### Data Format
- Bi-digit images: 28x56x1 (two 28x28 digits side by side), stored as flattened 1D arrays (length 1568) in `.pkl` files.
- Labels: `[digit_left, digit_right]` as int pairs.
- Pixel range: raw 0-255, normalized to [-1, 1] for training.

### Latent Space
- VAE encodes 28x56x1 images to 7x14x2 latent tensors (deterministic `mu`, not sampled).
- All generative model training and sampling happens in this latent space.
- VAE is always frozen during stage-2 training.

### Training Targets
- **Observed pairs** (in-distribution): 23, 25, 35, 38, 58, 52, 82, 83
- **OOD pairs** (must generalize to): 28, 32, 53, 85, 22, 33, 55, 88

### Noise Schedules
- **Diffusion (CFDG)**: Linear schedule `alpha_bar(t) = 1 - t`, epsilon-prediction. Sampling: DDIM backward t=1→0.
- **Flow (CFM)**: OT path with `sigma_min = 1e-5`, velocity prediction. Sampling: forward Euler t=0→1.
- Both use 50 sampling steps and CFG guidance scale w=3.0 by default.

### Superposition Principle & Sampling Strategies
The project investigates the decomposition of CFG-guided scores using null token `∅` (label 9):
- **Joint guided**: `e_∅ + w * (e(L,R) - e_∅)` where `e_∅ = model(z, t, ∅, ∅)`
- **Decomposed guided**: `e_∅ + w * (e(L,∅) + e(∅,R) - 2·e_∅)`

A **semantic bifurcation window** exists mid-sampling where joint and decomposed guided scores diverge significantly.

Three sampling strategies in `mass_sampling.py`:
- **joint** (default): joint guided score at all steps (2 forward passes per step).
- **decomposed**: decomposed guided score at all steps (3 forward passes per step).
- **hybrid**: decomposed inside bifurcation window (`--bif-start`/`--bif-end`), joint outside.

### Sampling Output
- `mass_sampling.py` outputs 84x56 PNG images (3 rows of 28x56 stacked vertically).
- Filename format: `seed{NNNN}_{left}{right}.png` (e.g., `seed0042_28.png`).
- Compatible with `disentangled_judge.py --mode judge`.

## File Structure
| File | Purpose |
|------|---------|
| `unet.py` | Conditional U-Net backbone (shared architecture) |
| `vae.py` | Beta-VAE encoder/decoder |
| `utils.py` | Dataset loading, checkpointing, logging, interference metrics |
| `train_vae.py` | Stage 1: VAE pre-training |
| `train_diffusion.py` | Stage 2a: CFDG diffusion training |
| `train_flow.py` | Stage 2b: CFM flow matching training |
| `mass_sampling.py` | Generate images from trained models (joint/decomposed/hybrid) |
| `visualize.py` | Visualization: VAE recon, interference plots, progress grids, score heatmaps |
| `disentangled_judge.py` | Evaluate generated samples (train classifier / judge) |
| `create_bidgit_data.py` | Create bi-digit datasets from single-digit MNIST |
| `inspect_bi_digit.py` | Interactive dataset inspection |

## `visualize.py` Subcommands
| Subcommand | Purpose |
|------------|---------|
| `recon` | VAE reconstruction on random samples from a .pkl dataset |
| `interference` | MSE + Cosine Similarity between joint and sum scores across sampling steps |
| `progress` | 5-row (milestone steps) x 5-col (joint, sum, e(L,0), e(0,R), e(0,0)) decoded image grid |
| `heatmap` | Same grid layout but showing L2 norm of score in latent space (7x14, inferno colormap) |

## Virtual Environment
Activate with: `source ../venv/bin/activate`

## Checkpoints
- `vae_checkpoint.npz`: Pre-trained VAE (must exist before stage-2 training)
- `diffusion_checkpoint.npz`: CFDG model weights + optimizer state
- `cfm_checkpoint.npz`: CFM model weights + optimizer state
- `judge_checkpoint.npz`: Digit classifier for evaluation
