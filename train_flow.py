import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from vae import VAE
from unet import UNet
from utils import (load_dataset, save_checkpoint, load_checkpoint,
                    get_batches, init_log, append_log, compute_interference)


SIGMA_MIN = 1e-5


def encode_dataset(vae, images, batch_size=256):
    """Encode all images to latent space using frozen VAE encoder (deterministic mu)."""
    latents = []
    n = images.shape[0]
    for i in range(0, n, batch_size):
        x = mx.array(images[i : i + batch_size])
        mu, _ = vae.encoder(x)
        mx.eval(mu)
        latents.append(np.array(mu))
    return np.concatenate(latents, axis=0)


def cfm_loss_fn(model, z_1, left, right):
    """Conditional Flow Matching loss with OT path.

    z_t = (1 - (1 - sigma_min)*t) * z_0 + t * z_1
    u_t = z_1 - (1 - sigma_min) * z_0
    Loss = MSE(v_theta(z_t, t, c), u_t)
    """
    B = z_1.shape[0]
    t = mx.random.uniform(shape=(B,))
    z_0 = mx.random.normal(z_1.shape)

    t_exp = t.reshape(B, 1, 1, 1)
    z_t = (1.0 - (1.0 - SIGMA_MIN) * t_exp) * z_0 + t_exp * z_1
    u_t = z_1 - (1.0 - SIGMA_MIN) * z_0

    v_t = model(z_t, t, left, right)
    return mx.mean((v_t - u_t) ** 2)


def main():
    parser = argparse.ArgumentParser(
        description="Train Conditional Flow Matching (stage 2)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="cfm_checkpoint.npz")
    parser.add_argument("--vae-checkpoint", type=str, default="vae_checkpoint.npz")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="mnist_bi_digit_diffusion_and_flow.pkl")
    parser.add_argument("--log", type=str, default="cfm_train_log.jsonl")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Compute interference metrics every N epochs (0=disable)")
    parser.add_argument("--eval-pairs", type=str, default="28,32,53,85",
                        help="Comma-separated digit pairs for interference evaluation")
    parser.add_argument("--eval-samples", type=int, default=16,
                        help="Number of noise samples for interference evaluation")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # Load VAE (frozen)
    print("Loading frozen VAE encoder...")
    vae = VAE()
    vae_data = dict(np.load(args.vae_checkpoint, allow_pickle=True))
    vae_weights = [(k[6:], mx.array(v))
                   for k, v in vae_data.items() if k.startswith("model.")]
    vae.load_weights(vae_weights)
    vae.freeze()

    # Load and encode dataset
    print("Loading dataset and encoding to latent space...")
    images, labels = load_dataset(args.data)
    latents = encode_dataset(vae, images)
    print(f"  Latents shape: {latents.shape}")

    # Create U-Net and optimizer
    model = UNet()
    optimizer = optim.AdamW(learning_rate=args.lr)

    start_epoch = 0
    if args.resume:
        try:
            start_epoch = load_checkpoint(args.checkpoint, model, optimizer)
            print(f"Resumed from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")

    loss_and_grad = nn.value_and_grad(model, cfm_loss_fn)
    init_log(args.log, resume=args.resume)

    # Prepare interference evaluation
    eval_pairs = ([p for p in args.eval_pairs.split(",") if len(p) == 2]
                  if args.eval_pairs else [])
    eval_ts = [0.1, 0.3, 0.5, 0.7, 0.9]
    eval_z = None
    if eval_pairs and args.eval_interval > 0:
        eval_z = mx.random.normal((args.eval_samples, *latents.shape[1:]))
        mx.eval(eval_z)
        print(f"  Interference eval every {args.eval_interval} epochs on pairs {eval_pairs}")

    print(f"Training CFM for epochs {start_epoch}-{start_epoch + args.epochs - 1}")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for z_batch, lbl_batch in get_batches(latents, labels, args.batch_size):
            left = lbl_batch[:, 0]
            right = lbl_batch[:, 1]

            # Disentangled label dropout
            B = left.shape[0]
            r = mx.random.uniform(shape=(B,))
            null_token = 9

            mask_drop_r = (r < 0.1)
            right = mx.where(mask_drop_r, mx.array(null_token), right)

            mask_drop_l = (r >= 0.1) & (r < 0.2)
            left = mx.where(mask_drop_l, mx.array(null_token), left)

            mask_drop_both = (r >= 0.2) & (r < 0.3)
            left = mx.where(mask_drop_both, mx.array(null_token), left)
            right = mx.where(mask_drop_both, mx.array(null_token), right)

            loss, grads = loss_and_grad(model, z_batch, left, right)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += loss.item()
            n_batches += 1
            if n_batches % 10 == 0:
                print(f"batch {n_batches:4d} | loss {loss.item():.6f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        dt = time.time() - t0

        log_record = {"epoch": epoch, "loss": avg_loss, "time": round(dt, 2)}

        if eval_z is not None and (epoch + 1) % args.eval_interval == 0:
            total_mse, total_cos, n_evals = 0.0, 0.0, 0
            for pair in eval_pairs:
                ld, rd = int(pair[0]), int(pair[1])
                for tv in eval_ts:
                    m, c = compute_interference(model, eval_z, tv, ld, rd)
                    total_mse += m
                    total_cos += c
                    n_evals += 1
            avg_mse = total_mse / n_evals
            avg_cos = total_cos / n_evals
            log_record["interference_mse"] = round(avg_mse, 8)
            log_record["interference_cos"] = round(avg_cos, 6)
            print(f"epoch {epoch:4d} | loss {avg_loss:.6f} | "
                  f"I_mse {avg_mse:.6f} I_cos {avg_cos:.4f} | {dt:.1f}s")
        else:
            print(f"epoch {epoch:4d} | loss {avg_loss:.6f} | {dt:.1f}s")

        append_log(args.log, log_record)
        save_checkpoint(args.checkpoint, model, optimizer, epoch + 1)

    print("CFM training done.")


if __name__ == "__main__":
    main()
