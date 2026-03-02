import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from vae import VAE
from utils import (load_dataset, save_checkpoint, load_checkpoint,
                    get_batches, init_log, append_log)


def vae_loss_fn(model, x, kl_weight):
    x_hat, mu, logvar = model(x)
    recon = mx.mean((x_hat - x) ** 2)
    kl = -0.5 * mx.mean(1 + logvar - mu * mu - mx.exp(logvar))
    total = recon + kl_weight * kl
    return total, (recon, kl)


def main():
    parser = argparse.ArgumentParser(description="Train VAE (stage 1)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="vae_checkpoint.npz")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="mnist_bi_digit.pkl")
    parser.add_argument("--log", type=str, default="vae_train_log.jsonl")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset...")
    images, labels = load_dataset(args.data)
    print(f"  {images.shape[0]} images, shape {images.shape[1:]}")

    model = VAE()
    optimizer = optim.AdamW(learning_rate=args.lr)

    start_epoch = 0
    if args.resume:
        try:
            start_epoch = load_checkpoint(args.checkpoint, model, optimizer)
            print(f"Resumed from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")

    loss_and_grad = nn.value_and_grad(model, vae_loss_fn)
    init_log(args.log, resume=args.resume)

    print(f"Training VAE for epochs {start_epoch}-{start_epoch + args.epochs - 1}")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for x_batch, _ in get_batches(images, labels, args.batch_size):
            (loss, (recon, kl)), grads = loss_and_grad(model, x_batch, args.kl_weight)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            n_batches += 1
            # print training progress in the epoch
            if n_batches % 10 == 0:
                print(f"batch {n_batches:4d} | loss {loss.item():.6f}")

        nb = max(n_batches, 1)
        avg_loss = epoch_loss / nb
        avg_recon = epoch_recon / nb
        avg_kl = epoch_kl / nb
        dt = time.time() - t0
        print(f"epoch {epoch:4d} | loss {avg_loss:.6f} "
              f"(recon {avg_recon:.6f}, kl {avg_kl:.6f}) | {dt:.1f}s")

        append_log(args.log, {
            "epoch": epoch, "loss": avg_loss,
            "recon": avg_recon, "kl": avg_kl, "time": round(dt, 2),
        })
        save_checkpoint(args.checkpoint, model, optimizer, epoch + 1)

    print("VAE training done.")


if __name__ == "__main__":
    main()
