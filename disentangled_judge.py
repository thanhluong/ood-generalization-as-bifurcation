import argparse
import os
import time
import pickle
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from PIL import Image

from utils import save_checkpoint, load_checkpoint, init_log, append_log


class DigitCNN(nn.Module):
    """Small CNN classifier for single 28x28 digit images (10 classes)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(8, 64)

        self.fc = nn.Linear(64, 10)

    def __call__(self, x):
        # x: (B, 28, 28, 1)
        h = nn.silu(self.norm1(self.conv1(x)))      # (B, 28, 28, 32)
        h = nn.silu(self.norm2(self.conv2(h)))       # (B, 14, 14, 64)
        h = nn.silu(self.norm3(self.conv3(h)))       # (B, 7, 7, 64)
        h = mx.mean(h, axis=(1, 2))                 # (B, 64) global avg pool
        return self.fc(h)                            # (B, 10) logits


def cross_entropy_loss(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))


def compute_accuracy(model, images, labels, batch_size=256):
    """Compute accuracy over a dataset."""
    correct = 0
    total = 0
    n = len(images)
    for i in range(0, n, batch_size):
        x = mx.array(images[i:i + batch_size])
        y = labels[i:i + batch_size]
        logits = model(x)
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)
        correct += int(mx.sum(preds == mx.array(y)).item())
        total += len(y)
    return correct / max(total, 1)


def train_mode(args):
    """Train the digit classifier on single-digit MNIST data."""
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    images = data["images"].astype(np.float32)
    labels = data["labels"].astype(np.int32)

    # Reshape to (N, 28, 28, 1) and scale to [0, 1]
    images = images.reshape(-1, 28, 28, 1) / 255.0

    # 90/10 train/val split (shuffled)
    n = len(images)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(0.9 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_images, train_labels = images[train_idx], labels[train_idx]
    val_images, val_labels = images[val_idx], labels[val_idx]
    print(f"  Train: {len(train_images)}, Val: {len(val_images)}")

    model = DigitCNN()
    optimizer = optim.AdamW(learning_rate=args.lr)

    start_epoch = 0
    if args.resume:
        try:
            start_epoch = load_checkpoint(args.checkpoint, model, optimizer)
            print(f"Resumed from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")

    loss_and_grad = nn.value_and_grad(model, cross_entropy_loss)
    init_log(args.log, resume=args.resume)

    print(f"Training DigitCNN for epochs {start_epoch}-{start_epoch + args.epochs - 1}")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle training data
        perm = np.random.permutation(len(train_images))
        for i in range(0, len(train_images), args.batch_size):
            idx = perm[i:i + args.batch_size]
            x_batch = mx.array(train_images[idx])
            y_batch = mx.array(train_labels[idx])

            loss, grads = loss_and_grad(model, x_batch, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_acc = compute_accuracy(model, train_images, train_labels)
        val_loss_total = 0.0
        val_batches = 0
        for i in range(0, len(val_images), args.batch_size):
            x_val = mx.array(val_images[i:i + args.batch_size])
            y_val = mx.array(val_labels[i:i + args.batch_size])
            vl = cross_entropy_loss(model, x_val, y_val)
            mx.eval(vl)
            val_loss_total += vl.item()
            val_batches += 1
        avg_val_loss = val_loss_total / max(val_batches, 1)
        val_acc = compute_accuracy(model, val_images, val_labels)

        dt = time.time() - t0
        print(f"epoch {epoch:4d} | train_loss {avg_loss:.4f} train_acc {train_acc:.4f} "
              f"| val_loss {avg_val_loss:.4f} val_acc {val_acc:.4f} | {dt:.1f}s")

        append_log(args.log, {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(avg_val_loss, 6),
            "val_acc": round(val_acc, 4),
            "time": round(dt, 2),
        })
        save_checkpoint(args.checkpoint, model, optimizer, epoch + 1)

    print("Judge training done.")


def parse_label_from_filename(filename):
    """Extract digit pair from filename: last 2 chars before .png → (left, right).

    E.g. 'image_abcd_28.png' → (2, 8)
    """
    stem = os.path.splitext(filename)[0]
    if len(stem) < 2:
        return None
    pair_str = stem[-2:]
    if pair_str.isdigit():
        return int(pair_str[0]), int(pair_str[1])
    return None


def judge_mode(args):
    """Judge generated samples by classifying left/right halves independently."""
    mx.random.seed(args.seed)

    print("Loading judge model...")
    model = DigitCNN()
    # Load weights only (no optimizer needed)
    data = dict(np.load(args.checkpoint, allow_pickle=True))
    weights = [(k[6:], mx.array(v)) for k, v in data.items() if k.startswith("model.")]
    model.load_weights(weights)

    samples_dir = args.samples_dir
    if not os.path.isdir(samples_dir):
        print(f"Error: samples directory '{samples_dir}' not found.")
        return

    png_files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith(".png")])
    if not png_files:
        print(f"No .png files found in '{samples_dir}'.")
        return

    print(f"Found {len(png_files)} PNG files in '{samples_dir}'")

    results = []

    for fname in png_files:
        label = parse_label_from_filename(fname)
        if label is None:
            print(f"  Skipping {fname}: cannot parse label from filename")
            continue

        expected_left, expected_right = label
        filepath = os.path.join(samples_dir, fname)
        img = Image.open(filepath).convert("L")
        img_np = np.array(img).astype(np.float32) / 255.0  # [0, 1]

        h, w = img_np.shape

        # Split into 28x56 rows if stacked (e.g. 84x56 → 3 rows)
        if h % 28 != 0 or w != 56:
            print(f"  Skipping {fname}: unexpected dimensions {h}x{w}")
            continue

        n_rows = h // 28
        for row_idx in range(n_rows):
            row = img_np[row_idx * 28:(row_idx + 1) * 28, :]  # (28, 56)

            left_half = row[:, :28].reshape(1, 28, 28, 1)     # (1, 28, 28, 1)
            right_half = row[:, 28:].reshape(1, 28, 28, 1)

            left_logits = model(mx.array(left_half))
            right_logits = model(mx.array(right_half))
            pred_left = int(mx.argmax(left_logits, axis=-1).item())
            pred_right = int(mx.argmax(right_logits, axis=-1).item())

            match = (pred_left == expected_left) and (pred_right == expected_right)
            results.append({
                "file": fname,
                "row": row_idx,
                "expected_left": expected_left,
                "expected_right": expected_right,
                "pred_left": pred_left,
                "pred_right": pred_right,
                "left_correct": pred_left == expected_left,
                "right_correct": pred_right == expected_right,
                "pair_correct": match,
            })

    if not results:
        print("No valid samples to judge.")
        return

    # Aggregate metrics
    total = len(results)
    left_correct = sum(1 for r in results if r["left_correct"])
    right_correct = sum(1 for r in results if r["right_correct"])
    pair_correct = sum(1 for r in results if r["pair_correct"])

    left_acc = left_correct / total
    right_acc = right_correct / total
    pair_acc = pair_correct / total

    # Per-label breakdown
    label_stats = {}
    for r in results:
        key = f"{r['expected_left']}{r['expected_right']}"
        if key not in label_stats:
            label_stats[key] = {"total": 0, "left_ok": 0, "right_ok": 0, "pair_ok": 0}
        label_stats[key]["total"] += 1
        label_stats[key]["left_ok"] += int(r["left_correct"])
        label_stats[key]["right_ok"] += int(r["right_correct"])
        label_stats[key]["pair_ok"] += int(r["pair_correct"])

    # Confusion details: count (expected → predicted) for each position
    left_confusion = {}
    right_confusion = {}
    for r in results:
        if not r["left_correct"]:
            key = f"{r['expected_left']}→{r['pred_left']}"
            left_confusion[key] = left_confusion.get(key, 0) + 1
        if not r["right_correct"]:
            key = f"{r['expected_right']}→{r['pred_right']}"
            right_confusion[key] = right_confusion.get(key, 0) + 1

    # Build report
    lines = []
    lines.append("# Disentangled Judge Report")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total samples | {total} |")
    lines.append(f"| Left digit accuracy | {left_acc:.1%} ({left_correct}/{total}) |")
    lines.append(f"| Right digit accuracy | {right_acc:.1%} ({right_correct}/{total}) |")
    lines.append(f"| Pair accuracy (both correct) | {pair_acc:.1%} ({pair_correct}/{total}) |")
    lines.append("")

    lines.append("## Per-Label Breakdown")
    lines.append("")
    lines.append("| Label | Total | Left Acc | Right Acc | Pair Acc |")
    lines.append("|-------|-------|----------|-----------|----------|")
    for key in sorted(label_stats.keys()):
        s = label_stats[key]
        la = s["left_ok"] / s["total"]
        ra = s["right_ok"] / s["total"]
        pa = s["pair_ok"] / s["total"]
        lines.append(f"| {key} | {s['total']} | {la:.1%} | {ra:.1%} | {pa:.1%} |")
    lines.append("")

    if left_confusion or right_confusion:
        lines.append("## Confusion Details")
        lines.append("")
        if left_confusion:
            lines.append("**Left position errors:**")
            lines.append("")
            for k in sorted(left_confusion.keys()):
                lines.append(f"- {k}: {left_confusion[k]} times")
            lines.append("")
        if right_confusion:
            lines.append("**Right position errors:**")
            lines.append("")
            for k in sorted(right_confusion.keys()):
                lines.append(f"- {k}: {right_confusion[k]} times")
            lines.append("")

    lines.append("## Sample Details")
    lines.append("")
    lines.append("| File | Row | Expected | Predicted | Match |")
    lines.append("|------|-----|----------|-----------|-------|")
    for r in results:
        exp = f"{r['expected_left']}{r['expected_right']}"
        pred = f"{r['pred_left']}{r['pred_right']}"
        mark = "Y" if r["pair_correct"] else "N"
        lines.append(f"| {r['file']} | {r['row']} | {exp} | {pred} | {mark} |")
    lines.append("")

    report = "\n".join(lines)

    with open(args.output, "w") as f:
        f.write(report)
    print(f"\nReport saved to {args.output}")
    print(f"\nSummary: Left {left_acc:.1%} | Right {right_acc:.1%} | Pair {pair_acc:.1%} ({pair_correct}/{total})")


def main():
    parser = argparse.ArgumentParser(description="Disentangled judge for bi-digit samples")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "judge"],
                        help="Mode: train the classifier or judge generated samples")
    parser.add_argument("--data", type=str, default="mnist_single_digit.pkl",
                        help="Single-digit dataset for training")
    parser.add_argument("--samples-dir", type=str, default="generated_samples/",
                        help="Directory containing generated .png files (judge mode)")
    parser.add_argument("--checkpoint", type=str, default="judge_checkpoint.npz",
                        help="Judge model checkpoint path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--log", type=str, default="judge_train_log.jsonl",
                        help="Training log path")
    parser.add_argument("--output", type=str, default="judge_report.md",
                        help="Judge report output path")
    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    else:
        judge_mode(args)


if __name__ == "__main__":
    main()
