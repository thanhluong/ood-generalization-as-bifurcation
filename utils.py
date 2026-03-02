import json
import pickle
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def load_dataset(path="mnist_bi_digit.pkl"):
    """Load bi-digit dataset, reshape to (N, 28, 56, 1) and scale to [-1, 1]."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    images = data["images"].astype(np.float32)
    labels = data["labels"].astype(np.int32)
    images = images.reshape(-1, 28, 56, 1) / 127.5 - 1.0
    return images, labels


def save_checkpoint(path, model, optimizer, epoch):
    """Save model weights, optimizer state, and epoch to a single .npz file."""
    flat = {}
    for k, v in tree_flatten(model.trainable_parameters()):
        flat["model." + k] = np.array(v)
    for k, v in tree_flatten(optimizer.state):
        flat["opt." + k] = np.array(v)
    flat["epoch"] = np.array(epoch)
    np.savez(path, **flat)


def load_checkpoint(path, model, optimizer):
    """Load model weights, optimizer state, and epoch from .npz checkpoint.
    Returns the saved epoch number."""
    data = dict(np.load(path, allow_pickle=True))
    epoch = int(data["epoch"])
    model_w = []
    opt_s = []
    for k, v in data.items():
        if k.startswith("model."):
            model_w.append((k[6:], mx.array(v)))
        elif k.startswith("opt."):
            opt_s.append((k[4:], mx.array(v)))
    model.load_weights(model_w)
    if opt_s:
        optimizer.state = tree_unflatten(opt_s)
    return epoch


def init_log(path, resume=False):
    """Initialize a JSON-lines log file. Truncate if not resuming."""
    if not resume:
        open(path, "w").close()


def append_log(path, record):
    """Append a single JSON record (one line) to the log file."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_log(path):
    """Read all JSON records from a log file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_interference(model, z, t_val, left_digit, right_digit, null_label=0):
    """Compute superposition interference metrics.

    Tests v(z,t|left,right) ≈ v(z,t|left,0) + v(z,t|0,right) - v(z,t|0,0)

    Returns:
        (mse, cosine_similarity) as Python floats.
    """
    B = z.shape[0]
    t = mx.full((B,), t_val)
    left = mx.full((B,), left_digit, dtype=mx.int32)
    right = mx.full((B,), right_digit, dtype=mx.int32)
    null = mx.full((B,), null_label, dtype=mx.int32)

    v_comb = model(z, t, left, right)
    v_left = model(z, t, left, null)
    v_right = model(z, t, null, right)
    v_null_base = model(z, t, null, null)
    v_sum = v_left + v_right - v_null_base

    # MSE: ||v_comb - v_sum||²
    diff = v_comb - v_sum
    mse = mx.mean(diff * diff)

    # Cosine similarity (per sample, then average)
    vc = v_comb.reshape(B, -1)
    vs = v_sum.reshape(B, -1)
    dot = mx.sum(vc * vs, axis=-1)
    norm_c = mx.sqrt(mx.sum(vc * vc, axis=-1) + 1e-8)
    norm_s = mx.sqrt(mx.sum(vs * vs, axis=-1) + 1e-8)
    cos = mx.mean(dot / (norm_c * norm_s))

    mx.eval(mse, cos)
    return mse.item(), cos.item()


def get_batches(images, labels, batch_size, shuffle=True):
    """Yield (mx.array images, mx.array labels) batches."""
    n = len(images)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, n, batch_size):
        idx = indices[i : i + batch_size]
        yield mx.array(images[idx]), mx.array(labels[idx])
