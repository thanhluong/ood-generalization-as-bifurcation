import mlx.core as mx
import mlx.nn as nn


def sinusoidal_embedding(t, dim):
    """Sinusoidal positional embedding for time step t in [0, 1].
    Args:
        t: (B,) time values
        dim: embedding dimension (must be even)
    Returns:
        (B, dim) embedding
    """
    half = dim // 2
    freqs = mx.exp(-mx.log(mx.array(10000.0)) * mx.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    return mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)


def upsample_nearest_2x(x):
    """Nearest-neighbor upsample by factor 2 (NHWC layout)."""
    B, H, W, C = x.shape
    x = x.reshape(B, H, 1, W, 1, C)
    x = mx.broadcast_to(x, (B, H, 2, W, 2, C))
    return x.reshape(B, H * 2, W * 2, C)


class CondResBlock(nn.Module):
    """Residual block with conditioning injection and GroupNorm."""

    def __init__(self, in_ch, out_ch, cond_dim, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.skip_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def __call__(self, x, cond):
        h = nn.silu(self.norm1(self.conv1(x)))
        # Inject conditioning: broadcast (B, cond_dim) -> (B, 1, 1, out_ch)
        h = h + self.cond_proj(cond).reshape(cond.shape[0], 1, 1, -1)
        h = nn.silu(self.norm2(self.conv2(h)))
        if self.skip_conv is not None:
            x = self.skip_conv(x)
        return x + h


class UNet(nn.Module):
    """Small U-Net for flow matching on 7x14xC latent space.

    Architecture (1-level downsample):
        Encoder: 7x14 -> 4x7
        Bottleneck: 4x7
        Decoder: 4x7 -> 7x14
    """

    def __init__(self, in_ch=2, base_ch=64, cond_dim=128,
                 num_digits=10, emb_dim=64):
        super().__init__()

        # --- Time embedding ---
        self.time_fc1 = nn.Linear(cond_dim, cond_dim)
        self.time_fc2 = nn.Linear(cond_dim, cond_dim)

        # --- Disentangled label embeddings ---
        self.emb_left = nn.Embedding(num_digits, emb_dim)
        self.emb_right = nn.Embedding(num_digits, emb_dim)
        self.label_proj = nn.Linear(emb_dim * 2, cond_dim)

        # --- U-Net layers ---
        # Input conv
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.norm_in = nn.GroupNorm(8, base_ch)

        # Down path (7x14 -> 4x7)
        self.down_res = CondResBlock(base_ch, base_ch, cond_dim)
        self.down_conv = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.down_norm = nn.GroupNorm(8, base_ch * 2)

        # Bottleneck (4x7)
        self.mid_res1 = CondResBlock(base_ch * 2, base_ch * 2, cond_dim)
        self.mid_res2 = CondResBlock(base_ch * 2, base_ch * 2, cond_dim)

        # Up path (4x7 -> 7x14)
        self.up_conv = nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1)
        self.up_norm = nn.GroupNorm(8, base_ch * 2)
        # After concat with skip: base_ch*2 + base_ch = base_ch*3
        self.up_res = CondResBlock(base_ch * 3, base_ch, cond_dim)

        # Output
        self.norm_out = nn.GroupNorm(8, base_ch)
        self.conv_out = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def __call__(self, x, t, left_digit, right_digit):
        """
        Args:
            x: (B, 7, 14, in_ch) noisy latent
            t: (B,) time step in [0, 1]
            left_digit: (B,) int left digit indices
            right_digit: (B,) int right digit indices
        Returns:
            (B, 7, 14, in_ch) predicted vector field
        """
        # Time embedding
        t_emb = sinusoidal_embedding(t, self.time_fc1.weight.shape[-1])
        t_emb = self.time_fc2(nn.silu(self.time_fc1(t_emb)))

        # Label embedding
        c_left = self.emb_left(left_digit)    # (B, emb_dim)
        c_right = self.emb_right(right_digit)  # (B, emb_dim)
        c_raw = mx.concatenate([c_left, c_right], axis=-1)  # (B, 2*emb_dim)
        c_emb = self.label_proj(c_raw)  # (B, cond_dim)

        # Combined condition
        cond = t_emb + c_emb  # (B, cond_dim)

        # Encoder
        h = nn.silu(self.norm_in(self.conv_in(x)))         # (B, 7, 14, base_ch)
        h_skip = self.down_res(h, cond)                    # (B, 7, 14, base_ch)
        h = nn.silu(self.down_norm(self.down_conv(h_skip)))  # (B, 4, 7, base_ch*2)

        # Bottleneck
        h = self.mid_res1(h, cond)
        h = self.mid_res2(h, cond)

        # Decoder
        h = upsample_nearest_2x(h)                         # (B, 8, 14, base_ch*2)
        h = h[:, :h_skip.shape[1], :h_skip.shape[2], :]    # crop to (B, 7, 14, ...)
        h = nn.silu(self.up_norm(self.up_conv(h)))          # (B, 7, 14, base_ch*2)
        h = mx.concatenate([h, h_skip], axis=-1)            # (B, 7, 14, base_ch*3)
        h = self.up_res(h, cond)                            # (B, 7, 14, base_ch)

        # Output
        h = self.conv_out(nn.silu(self.norm_out(h)))        # (B, 7, 14, in_ch)
        return h
