import mlx.core as mx
import mlx.nn as nn


def upsample_nearest_2x(x):
    """Nearest-neighbor upsample by factor 2 (NHWC layout)."""
    B, H, W, C = x.shape
    x = x.reshape(B, H, 1, W, 1, C)
    x = mx.broadcast_to(x, (B, H, 2, W, 2, C))
    return x.reshape(B, H * 2, W * 2, C)


class ResBlock(nn.Module):
    """Residual block with GroupNorm for stability (no conditioning)."""

    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels)

    def __call__(self, x):
        h = nn.silu(self.norm1(self.conv1(x)))
        h = nn.silu(self.norm2(self.conv2(h)))
        return x + h


class Encoder(nn.Module):
    """Encode 28x56x1 images to 7x14x(2*latent_ch) for mu and logvar."""

    def __init__(self, in_ch=1, latent_ch=2, base_ch=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.norm_in = nn.GroupNorm(8, base_ch)

        self.conv_d1 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        self.norm_d1 = nn.GroupNorm(8, base_ch)

        self.conv_d2 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.norm_d2 = nn.GroupNorm(8, base_ch * 2)

        self.mid1 = ResBlock(base_ch * 2)
        self.mid2 = ResBlock(base_ch * 2)

        self.norm_out = nn.GroupNorm(8, base_ch * 2)
        self.conv_out = nn.Conv2d(base_ch * 2, latent_ch * 2, 3, padding=1)

    def __call__(self, x):
        h = nn.silu(self.norm_in(self.conv_in(x)))
        h = nn.silu(self.norm_d1(self.conv_d1(h)))   # 14x28
        h = nn.silu(self.norm_d2(self.conv_d2(h)))   # 7x14
        h = self.mid1(h)
        h = self.mid2(h)
        h = nn.silu(self.norm_out(h))
        h = self.conv_out(h)
        mu, logvar = h[..., :h.shape[-1] // 2], h[..., h.shape[-1] // 2:]
        return mu, logvar


class Decoder(nn.Module):
    """Decode 7x14xlatent_ch latents back to 28x56x1 images."""

    def __init__(self, out_ch=1, latent_ch=2, base_ch=64):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_ch, base_ch * 2, 3, padding=1)
        self.norm_in = nn.GroupNorm(8, base_ch * 2)

        self.mid1 = ResBlock(base_ch * 2)
        self.mid2 = ResBlock(base_ch * 2)

        # After first upsample (7x14 -> 14x28)
        self.conv_u1 = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        self.norm_u1 = nn.GroupNorm(8, base_ch)

        # After second upsample (14x28 -> 28x56)
        self.conv_u2 = nn.Conv2d(base_ch, base_ch, 3, padding=1)
        self.norm_u2 = nn.GroupNorm(8, base_ch)

        self.norm_out = nn.GroupNorm(8, base_ch)
        self.conv_out = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def __call__(self, z):
        h = nn.silu(self.norm_in(self.conv_in(z)))
        h = self.mid1(h)
        h = self.mid2(h)
        h = upsample_nearest_2x(h)                    # 14x28
        h = nn.silu(self.norm_u1(self.conv_u1(h)))
        h = upsample_nearest_2x(h)                    # 28x56
        h = nn.silu(self.norm_u2(self.conv_u2(h)))
        h = self.conv_out(nn.silu(self.norm_out(h)))
        return h


class VAE(nn.Module):
    def __init__(self, in_ch=1, latent_ch=2, base_ch=64):
        super().__init__()
        self.encoder = Encoder(in_ch, latent_ch, base_ch)
        self.decoder = Decoder(in_ch, latent_ch, base_ch)

    def reparameterize(self, mu, logvar):
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mu.shape)
        return mu + eps * std

    def __call__(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        return self.decoder(z)
