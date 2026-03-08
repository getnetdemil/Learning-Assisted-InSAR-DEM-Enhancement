"""
FiLM-conditioned U-Net for self-supervised InSAR interferogram enhancement.

Architecture:
- Encoder-decoder U-Net with skip connections
- Feature-wise Linear Modulation (FiLM) at every encoder and decoder level
- Input: complex interferogram (2ch: Re, Im) + optional coherence (1ch)
- Conditioning: geometry metadata vector [Δt, Δθ_inc, Δθ_graze, B_perp,
                                          mode_embed, look_embed, SNR_proxy] (dim=7)
- Output: denoised complex interferogram (2ch) + per-pixel log-variance (1ch)

FiLM modulation (Perez et al. 2018):
    y = γ(c) * x + β(c)
where γ, β are channel-wise scale/shift predicted from the conditioning vector c.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """Applies FiLM conditioning: y = γ(c) * x + β(c)."""

    def __init__(self, num_channels: int, metadata_dim: int):
        super().__init__()
        self.scale_shift = nn.Linear(metadata_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: (B, metadata_dim)  x: (B, C, H, W)
        params = self.scale_shift(c)             # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)     # each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * x + beta            # residual scale around 1


class FiLMDoubleConv(nn.Module):
    """Two conv-BN-ReLU layers with FiLM applied after the first BN."""

    def __init__(self, in_channels: int, out_channels: int, metadata_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.film = FiLMLayer(out_channels, metadata_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.film(self.bn1(self.conv1(x)), c), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class MetadataEncoder(nn.Module):
    """Encodes raw geometry metadata vector into a fixed embedding."""

    def __init__(self, metadata_dim: int = 7, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.net(metadata)


class FiLMUNet(nn.Module):
    """
    FiLM-conditioned U-Net for interferogram denoising with uncertainty.

    Args:
        in_channels:  Number of input image channels (default 2: Re, Im).
                      Set to 3 to also pass coherence as a third channel.
        metadata_dim: Dimension of the raw geometry conditioning vector (default 7).
        features:     Channel widths for each encoder level.
        embed_dim:    Hidden size of the metadata MLP encoder.

    Forward inputs:
        x        (B, in_channels, H, W) — complex interferogram channels
        metadata (B, metadata_dim)      — [Δt, Δθ_inc, Δθ_graze, B_perp,
                                           mode_embed, look_embed, SNR_proxy]

    Forward outputs:
        denoised (B, 2, H, W)  — denoised Re/Im interferogram
        log_var  (B, 1, H, W)  — per-pixel log-variance (aleatoric uncertainty)
    """

    def __init__(
        self,
        in_channels: int = 2,
        metadata_dim: int = 7,
        features: List[int] = None,
        embed_dim: int = 64,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.metadata_encoder = MetadataEncoder(metadata_dim, embed_dim)

        self.pool = nn.MaxPool2d(2, 2)
        self.downs: nn.ModuleList[FiLMDoubleConv] = nn.ModuleList()
        self.ups_t: nn.ModuleList[nn.ConvTranspose2d] = nn.ModuleList()
        self.ups_conv: nn.ModuleList[FiLMDoubleConv] = nn.ModuleList()

        # Encoder
        ch = in_channels
        for feat in features:
            self.downs.append(FiLMDoubleConv(ch, feat, embed_dim))
            ch = feat

        # Bottleneck
        self.bottleneck = FiLMDoubleConv(features[-1], features[-1] * 2, embed_dim)

        # Decoder
        rev_feats = list(reversed(features))
        up_ch = features[-1] * 2
        for feat in rev_feats:
            self.ups_t.append(nn.ConvTranspose2d(up_ch, feat, kernel_size=2, stride=2))
            self.ups_conv.append(FiLMDoubleConv(feat * 2, feat, embed_dim))
            up_ch = feat

        # Output heads
        self.head_denoised = nn.Conv2d(features[0], 2, kernel_size=1)
        self.head_log_var = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(
        self, x: torch.Tensor, metadata: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.metadata_encoder(metadata)  # (B, embed_dim)

        # Encoder
        skips = []
        for down in self.downs:
            x = down(x, c)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, c)
        skips = skips[::-1]

        # Decoder
        for i, (up_t, up_conv) in enumerate(zip(self.ups_t, self.ups_conv)):
            x = up_t(x)
            skip = skips[i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = up_conv(x, c)

        denoised = self.head_denoised(x)
        log_var = self.head_log_var(x)
        return denoised, log_var
