"""
src/detection/tracknet.py
──────────────────────────
TrackNetV2 architecture.

Key idea
────────
Instead of detecting the ball in ONE frame (like YOLO),
TrackNet stacks 3 consecutive BGR frames → 9-channel input tensor.
The network learns to detect the ball from its MOTION TRAIL across
3 frames — so motion blur is a feature, not a problem.

Architecture: VGG16-style encoder  +  U-Net decoder
  Input  : (B, 9, H, W)   — 3 frames × 3 channels
  Output : (B, 1, H, W)   — Gaussian heatmap, values 0..1

The peak of the heatmap = ball centre (sub-pixel accuracy via argmax
on the upsampled map, or weighted centroid).

Reference: TrackNetV2 — https://arxiv.org/abs/2007.13872
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pad: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class TrackNet(nn.Module):
    """
    TrackNetV2 — 3-frame spatiotemporal ball detector.

    Parameters
    ----------
    in_frames : int   Number of stacked input frames (default 3 → 9 channels)
    out_channels : int  Output channels (1 = single Gaussian heatmap)
    """

    def __init__(self, in_frames: int = 3, out_channels: int = 1):
        super().__init__()
        in_ch = in_frames * 3   # 3 frames × RGB

        # ── Encoder (VGG-style, with skip connections for U-Net) ──────────
        # Block 1
        self.enc1 = nn.Sequential(
            ConvBNReLU(in_ch, 64),
            ConvBNReLU(64, 64),
        )
        self.pool1 = nn.MaxPool2d(2, 2)   # /2

        # Block 2
        self.enc2 = nn.Sequential(
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
        )
        self.pool2 = nn.MaxPool2d(2, 2)   # /4

        # Block 3
        self.enc3 = nn.Sequential(
            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
            ConvBNReLU(256, 256),
        )
        self.pool3 = nn.MaxPool2d(2, 2)   # /8

        # Block 4
        self.enc4 = nn.Sequential(
            ConvBNReLU(256, 512),
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512),
        )
        self.pool4 = nn.MaxPool2d(2, 2)   # /16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512),
        )

        # ── Decoder (U-Net upsampling with skip connections) ──────────────
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            ConvBNReLU(512 + 512, 512),
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 256),
        )

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBNReLU(256 + 256, 256),
            ConvBNReLU(256, 256),
            ConvBNReLU(256, 128),
        )

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBNReLU(128 + 128, 128),
            ConvBNReLU(128, 64),
        )

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBNReLU(64 + 64, 64),
            ConvBNReLU(64, 64),
        )

        # ── Output head ───────────────────────────────────────────────────
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)
        # Sigmoid → output in [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 9, H, W)   stacked 3-frame input, values normalised 0..1

        Returns
        -------
        (B, 1, H, W)  heatmap, values 0..1
        """
        # Encode
        e1 = self.enc1(x)           # (B,  64, H,    W   )
        e2 = self.enc2(self.pool1(e1))  # (B, 128, H/2,  W/2 )
        e3 = self.enc3(self.pool2(e2))  # (B, 256, H/4,  W/4 )
        e4 = self.enc4(self.pool3(e3))  # (B, 512, H/8,  W/8 )
        bt = self.bottleneck(self.pool4(e4))  # (B, 512, H/16, W/16)

        # Decode with skip connections
        d4 = self.dec4(torch.cat([self.up4(bt), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))   # (B, 1, H, W)

    # ── Convenience ───────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"TrackNet(in_channels=9, params={self.count_parameters()/1e6:.1f}M)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Heatmap utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_heatmap(
    cx: float,
    cy: float,
    H: int,
    W: int,
    sigma: float = 5.0,
) -> torch.Tensor:
    """
    Generate a 2D Gaussian heatmap centred at (cx, cy).

    Used during training to create target maps from (cx, cy) annotations.
    The Gaussian is clamped to [0, 1].

    Parameters
    ----------
    cx, cy  : ball centre in PIXEL coordinates (float, can be sub-pixel)
    H, W    : output map height and width  (should match model input size)
    sigma   : Gaussian spread in pixels (default 5.0 — adjust to ball size)

    Returns
    -------
    torch.Tensor  shape (1, H, W), dtype float32, values 0..1
    """
    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    heatmap = torch.exp(
        -((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma ** 2)
    )
    return heatmap.unsqueeze(0)   # (1, H, W)


def heatmap_to_point(
    heatmap: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float] | None:
    """
    Extract ball (cx, cy) from a predicted heatmap.

    Uses weighted centroid (more precise than argmax) if the peak
    exceeds `threshold`. Returns None if no ball detected.

    Parameters
    ----------
    heatmap   : (1, H, W) or (H, W) tensor, values 0..1
    threshold : minimum peak value to consider a detection valid

    Returns
    -------
    (cx, cy) in pixel coordinates, or None
    """
    if heatmap.dim() == 3:
        heatmap = heatmap.squeeze(0)   # → (H, W)

    peak_val = float(heatmap.max())
    if peak_val < threshold:
        return None

    # Mask: only pixels above 50% of peak
    mask = (heatmap >= peak_val * 0.5).float()
    heatmap_masked = heatmap * mask

    total = heatmap_masked.sum().clamp(min=1e-8)
    H, W = heatmap.shape

    ys = torch.arange(H, dtype=torch.float32, device=heatmap.device)
    xs = torch.arange(W, dtype=torch.float32, device=heatmap.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    cx = float((grid_x * heatmap_masked).sum() / total)
    cy = float((grid_y * heatmap_masked).sum() / total)
    return (cx, cy)