"""
UNet3++ — PyTorch Implementation
=================================
Full-scale skip connections from encoder to every decoder node and from
shallower decoder nodes to deeper ones.

Reference: Huang et al., "UNet 3+: A Full-Scale Connected UNet for Medical
Image Segmentation" (ICASSP 2020).

Architecture used here:
  Encoder depths  : [64, 128, 256, 512, 1024]   (5 levels incl. bottleneck)
  cat_channels    : 64   (each connection projected to 64 ch before concat)
  agg_channels    : 64*5 = 320  (5 connections per decoder node)
  Output          : sigmoid binary mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _proj(in_ch, cat_ch=64):
    """1×1 conv to project a feature map to cat_ch channels."""
    return nn.Sequential(
        nn.Conv2d(in_ch, cat_ch, 1, bias=False),
        nn.BatchNorm2d(cat_ch),
        nn.ReLU(inplace=True),
    )


class UNet3Plus(nn.Module):
    """
    UNet3++ with 5-level encoder and full-scale skip connections.

    Parameters
    ----------
    in_channels  : int   input image channels (default 1 for grayscale)
    out_channels : int   number of output classes  (default 1 binary)
    filters      : list  encoder filter counts per level
    cat_channels : int   channels each connection is projected to before concat
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        filters=(64, 128, 256, 512, 1024),
        cat_channels=64,
    ):
        super().__init__()
        f = filters
        C = cat_channels
        num_scales = len(f)          # 5
        agg = C * num_scales         # 320

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = _conv_block(in_channels, f[0])
        self.enc2 = _conv_block(f[0],  f[1])
        self.enc3 = _conv_block(f[1],  f[2])
        self.enc4 = _conv_block(f[2],  f[3])
        self.enc5 = _conv_block(f[3],  f[4])   # bottleneck
        self.pool = nn.MaxPool2d(2)

        # ── Decoder node d4: receives e1↓8, e2↓4, e3↓2, e4, e5↑2 ──────────
        self.d4_e1 = _proj(f[0], C)
        self.d4_e2 = _proj(f[1], C)
        self.d4_e3 = _proj(f[2], C)
        self.d4_e4 = _proj(f[3], C)
        self.d4_e5 = _proj(f[4], C)
        self.dec4  = _conv_block(agg, agg)

        # ── Decoder node d3: receives e1↓4, e2↓2, e3, d4↑2, e5↑4 ──────────
        self.d3_e1 = _proj(f[0], C)
        self.d3_e2 = _proj(f[1], C)
        self.d3_e3 = _proj(f[2], C)
        self.d3_d4 = _proj(agg,  C)
        self.d3_e5 = _proj(f[4], C)
        self.dec3  = _conv_block(agg, agg)

        # ── Decoder node d2: receives e1↓2, e2, d3↑2, d4↑4, e5↑8 ──────────
        self.d2_e1 = _proj(f[0], C)
        self.d2_e2 = _proj(f[1], C)
        self.d2_d3 = _proj(agg,  C)
        self.d2_d4 = _proj(agg,  C)
        self.d2_e5 = _proj(f[4], C)
        self.dec2  = _conv_block(agg, agg)

        # ── Decoder node d1: receives e1, d2↑2, d3↑4, d4↑8, e5↑16 ─────────
        self.d1_e1 = _proj(f[0], C)
        self.d1_d2 = _proj(agg,  C)
        self.d1_d3 = _proj(agg,  C)
        self.d1_d4 = _proj(agg,  C)
        self.d1_e5 = _proj(f[4], C)
        self.dec1  = _conv_block(agg, agg)

        # ── Output ────────────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(agg, out_channels, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _up(x, ref):
        """Bilinear upsample x to match ref spatial dims."""
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _down(x, ref):
        """Adaptive avg pool x down to match ref spatial dims."""
        return F.adaptive_avg_pool2d(x, ref.shape[2:])

    def forward(self, x):
        # ── Encoder ──────────────────────────────────────────────────────────
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # ── d4 (spatial size = e4) ───────────────────────────────────────────
        d4 = torch.cat([
            self.d4_e1(self._down(e1, e4)),
            self.d4_e2(self._down(e2, e4)),
            self.d4_e3(self._down(e3, e4)),
            self.d4_e4(e4),
            self.d4_e5(self._up(e5,  e4)),
        ], dim=1)
        d4 = self.dec4(d4)

        # ── d3 (spatial size = e3) ───────────────────────────────────────────
        d3 = torch.cat([
            self.d3_e1(self._down(e1, e3)),
            self.d3_e2(self._down(e2, e3)),
            self.d3_e3(e3),
            self.d3_d4(self._up(d4, e3)),
            self.d3_e5(self._up(e5, e3)),
        ], dim=1)
        d3 = self.dec3(d3)

        # ── d2 (spatial size = e2) ───────────────────────────────────────────
        d2 = torch.cat([
            self.d2_e1(self._down(e1, e2)),
            self.d2_e2(e2),
            self.d2_d3(self._up(d3, e2)),
            self.d2_d4(self._up(d4, e2)),
            self.d2_e5(self._up(e5, e2)),
        ], dim=1)
        d2 = self.dec2(d2)

        # ── d1 (spatial size = e1) ───────────────────────────────────────────
        d1 = torch.cat([
            self.d1_e1(e1),
            self.d1_d2(self._up(d2, e1)),
            self.d1_d3(self._up(d3, e1)),
            self.d1_d4(self._up(d4, e1)),
            self.d1_e5(self._up(e5, e1)),
        ], dim=1)
        d1 = self.dec1(d1)

        # ── Output ────────────────────────────────────────────────────────────
        out = self.out_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(out)
