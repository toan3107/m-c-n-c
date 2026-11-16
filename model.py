from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetWithSDFAux(nn.Module):
    """One shared U-Net decoder with segmentation and auxiliary SDF heads."""

    def __init__(self, base_channels: int = 24, in_channels: int = 3) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 16)
        self.dec4 = UpBlock(c * 16, c * 8, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)
        self.segmentation_head = nn.Conv2d(c, 1, 1)
        self.sdf_head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.dec4(x, e4)
        x = self.dec3(x, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        return self.segmentation_head(x), self.sdf_head(x)


class SpatialCfCCell(nn.Module):
    """Convolutional adaptation of the default CfC gated state update.

    Reference: Hasani et al., 2022, doi:10.1038/s42256-022-00556-7.
    The gate equation follows ncps/torch/cfc_cell.py; spatial convolutions
    replace dense feature transformations. This is not an exact ODE solver.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GroupNorm(group_count(channels), channels),
            nn.Tanh(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Tanh(),
        )
        self.branch1 = nn.Conv2d(channels, channels, 1)
        self.branch2 = nn.Conv2d(channels, channels, 1)
        self.time_a = nn.Conv2d(channels, channels, 1)
        self.time_b = nn.Conv2d(channels, channels, 1)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(elapsed).all() or (elapsed < 0).any():
            raise ValueError("CfC elapsed time must be finite and nonnegative")
        features = self.backbone(torch.cat((observation, hidden), dim=1))
        first = torch.tanh(self.branch1(features))
        second = torch.tanh(self.branch2(features))
        gate = torch.sigmoid(self.time_a(features) * elapsed.reshape(-1, 1, 1, 1) + self.time_b(features))
        return first * (1.0 - gate) + second * gate


class SpatialGRUCell(nn.Module):
    """ConvGRU; optional log1p elapsed-time channel, not the GRU-D algorithm."""

    def __init__(self, channels: int, time_aware: bool = False) -> None:
        super().__init__()
        self.time_aware = time_aware
        inputs = channels * 2 + int(time_aware)
        self.gates = nn.Conv2d(inputs, channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(inputs, channels, 3, padding=1)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(elapsed).all() or (elapsed < 0).any():
            raise ValueError("GRU elapsed time must be finite and nonnegative")
        time = torch.log1p(elapsed).reshape(-1, 1, 1, 1).expand(-1, 1, *hidden.shape[-2:])
        features = [observation, hidden] + ([time] if self.time_aware else [])
        reset, update = torch.sigmoid(self.gates(torch.cat(features, dim=1))).chunk(2, dim=1)
        candidate_inputs = [observation, reset * hidden] + ([time] if self.time_aware else [])
        candidate = torch.tanh(self.candidate(torch.cat(candidate_inputs, dim=1)))
        return (1.0 - update) * hidden + update * candidate


class CfCMaskForecaster(nn.Module):
    """Past RGB images -> spatial CfC -> future mask and auxiliary SDF."""

    def __init__(self, base_channels: int = 16, time_scale_days: float = 30.0,
                 *, temporal_kind: str = "cfc", use_actual_time: bool = True) -> None:
        super().__init__()
        if time_scale_days <= 0:
            raise ValueError("time_scale_days must be positive")
        self.time_scale_days = time_scale_days
        self.use_actual_time = use_actual_time
        c, h = base_channels, base_channels * 8
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 16)
        self.to_dynamic = nn.Conv2d(c * 16, h, 1)
        self.event_embedding = nn.Sequential(nn.Linear(3, h), nn.Tanh(), nn.Linear(h, h))
        self.future_query = nn.Parameter(torch.zeros(1, h, 1, 1))
        self.temporal = (SpatialCfCCell(h) if temporal_kind == "cfc"
                         else SpatialGRUCell(h, time_aware=use_actual_time))
        self.future_bottleneck = ConvBlock(c * 16 + h, c * 16)
        self.skip_conditions = nn.ModuleList([nn.Conv2d(h, channels, 1) for channels in (c, c * 2, c * 4, c * 8)])
        self.dec4 = UpBlock(c * 16, c * 8, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)
        self.segmentation_head = nn.Conv2d(c, 1, 1)
        self.sdf_head = nn.Conv2d(c, 1, 1)

    def forward(self, context_images: torch.Tensor, delta_days: torch.Tensor,
                horizon_days: torch.Tensor, context_season: torch.Tensor,
                target_season: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context_images.ndim != 5 or context_images.shape[2] != 3:
            raise ValueError("Expected context images shaped B,T,3,H,W")
        batch, steps, channels, height, width = context_images.shape
        if height % 16 or width % 16:
            raise ValueError("Spatial dimensions must be divisible by 16")
        if delta_days.shape != (batch, steps) or context_season.shape != (batch, steps, 2):
            raise ValueError("Context timestamp tensors do not match image sequence")
        if not torch.isfinite(horizon_days).all() or (horizon_days <= 0).any():
            raise ValueError("Forecast horizon must be finite and positive")
        if not self.use_actual_time:
            delta_days = torch.full_like(delta_days, self.time_scale_days)
            delta_days[:, 0] = 0
            horizon_days = torch.full_like(horizon_days, self.time_scale_days)
            context_season = torch.zeros_like(context_season)
            target_season = torch.zeros_like(target_season)
        x = context_images.reshape(batch * steps, channels, height, width)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        dynamic = self.to_dynamic(bottleneck)
        dynamic = dynamic.reshape(batch, steps, *dynamic.shape[1:])
        hidden = torch.zeros_like(dynamic[:, 0])
        for index in range(steps):
            event = torch.cat((context_season[:, index], torch.ones_like(horizon_days).reshape(batch, 1)), dim=1)
            encoded_event = self.event_embedding(event).unsqueeze(-1).unsqueeze(-1)
            hidden = self.temporal(dynamic[:, index] + encoded_event, hidden,
                                   delta_days[:, index] / self.time_scale_days)
        # The query contains only the known target date; no future image is supplied.
        query_event = torch.cat((target_season, torch.zeros_like(horizon_days).reshape(batch, 1)), dim=1)
        query = self.future_query + self.event_embedding(query_event).unsqueeze(-1).unsqueeze(-1)
        hidden = self.temporal(query.expand_as(hidden), hidden, horizon_days / self.time_scale_days)
        last = lambda features: features.reshape(batch, steps, *features.shape[1:])[:, -1]
        skips = [
            last(features) + projection(F.interpolate(hidden, size=features.shape[-2:], mode="bilinear", align_corners=False))
            for features, projection in zip((e1, e2, e3, e4), self.skip_conditions)
        ]
        x = self.future_bottleneck(torch.cat((last(bottleneck), hidden), dim=1))
        x = self.dec4(x, skips[3])
        x = self.dec3(x, skips[2])
        x = self.dec2(x, skips[1])
        x = self.dec1(x, skips[0])
        return self.segmentation_head(x), self.sdf_head(x)


class LastFrameForecaster(UNetWithSDFAux):
    def forward(self, context_images: torch.Tensor, delta_days: torch.Tensor,
                horizon_days: torch.Tensor, context_season: torch.Tensor,
                target_season: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(context_images[:, -1])


MODEL_NAMES = ("segmenter_persistence", "last_frame", "convgru", "dt_convgru",
               "spatial_cfc_notime", "spatial_cfc")


def build_model(config: dict) -> nn.Module:
    name = config.get("model", "spatial_cfc")
    channels = int(config["base_channels"])
    if name in ("last_frame", "segmenter_persistence"):
        return LastFrameForecaster(base_channels=channels)
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    return CfCMaskForecaster(channels, float(config.get("time_scale_days", 30.)),
                            temporal_kind="convgru" if "convgru" in name else "cfc",
                            use_actual_time=name in ("dt_convgru", "spatial_cfc"))
