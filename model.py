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


class FeatureChangeConditioner(nn.Module):
    """Residual feature differences; optionally rates per 30-day time unit."""

    def __init__(self, channels: int, mode: str, time_scale_days: float) -> None:
        super().__init__()
        if mode not in ("change", "rate") or time_scale_days <= 0:
            raise ValueError("Feature change requires change/rate mode and a positive time scale")
        self.mode = mode
        self.time_scale_days = time_scale_days
        self.projection = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, features: torch.Tensor, delta_days: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = features.shape
        if delta_days.shape != (batch, steps):
            raise ValueError("Feature-change time gaps do not match the sequence")
        gaps = delta_days[:, 1:]
        if not torch.isfinite(gaps).all() or (gaps <= 0).any():
            raise ValueError("Observed feature-change gaps must be finite and positive")
        changes = features[:, 1:] - features[:, :-1]
        if self.mode == "rate":
            # Acquisitions have day precision; the first frame has no derivative.
            duration = gaps.clamp_min(1.0) / self.time_scale_days
            changes = changes / duration[:, :, None, None, None]
        changes = torch.cat((torch.zeros_like(features[:, :1]), changes), dim=1)
        residual = self.projection(changes.reshape(batch * steps, channels, height, width))
        return features + residual.reshape_as(features)


class CfCMaskForecaster(nn.Module):
    """Past RGB images -> spatial CfC -> future mask and auxiliary SDF."""

    def __init__(self, base_channels: int = 16, time_scale_days: float = 30.0,
                 *, temporal_kind: str = "cfc", use_actual_time: bool = True,
                 use_calendar: bool | None = None) -> None:
        super().__init__()
        if time_scale_days <= 0:
            raise ValueError("time_scale_days must be positive")
        self.time_scale_days = time_scale_days
        self.use_actual_time = use_actual_time
        self.use_calendar = use_actual_time if use_calendar is None else use_calendar
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

    def _dynamic_inputs(self, features: torch.Tensor, delta_days: torch.Tensor) -> torch.Tensor:
        return features

    def _evolve(self, dynamic: torch.Tensor, delta_days: torch.Tensor,
                horizon_days: torch.Tensor, context_season: torch.Tensor,
                target_season: torch.Tensor, *, cell: nn.Module,
                event_embedding: nn.Module, future_query: torch.Tensor,
                start_index: int = 0) -> torch.Tensor:
        batch, steps = dynamic.shape[:2]
        hidden = torch.zeros_like(dynamic[:, 0])
        for index in range(start_index, steps):
            event = torch.cat((context_season[:, index], torch.ones_like(horizon_days).reshape(batch, 1)), dim=1)
            encoded_event = event_embedding(event).unsqueeze(-1).unsqueeze(-1)
            hidden = cell(dynamic[:, index] + encoded_event, hidden,
                          delta_days[:, index] / self.time_scale_days)
        query_event = torch.cat((target_season, torch.zeros_like(horizon_days).reshape(batch, 1)), dim=1)
        query = future_query + event_embedding(query_event).unsqueeze(-1).unsqueeze(-1)
        return cell(query.expand_as(hidden), hidden, horizon_days / self.time_scale_days)

    def _extra_dynamics(self, features: torch.Tensor, delta_days: torch.Tensor,
                        horizon_days: torch.Tensor, context_season: torch.Tensor,
                        target_season: torch.Tensor) -> torch.Tensor | None:
        return None

    def _fuse_extra(self, bottleneck: torch.Tensor, skips: list[torch.Tensor],
                    extra: torch.Tensor | None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return bottleneck, skips

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
        if not self.use_calendar:
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
        dynamic = self._dynamic_inputs(dynamic, delta_days)
        # The query contains only the known target date; no future image is supplied.
        hidden = self._evolve(dynamic, delta_days, horizon_days, context_season, target_season,
                              cell=self.temporal, event_embedding=self.event_embedding,
                              future_query=self.future_query)
        fine_features = e4.reshape(batch, steps, *e4.shape[1:])
        extra = self._extra_dynamics(fine_features, delta_days, horizon_days, context_season, target_season)
        last = lambda features: features.reshape(batch, steps, *features.shape[1:])[:, -1]
        skips = [
            last(features) + projection(F.interpolate(hidden, size=features.shape[-2:], mode="bilinear", align_corners=False))
            for features, projection in zip((e1, e2, e3, e4), self.skip_conditions)
        ]
        x = self.future_bottleneck(torch.cat((last(bottleneck), hidden), dim=1))
        x, skips = self._fuse_extra(x, skips, extra)
        x = self.dec4(x, skips[3])
        x = self.dec3(x, skips[2])
        x = self.dec2(x, skips[1])
        x = self.dec1(x, skips[0])
        return self.segmentation_head(x), self.sdf_head(x)


class ArchitectureCfCForecaster(CfCMaskForecaster):
    """Controlled change/rate and two-resolution CfC hypotheses, not new SDF dynamics."""

    def __init__(self, base_channels: int = 16, time_scale_days: float = 30.0, *,
                 change_mode: str | None = None, multiscale: bool = False,
                 fine_history: bool = True) -> None:
        super().__init__(base_channels, time_scale_days)
        c, h, fine = base_channels, base_channels * 8, base_channels * 4
        self.change_conditioner = (FeatureChangeConditioner(h, change_mode, time_scale_days)
                                   if change_mode is not None else None)
        self.multiscale = multiscale
        self.fine_history = fine_history
        if multiscale:
            self.fine_projection = nn.Conv2d(c * 8, fine, 1)
            self.fine_temporal = SpatialCfCCell(fine)
            self.fine_event_embedding = nn.Sequential(nn.Linear(3, fine), nn.Tanh(), nn.Linear(fine, fine))
            self.fine_future_query = nn.Parameter(torch.zeros(1, fine, 1, 1))
            self.fine_change_conditioner = (FeatureChangeConditioner(fine, change_mode, time_scale_days)
                                            if change_mode is not None else None)
            self.fine_to_bottleneck = nn.Conv2d(fine, c * 16, 1)
            self.fine_to_skips = nn.ModuleList([nn.Conv2d(fine, width, 1) for width in (c, c * 2, c * 4, c * 8)])

    def _dynamic_inputs(self, features: torch.Tensor, delta_days: torch.Tensor) -> torch.Tensor:
        return self.change_conditioner(features, delta_days) if self.change_conditioner is not None else features

    def _extra_dynamics(self, features: torch.Tensor, delta_days: torch.Tensor,
                        horizon_days: torch.Tensor, context_season: torch.Tensor,
                        target_season: torch.Tensor) -> torch.Tensor | None:
        if not self.multiscale:
            return None
        batch, steps, channels, height, width = features.shape
        fine = self.fine_projection(features.reshape(batch * steps, channels, height, width))
        fine = fine.reshape(batch, steps, *fine.shape[1:])
        if self.fine_change_conditioner is not None:
            fine = self.fine_change_conditioner(fine, delta_days)
        start_index = 0 if self.fine_history else steps - 1
        if not self.fine_history:
            delta_days = delta_days.clone()
            delta_days[:, start_index] = 0
        return self._evolve(fine, delta_days, horizon_days, context_season, target_season,
                            cell=self.fine_temporal, event_embedding=self.fine_event_embedding,
                            future_query=self.fine_future_query, start_index=start_index)

    def _fuse_extra(self, bottleneck: torch.Tensor, skips: list[torch.Tensor],
                    extra: torch.Tensor | None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if extra is None:
            return bottleneck, skips
        bottleneck = bottleneck + self.fine_to_bottleneck(
            F.interpolate(extra, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False))
        skips = [skip + projection(F.interpolate(extra, size=skip.shape[-2:], mode="bilinear", align_corners=False))
                 for skip, projection in zip(skips, self.fine_to_skips)]
        return bottleneck, skips


class LastFrameForecaster(UNetWithSDFAux):
    def forward(self, context_images: torch.Tensor, delta_days: torch.Tensor,
                horizon_days: torch.Tensor, context_season: torch.Tensor,
                target_season: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(context_images[:, -1])


ARCHITECTURE_MODELS = {
    "spatial_cfc_change": {"change_mode": "change"},
    "spatial_cfc_rate": {"change_mode": "rate"},
    "spatial_cfc_multiscale": {"multiscale": True},
    "spatial_cfc_multiscale_rate": {"multiscale": True, "change_mode": "rate"},
    "spatial_cfc_multiscale_lastfine": {"multiscale": True, "fine_history": False},
}
MODEL_NAMES = ("segmenter_persistence", "last_frame", "convgru", "dt_convgru",
               "spatial_cfc_notime", "spatial_cfc", "spatial_cfc_nogap", "spatial_cfc_nocalendar",
               *ARCHITECTURE_MODELS)


def build_model(config: dict) -> nn.Module:
    name = config.get("model", "spatial_cfc")
    channels = int(config["base_channels"])
    if name in ("last_frame", "segmenter_persistence"):
        return LastFrameForecaster(base_channels=channels)
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    if name in ARCHITECTURE_MODELS:
        return ArchitectureCfCForecaster(channels, float(config.get("time_scale_days", 30.)),
                                         **ARCHITECTURE_MODELS[name])
    return CfCMaskForecaster(channels, float(config.get("time_scale_days", 30.)),
                            temporal_kind="convgru" if "convgru" in name else "cfc",
                            use_actual_time=name in ("dt_convgru", "spatial_cfc", "spatial_cfc_nocalendar"),
                            use_calendar=name in ("dt_convgru", "spatial_cfc", "spatial_cfc_nogap"))
