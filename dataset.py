from __future__ import annotations

import csv
import math
import random
from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset


def resolve_asset(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


@dataclass(frozen=True)
class Sample:
    split: str
    reservoir: str
    acquisition_date: date
    image_path: Path
    mask_path: Path
    conversion_table: Path
    pixel_area_m2: float
    geometry_status: str
    level_evaluation: str


@dataclass(frozen=True)
class Window:
    reservoir: str
    context: tuple[Sample, ...]
    target: Sample

    @property
    def horizon_days(self) -> float:
        return float((self.target.acquisition_date - self.context[-1].acquisition_date).days)


def read_samples(path: str | Path, splits: set[str]) -> list[Sample]:
    manifest = Path(path).resolve()
    samples = []
    seen = set()
    assignments: dict[str, dict[str, list[date]]] = defaultdict(lambda: defaultdict(list))
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            reservoir, split = row["reservoir"], row["split"]
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unknown split: {split}")
            acquisition_date = date.fromisoformat(row["acquisition_date"])
            assignments[reservoir][split].append(acquisition_date)
            key = (reservoir, row["acquisition_date"])
            if key in seen:
                raise ValueError(f"Duplicate reservoir/date: {key}")
            seen.add(key)
            if split not in splits:
                continue
            if row["input_contract"] != "preprocessed_enhanced_rgb_v1":
                raise ValueError("This export requires preprocessed 3-channel RGB images")
            samples.append(Sample(
                split=split, reservoir=reservoir,
                acquisition_date=acquisition_date,
                image_path=resolve_asset(row["image_path"], manifest),
                mask_path=resolve_asset(row["mask_path"], manifest),
                conversion_table=resolve_asset(row["conversion_table"], manifest),
                pixel_area_m2=float(row["pixel_area_m2"]),
                geometry_status=row["geometry_status"],
                level_evaluation=row["level_evaluation"],
            ))
    for reservoir, dates in assignments.items():
        if "test" in dates and len(dates) > 1:
            raise ValueError(f"Test reservoir occurs in development splits: {reservoir}")
        if "train" in dates and "val" in dates and max(dates["train"]) >= min(dates["val"]):
            raise ValueError(f"Training dates must precede validation dates: {reservoir}")
    if not samples:
        raise ValueError(f"No samples for {sorted(splits)}")
    return samples


def make_windows(samples: list[Sample], context_length: int = 3, max_horizon_days: float | None = None) -> list[Window]:
    if context_length < 1:
        raise ValueError("context_length must be positive")
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.reservoir].append(sample)
    windows = []
    for reservoir, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: item.acquisition_date)
        for index in range(context_length, len(ordered)):
            window = Window(reservoir, tuple(ordered[index - context_length:index]), ordered[index])
            if window.horizon_days <= 0:
                raise ValueError("Forecast target must follow every context image")
            allowed_context = {"train", "val"} if window.target.split == "val" else {window.target.split}
            if any(item.split not in allowed_context for item in window.context):
                raise ValueError("Context crosses a forbidden split boundary")
            if max_horizon_days is None or window.horizon_days <= max_horizon_days:
                windows.append(window)
    return windows


def load_windows(path: str | Path, splits: set[str], context_length: int = 3,
                 max_horizon_days: float | None = None) -> list[Window]:
    # Validation may use already-observed training RGB, never validation labels for training.
    history_splits = set(splits) | ({"train"} if "val" in splits else set())
    samples = read_samples(path, history_splits)
    return [window for window in make_windows(samples, context_length, max_horizon_days)
            if window.target.split in splits]


def signed_distance(mask: np.ndarray, truncation_px: float) -> np.ndarray:
    inside = mask.astype(bool)
    sdf = distance_transform_edt(~inside) - distance_transform_edt(inside)
    return np.clip(sdf, -truncation_px, truncation_px).astype(np.float32) / truncation_px


def censored_sdf_valid(mask: np.ndarray, margin_px: int) -> np.ndarray:
    valid = np.ones(mask.shape, dtype=np.float32)
    margin = max(1, min(margin_px, min(mask.shape) // 4))
    if np.any(mask[0, :]):
        valid[:margin, :] = 0
    if np.any(mask[-1, :]):
        valid[-margin:, :] = 0
    if np.any(mask[:, 0]):
        valid[:, :margin] = 0
    if np.any(mask[:, -1]):
        valid[:, -margin:] = 0
    return valid


def geometry_for(width: int, height: int, size: int) -> dict[str, int | float]:
    scale = min(size / width, size / height)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    return dict(native_width=width, native_height=height, resized_width=rw,
                resized_height=rh, letterbox_left=(size - rw) // 2,
                letterbox_top=(size - rh) // 2, letterbox_scale=scale)


def letterbox_rgb(image: np.ndarray, size: int, geometry: dict) -> np.ndarray:
    height, width = image.shape[:2]
    if (width, height) != (geometry["native_width"], geometry["native_height"]):
        raise ValueError("Images within a reservoir must share a fixed native grid")
    rw, rh = int(geometry["resized_width"]), int(geometry["resized_height"])
    left, top = int(geometry["letterbox_left"]), int(geometry["letterbox_top"])
    resized = np.asarray(Image.fromarray(image).resize((rw, rh), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[top:top + rh, left:left + rw] = resized
    return canvas


def letterbox_field(array: np.ndarray, size: int, geometry: dict, nearest: bool = False) -> np.ndarray:
    rw, rh = int(geometry["resized_width"]), int(geometry["resized_height"])
    left, top = int(geometry["letterbox_left"]), int(geometry["letterbox_top"])
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    resized = np.asarray(Image.fromarray(array.astype(np.float32)).resize((rw, rh), method), dtype=np.float32)
    canvas = np.zeros((size, size), dtype=np.float32)
    canvas[top:top + rh, left:left + rw] = resized
    return canvas


def season(value: date) -> tuple[float, float]:
    phase = 2 * math.pi * (value.timetuple().tm_yday - 1) / 365.2425
    return math.sin(phase), math.cos(phase)


def context_timing(dates: list[date], target: date) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    deltas = [0.0] + [float((right - left).days) for left, right in zip(dates, dates[1:])]
    horizon = float((target - dates[-1]).days)
    if any(gap <= 0 for gap in deltas[1:]) or horizon <= 0:
        raise ValueError("Context dates must increase and target must be in the future")
    return (torch.tensor(deltas, dtype=torch.float32), torch.tensor(horizon),
            torch.tensor([season(day) for day in dates], dtype=torch.float32),
            torch.tensor(season(target), dtype=torch.float32))


@lru_cache(maxsize=128)
def cached_rgb(path: str, size: int, width: int, height: int) -> np.ndarray:
    with Image.open(path) as handle:
        rgb = np.asarray(handle.convert("RGB"))
    return letterbox_rgb(rgb, size, geometry_for(width, height, size))


@lru_cache(maxsize=128)
def cached_label(path: str, size: int, truncation_fraction: float, border_fraction: float) -> tuple:
    with Image.open(path) as handle:
        mask_native = (np.asarray(handle.convert("L")) >= 128).astype(np.float32)
    height, width = mask_native.shape
    geometry = geometry_for(width, height, size)
    truncation = max(1., truncation_fraction * math.hypot(width, height))
    margin = max(1, round(border_fraction * min(width, height)))
    mask = letterbox_field(mask_native, size, geometry, nearest=True)
    sdf = letterbox_field(signed_distance(mask_native, truncation), size, geometry)
    sdf_valid = letterbox_field(censored_sdf_valid(mask_native, margin), size, geometry, nearest=True)
    valid = letterbox_field(np.ones_like(mask_native), size, geometry, nearest=True)
    return mask, sdf, sdf_valid, valid, geometry, int(np.count_nonzero(mask_native))


class ReservoirSequenceDataset(Dataset):
    def __init__(self, split_csv: str | Path, splits: set[str], image_size: int = 512,
                 context_length: int = 3, max_horizon_days: float | None = None,
                 sdf_truncation_fraction: float = 0.05, sdf_border_margin_fraction: float = 0.01,
                 augment: bool = False, supervision: str = "future_mask") -> None:
        if not 0 < sdf_truncation_fraction <= 0.5:
            raise ValueError("Invalid SDF truncation fraction")
        if not 0 <= sdf_border_margin_fraction <= 0.25:
            raise ValueError("Invalid SDF border fraction")
        self.samples = load_windows(split_csv, splits, context_length, max_horizon_days)
        if not self.samples:
            raise ValueError("No forecasting windows remain for this split/settings")
        self.image_size = image_size
        self.sdf_truncation_fraction = sdf_truncation_fraction
        self.sdf_border_margin_fraction = sdf_border_margin_fraction
        self.augment = augment
        if supervision not in ("future_mask", "last_context_mask"):
            raise ValueError("Unknown supervision mode")
        self.supervision = supervision

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        window = self.samples[index]
        target = window.target
        label = window.context[-1] if self.supervision == "last_context_mask" else target
        # Labels are supervision only; no mask or future RGB enters the encoder.
        mask, sdf, sdf_valid, valid, geometry, water_pixels = cached_label(
            str(label.mask_path), self.image_size, self.sdf_truncation_fraction,
            self.sdf_border_margin_fraction)
        width, height = int(geometry["native_width"]), int(geometry["native_height"])
        images = [cached_rgb(str(sample.image_path), self.image_size, width, height)
                  for sample in window.context]
        context = np.stack(images)
        if self.augment:
            flip_x, flip_y, rotations = random.random() < .5, random.random() < .5, random.randrange(4)
            arrays = [mask, sdf, sdf_valid, valid]
            if flip_x:
                context = np.flip(context, axis=2)
                arrays = [np.flip(item, axis=1) for item in arrays]
            if flip_y:
                context = np.flip(context, axis=1)
                arrays = [np.flip(item, axis=0) for item in arrays]
            context = np.rot90(context, rotations, axes=(1, 2))
            mask, sdf, sdf_valid, valid = [np.rot90(item, rotations) for item in arrays]
        delta, horizon, context_season, target_season = context_timing(
            [sample.acquisition_date for sample in window.context], target.acquisition_date)
        tensor = lambda array: torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
        return {
            "context_images": torch.from_numpy(np.ascontiguousarray(context)).permute(0, 3, 1, 2),
            "delta_days": delta, "horizon_days": horizon,
            "context_season": context_season, "target_season": target_season,
            "mask": tensor(mask), "sdf": tensor(sdf),
            "valid": tensor(valid), "sdf_valid": tensor(sdf_valid),
            "reservoir": target.reservoir, "image_path": str(target.image_path),
            "target_mask_path": str(label.mask_path), "supervision": self.supervision,
            "conversion_table": str(target.conversion_table),
            "level_evaluation": target.level_evaluation,
            "geometry_status": target.geometry_status, "pixel_area_m2": target.pixel_area_m2,
            "target_area_km2": water_pixels * target.pixel_area_m2 / 1e6,
            "target_date": target.acquisition_date.isoformat(),
            "context_dates": ",".join(item.acquisition_date.isoformat() for item in window.context),
            **geometry,
        }
