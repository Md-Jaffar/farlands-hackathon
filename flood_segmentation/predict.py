from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds


class FloodSegmentationError(ValueError):
    """Custom exception for invalid or corrupt image inputs when strict mode is used."""
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _resolve_path(path_str: str | Path) -> Path:
    """Resolve path directly or relative to predict.py directory (Rule 3)."""
    p = Path(path_str)
    if p.exists() or p.is_absolute():
        return p
    alt = Path(__file__).resolve().parent / p
    return alt if alt.exists() else p


def synthesize_sar_three_channel(vv: np.ndarray, vh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Take 2-band Sentinel-1 (VV, VH) in dB scale, synthesize the 3rd channel (VV - VH),
    and return:
      1. stacked_raw: (3, H, W) float32 array [VV, VH, VV - VH]
      2. rgb_uint8: (H, W, 3) uint8 array scaled to [0, 255] for visual previews
    """
    valid_mask = np.isfinite(vv) & np.isfinite(vh)
    vv_clean = np.where(valid_mask, vv, -25.0).astype(np.float32)
    vh_clean = np.where(valid_mask, vh, -32.0).astype(np.float32)
    diff_clean = (vv_clean - vh_clean).astype(np.float32)

    stacked_raw = np.stack([vv_clean, vh_clean, diff_clean], axis=0)

    # Scale to 0..255 using standard Sentinel-1 dB dynamic ranges for false-color visualization
    vv_norm = np.clip((vv_clean - (-25.0)) / (0.0 - (-25.0)), 0.0, 1.0)
    vh_norm = np.clip((vh_clean - (-32.0)) / (-5.0 - (-32.0)), 0.0, 1.0)
    diff_norm = np.clip((diff_clean - 0.0) / 15.0, 0.0, 1.0)

    rgb_uint8 = (np.stack([vv_norm, vh_norm, diff_norm], axis=-1) * 255.0).astype(np.uint8)
    rgb_uint8[~valid_mask] = 0
    return stacked_raw, rgb_uint8


def load_model(model_dir: str | Path | None = None, device: str = "cpu") -> dict[str, Any]:
    """
    Load the fine-tuned SegFormer checkpoint from model/best_model/ if present;
    otherwise return the Milestone M0 physics-guided SAR backscatter threshold model.
    Never executes at import time (Rule 4).
    """
    base_dir = Path(__file__).resolve().parent
    cfg_path = base_dir / "configs" / "default_config.json"
    config: dict[str, Any] = {
        "vv_water_threshold_db": -16.5,
        "vh_water_threshold_db": -23.0,
        "confidence_threshold": 0.5,
    }
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                config.update(json.load(f))
        except Exception:
            pass

    target_dir = _resolve_path(model_dir) if model_dir else (base_dir / "model" / "best_model")
    weights_file_bin = target_dir / "pytorch_model.bin"
    weights_file_safe = target_dir / "model.safetensors"

    if target_dir.exists() and (weights_file_bin.exists() or weights_file_safe.exists()):
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation

            net = SegformerForSemanticSegmentation.from_pretrained(
                str(target_dir), local_files_only=True
            )
            net.to(device)
            net.eval()
            return {
                "backend": "segformer",
                "model": net,
                "device": device,
                "model_path": str(target_dir),
                "config": config,
            }
        except Exception:
            pass

    return {
        "backend": "sar_threshold_m0",
        "model": None,
        "device": "cpu",
        "model_path": "baseline_sar_threshold",
        "config": config,
    }


def _compute_pixel_area_m2(crs: Any, transform: Affine, height: int, width: int) -> float | None:
    """Calculate the real-world area in m^2 of a single pixel when georeferenced."""
    if crs is None or transform is None or transform == Affine.identity():
        return None
    try:
        if crs.is_geographic:
            # Convert bounding box in degrees (e.g., EPSG:4326) to meters using local latitude
            west, south, east, north = rasterio.transform.array_bounds(height, width, transform)
            mid_lat_rad = math.radians((south + north) / 2.0)
            meters_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * mid_lat_rad)
            meters_per_deg_lon = 111412.84 * math.cos(mid_lat_rad)
            pixel_w_m = abs(transform.a) * meters_per_deg_lon
            pixel_h_m = abs(transform.e) * meters_per_deg_lat
            return float(pixel_w_m * pixel_h_m)
        else:
            # Projected metric CRS (e.g., UTM)
            return float(abs(transform.a * transform.e))
    except Exception:
        return None


def analyze_flood_image(
    image_path: str,
    output_dir: str,
    model_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Analyze a Sentinel-1 SAR GeoTIFF (or RGB image), write <stem>_mask.tif,
    <stem>_overlay.png, and <stem>_preview.png to output_dir, and return the exact
    contract dictionary.
    """
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    resolved_img = _resolve_path(image_path)
    stem = resolved_img.stem

    if not resolved_img.exists() or not resolved_img.is_file():
        return {
            "status": "ERROR",
            "error_code": "FILE_NOT_FOUND",
            "reason": f"Image file not found: '{image_path}'",
            "image_path": str(image_path),
            "flood_pixel_count": 0,
            "total_valid_pixels": 0,
            "flood_fraction": 0.0,
            "confidence": 0.0,
            "georeferenced": False,
            "crs": None,
            "bounds": None,
            "real_world_area_m2": None,
            "mask_path": None,
            "overlay_path": None,
            "preview_path": None,
            "provenance": "MODEL_DERIVED",
        }

    if model_bundle is None:
        model_bundle = load_model()

    # Read image via rasterio (handles GeoTIFF and standard images)
    try:
        with rasterio.open(resolved_img) as src:
            data = src.read()  # (C, H, W)
            crs = src.crs
            transform = src.transform
            nodata = src.nodata
            bounds = list(src.bounds) if crs is not None else None
    except Exception as exc:
        return {
            "status": "ERROR",
            "error_code": "CORRUPT_IMAGE",
            "reason": f"Corrupt or unreadable image '{image_path}': {exc}",
            "image_path": str(image_path),
            "flood_pixel_count": 0,
            "total_valid_pixels": 0,
            "flood_fraction": 0.0,
            "confidence": 0.0,
            "georeferenced": False,
            "crs": None,
            "bounds": None,
            "real_world_area_m2": None,
            "mask_path": None,
            "overlay_path": None,
            "preview_path": None,
            "provenance": "MODEL_DERIVED",
        }

    if data.ndim != 3 or data.shape[0] < 1 or data.shape[1] == 0 or data.shape[2] == 0:
        return {
            "status": "ERROR",
            "error_code": "CORRUPT_IMAGE",
            "reason": f"Invalid raster dimensions {data.shape} in '{image_path}'",
            "image_path": str(image_path),
            "flood_pixel_count": 0,
            "total_valid_pixels": 0,
            "flood_fraction": 0.0,
            "confidence": 0.0,
            "georeferenced": False,
            "crs": None,
            "bounds": None,
            "real_world_area_m2": None,
            "mask_path": None,
            "overlay_path": None,
            "preview_path": None,
            "provenance": "MODEL_DERIVED",
        }

    channels, height, width = data.shape
    is_georef = crs is not None and transform != Affine.identity()

    # Determine valid vs nodata pixels (ignore_index = -1)
    raw_float = data.astype(np.float32)
    valid_mask = np.isfinite(raw_float).all(axis=0)
    if nodata is not None and not math.isnan(nodata):
        valid_mask &= ~(raw_float == float(nodata)).any(axis=0)

    if channels >= 2 and (raw_float[:2].min() < 0.0 or raw_float[:2].max() <= 5.0):
        # 2-channel Sentinel-1 SAR (VV, VH) in dB scale
        vv = raw_float[0]
        vh = raw_float[1]
        stacked_3ch, preview_rgb = synthesize_sar_three_channel(vv, vh)
    elif channels >= 3:
        # 3-channel RGB image fallback
        rgb = np.transpose(raw_float[:3], (1, 2, 0))
        if rgb.max() > 1.0:
            preview_rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            preview_rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        stacked_3ch = (np.transpose(preview_rgb.astype(np.float32), (2, 0, 1)) / 255.0)
    else:
        # Single-band grayscale/SAR fallback
        b0 = raw_float[0]
        stacked_3ch, preview_rgb = synthesize_sar_three_channel(b0, b0 - 6.0)

    # Inference: SegFormer if loaded, otherwise M0 physics-based SAR backscatter threshold
    cfg = model_bundle.get("config", {})
    if model_bundle.get("backend") == "segformer" and model_bundle.get("model") is not None:
        import torch
        import torch.nn.functional as F

        net = model_bundle["model"]
        device = model_bundle.get("device", "cpu")
        # Standardize 3-channel tensor for SegFormer
        c_mean = np.array([-12.0, -19.0, 7.0], dtype=np.float32).reshape(3, 1, 1)
        c_std = np.array([5.0, 5.0, 4.0], dtype=np.float32).reshape(3, 1, 1)
        norm_tensor = (stacked_3ch - c_mean) / c_std
        inp = torch.from_numpy(norm_tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = net(pixel_values=inp)
            logits = F.interpolate(
                outputs.logits, size=(height, width), mode="bilinear", align_corners=False
            )
            probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
        flood_pred = (probs >= float(cfg.get("confidence_threshold", 0.5))) & valid_mask
        conf_vals = probs[flood_pred] if np.any(flood_pred) else (1.0 - probs[valid_mask])
        confidence = float(np.mean(conf_vals)) if conf_vals.size > 0 else 0.85
    else:
        # Milestone M0 SAR Threshold Baseline (or dark-water RGB fallback)
        if channels >= 2 and (raw_float[:2].min() < 0.0 or raw_float[:2].max() <= 5.0):
            vv_thresh = float(cfg.get("vv_water_threshold_db", -16.5))
            vh_thresh = float(cfg.get("vh_water_threshold_db", -23.0))
            flood_pred = ((stacked_3ch[0] < vv_thresh) | (stacked_3ch[1] < vh_thresh)) & valid_mask
            margin = np.clip((vv_thresh - stacked_3ch[0]) / 6.0, 0.55, 0.95)
            confidence = (
                float(np.mean(margin[flood_pred])) if np.any(flood_pred) else 0.82
            )
        else:
            # Simple pixel intensity threshold for standard RGB/grayscale
            brightness = preview_rgb.mean(axis=-1)
            flood_pred = (brightness < 70.0) & valid_mask
            confidence = 0.78

    # Construct output mask: 1 = flood, 0 = non-flood, -1 = nodata/ignore_index
    mask_arr = np.full((height, width), -1, dtype=np.int16)
    mask_arr[valid_mask] = 0
    mask_arr[flood_pred] = 1

    # 1. Save <stem>_mask.tif (preserving CRS and georeferencing transform if present)
    mask_path = out_dir_path / f"{stem}_mask.tif"
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "int16",
        "nodata": -1,
    }
    if is_georef:
        profile["crs"] = crs
        profile["transform"] = transform
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask_arr, 1)

    # 2. Save <stem>_preview.png
    preview_path = out_dir_path / f"{stem}_preview.png"
    Image.fromarray(preview_rgb, mode="RGB").save(preview_path)

    # 3. Save <stem>_overlay.png (blend cyan/blue flood highlight onto preview_rgb)
    overlay_rgb = preview_rgb.copy()
    if np.any(flood_pred):
        flood_color = np.array([0, 180, 255], dtype=np.float32)  # High-contrast cyan flood mask
        overlay_rgb[flood_pred] = (
            0.45 * overlay_rgb[flood_pred].astype(np.float32) + 0.55 * flood_color
        ).astype(np.uint8)
    overlay_path = out_dir_path / f"{stem}_overlay.png"
    Image.fromarray(overlay_rgb, mode="RGB").save(overlay_path)

    flood_pixels = int(np.sum(mask_arr == 1))
    valid_pixels = int(np.sum(valid_mask))
    flood_fraction = round(flood_pixels / valid_pixels, 6) if valid_pixels > 0 else 0.0

    pixel_area = _compute_pixel_area_m2(crs, transform, height, width) if is_georef else None
    real_area_m2 = round(flood_pixels * pixel_area, 2) if pixel_area is not None else None

    return {
        "status": "OK",
        "error_code": None,
        "reason": "Flood segmentation completed successfully",
        "image_path": str(resolved_img),
        "width": width,
        "height": height,
        "flood_pixel_count": flood_pixels,
        "total_valid_pixels": valid_pixels,
        "flood_fraction": flood_fraction,
        "confidence": round(confidence, 4),
        "georeferenced": bool(is_georef),
        "crs": str(crs) if is_georef else None,
        "bounds": bounds if is_georef else None,
        "real_world_area_m2": real_area_m2,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "preview_path": str(preview_path),
        "model_backend": model_bundle.get("backend", "sar_threshold_m0"),
        "provenance": "MODEL_DERIVED",
    }


def analyze_batch(
    image_paths: list[str] | str,
    output_dir: str,
    model_bundle: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run flood segmentation across a list of image paths or a directory of images.
    """
    if model_bundle is None:
        model_bundle = load_model()

    if isinstance(image_paths, (str, Path)):
        dir_p = _resolve_path(image_paths)
        if dir_p.is_dir():
            paths = [
                str(p)
                for p in sorted(dir_p.iterdir())
                if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")
                and not p.stem.endswith("_mask")
            ]
        else:
            paths = [str(image_paths)]
    else:
        paths = [str(p) for p in image_paths]

    return [
        analyze_flood_image(p, output_dir=output_dir, model_bundle=model_bundle)
        for p in paths
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloodSense AI M5 Flood Segmentation CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image or folder")
    parser.add_argument("--out", type=str, required=True, help="Output directory for masks/overlays")
    parser.add_argument("--model-dir", type=str, default=None, help="Optional checkpoint directory")
    args = parser.parse_args()

    bundle = load_model(args.model_dir)
    target = _resolve_path(args.image)
    if target.is_dir():
        res = analyze_batch(str(target), args.out, model_bundle=bundle)
    else:
        res = analyze_flood_image(str(target), args.out, model_bundle=bundle)
    print(json.dumps(res, indent=2))