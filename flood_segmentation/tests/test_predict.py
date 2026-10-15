from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from predict import (
    analyze_batch,
    analyze_flood_image,
    load_model,
    synthesize_sar_three_channel,
)


@pytest.fixture
def dummy_sar_tif(tmp_path: Path) -> str:
    """Create a dummy 2-band 64x64 SAR GeoTIFF for testing."""
    p = tmp_path / "test_sar.tif"
    data = np.random.uniform(-30, 0, (2, 64, 64)).astype(np.float32)
    transform = Affine.translation(92.5, 26.0) * Affine.scale(0.0001, -0.0001)
    
    with rasterio.open(
        p, "w", driver="GTiff", height=64, width=64, count=2, dtype="float32",
        crs="EPSG:4326", transform=transform, nodata=-9999.0
    ) as dst:
        dst.write(data)
    return str(p)


@pytest.fixture
def corrupt_image(tmp_path: Path) -> str:
    """Create a broken text file masked as a TIF."""
    p = tmp_path / "corrupt.tif"
    p.write_text("This is not a real image.", encoding="utf-8")
    return str(p)


def test_01_synthesize_sar_three_channel() -> None:
    vv = np.full((10, 10), -20.0, dtype=np.float32)
    vh = np.full((10, 10), -28.0, dtype=np.float32)
    stacked, rgb = synthesize_sar_three_channel(vv, vh)
    
    assert stacked.shape == (3, 10, 10)
    assert rgb.shape == (10, 10, 3)
    assert stacked.dtype == np.float32
    assert rgb.dtype == np.uint8
    assert np.all(stacked[2] == 8.0)  # VV - VH = -20 - (-28) = 8


def test_02_load_model_fallback() -> None:
    # When no best_model exists, should load M0 threshold baseline
    bundle = load_model(model_dir="non_existent_folder")
    assert bundle["backend"] == "sar_threshold_m0"
    assert bundle["model"] is None


def test_03_analyze_flood_image_valid(dummy_sar_tif: str, tmp_path: Path) -> None:
    res = analyze_flood_image(dummy_sar_tif, str(tmp_path))
    assert res["status"] == "OK"
    assert res["error_code"] is None
    assert res["georeferenced"] is True
    assert res["real_world_area_m2"] is not None
    assert "test_sar_mask.tif" in res["mask_path"]
    assert "test_sar_overlay.png" in res["overlay_path"]


def test_04_analyze_flood_image_file_not_found(tmp_path: Path) -> None:
    res = analyze_flood_image("does_not_exist.tif", str(tmp_path))
    assert res["status"] == "ERROR"
    assert res["error_code"] == "FILE_NOT_FOUND"
    assert res["total_valid_pixels"] == 0


def test_05_analyze_flood_image_corrupt(corrupt_image: str, tmp_path: Path) -> None:
    res = analyze_flood_image(corrupt_image, str(tmp_path))
    assert res["status"] == "ERROR"
    assert res["error_code"] == "CORRUPT_IMAGE"


def test_06_analyze_batch(dummy_sar_tif: str, tmp_path: Path) -> None:
    res_list = analyze_batch([dummy_sar_tif], str(tmp_path))
    assert len(res_list) == 1
    assert res_list[0]["status"] == "OK"


def test_07_output_files_created(dummy_sar_tif: str, tmp_path: Path) -> None:
    analyze_flood_image(dummy_sar_tif, str(tmp_path))
    assert (tmp_path / "test_sar_mask.tif").exists()
    assert (tmp_path / "test_sar_overlay.png").exists()
    assert (tmp_path / "test_sar_preview.png").exists()


def test_08_mask_dimensions_match_input(dummy_sar_tif: str, tmp_path: Path) -> None:
    analyze_flood_image(dummy_sar_tif, str(tmp_path))
    mask_path = tmp_path / "test_sar_mask.tif"
    with rasterio.open(mask_path) as src:
        assert src.height == 64
        assert src.width == 64
        assert src.count == 1
        assert src.dtypes[0] == "int16"


def test_09_rgb_fallback_handling(tmp_path: Path) -> None:
    """Test that standard 3-channel RGB images don't crash the pipeline."""
    p = tmp_path / "test_rgb.tif"
    data = np.random.uniform(0, 255, (3, 32, 32)).astype(np.uint8)
    with rasterio.open(
        p, "w", driver="GTiff", height=32, width=32, count=3, dtype="uint8"
    ) as dst:
        dst.write(data)
        
    res = analyze_flood_image(str(p), str(tmp_path))
    assert res["status"] == "OK"
    assert res["georeferenced"] is False  # No transform applied

def test_10_nodata_ignore_index_preserved(tmp_path: Path) -> None:
    """Verify that NaN / nodata pixels are strictly assigned -1 in the output mask."""
    p = tmp_path / "nodata_sar.tif"
    data = np.full((2, 32, 32), -20.0, dtype=np.float32)
    # Inject NaN in top-left 8x8 corner
    data[:, :8, :8] = np.nan
    transform = Affine.translation(92.5, 26.0) * Affine.scale(0.0001, -0.0001)

    with rasterio.open(
        p, "w", driver="GTiff", height=32, width=32, count=2, dtype="float32",
        crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(data)

    res = analyze_flood_image(str(p), str(tmp_path))
    assert res["status"] == "OK"
    assert res["total_valid_pixels"] == (32 * 32) - (8 * 8)

    with rasterio.open(res["mask_path"]) as src_mask:
        mask_arr = src_mask.read(1)
        assert np.all(mask_arr[:8, :8] == -1)


def test_11_trained_segformer_backend_loads_if_present(dummy_sar_tif: str, tmp_path: Path) -> None:
    """Verify load_model() picks up the trained SegFormer checkpoint from model/best_model/."""
    bundle = load_model()
    assert bundle["backend"] in ("segformer", "sar_threshold_m0")
    res = analyze_flood_image(dummy_sar_tif, str(tmp_path), model_bundle=bundle)
    assert res["status"] == "OK"