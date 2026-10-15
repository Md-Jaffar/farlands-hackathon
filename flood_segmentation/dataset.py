from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from predict import synthesize_sar_three_channel


# Standard Sentinel-1 dB channel mean and std for [VV, VH, VV - VH]
SAR_MEAN = np.array([-12.0, -19.0, 7.0], dtype=np.float32).reshape(3, 1, 1)
SAR_STD = np.array([5.0, 5.0, 4.0], dtype=np.float32).reshape(3, 1, 1)


def find_s1hand_pairs(data_dir: str | Path) -> list[tuple[Path, Path]]:
    """
    Find all (<prefix>_S1Hand.tif, <prefix>_LabelHand.tif) pairs in a directory
    (searches recursively so it works whether images and masks are in the same
    folder or separate subfolders).
    """
    root = Path(data_dir)
    if not root.exists():
        return []

    s1_files = sorted(root.rglob("*_S1Hand.tif"))
    label_map: dict[str, Path] = {}
    for lbl_path in root.rglob("*_LabelHand.tif"):
        prefix = lbl_path.name.replace("_LabelHand.tif", "")
        label_map[prefix] = lbl_path

    pairs: list[tuple[Path, Path]] = []
    for s1_path in s1_files:
        prefix = s1_path.name.replace("_S1Hand.tif", "")
        if prefix in label_map:
            pairs.append((s1_path, label_map[prefix]))
    return pairs


class Sen1Floods11Dataset(Dataset):
    """
    PyTorch Dataset for Sen1Floods11 2-band Sentinel-1 SAR (VV, VH) chips.
    Synthesizes the 3rd channel (VV - VH in dB scale) -> (3, 512, 512) tensor
    and preserves ignore_index = -1 on ground-truth masks.
    """

    def __init__(
        self,
        pairs: list[tuple[str | Path, str | Path]],
        augment: bool = False,
        ignore_index: int = -1,
    ) -> None:
        self.pairs = [(Path(img), Path(lbl)) for img, lbl in pairs]
        self.augment = augment
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, lbl_path = self.pairs[idx]

        with rasterio.open(img_path) as src_img:
            sar_data = src_img.read().astype(np.float32)  # Expected (2, 512, 512)

        with rasterio.open(lbl_path) as src_lbl:
            mask_data = src_lbl.read(1).astype(np.int64)  # Expected (512, 512) with {0, 1, -1}

        vv = sar_data[0]
        vh = sar_data[1] if sar_data.shape[0] > 1 else (vv - 6.0)

        # Track invalid/nodata radar pixels and enforce ignore_index = -1 in mask
        valid_sar = np.isfinite(vv) & np.isfinite(vh)
        stacked_3ch, _ = synthesize_sar_three_channel(vv, vh)

        # Normalize [VV, VH, VV - VH] for neural network stability
        norm_3ch = (stacked_3ch - SAR_MEAN) / SAR_STD

        # Clean mask: anything outside {0, 1} or where SAR is invalid becomes ignore_index (-1)
        clean_mask = np.where(
            valid_sar & ((mask_data == 0) | (mask_data == 1)),
            mask_data,
            self.ignore_index,
        ).astype(np.int64)

        # Optional deterministic-safe geometric flips during training
        if self.augment:
            if np.random.rand() > 0.5:
                norm_3ch = np.flip(norm_3ch, axis=2).copy()
                clean_mask = np.flip(clean_mask, axis=1).copy()
            if np.random.rand() > 0.5:
                norm_3ch = np.flip(norm_3ch, axis=1).copy()
                clean_mask = np.flip(clean_mask, axis=0).copy()

        return {
            "pixel_values": torch.from_numpy(norm_3ch).float(),  # (3, 512, 512)
            "labels": torch.from_numpy(clean_mask).long(),       # (512, 512)
            "image_path": str(img_path),
            "mask_path": str(lbl_path),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify Sen1Floods11 PyTorch DataLoader")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../Research_Pack",
        help="Folder containing *_S1Hand.tif and *_LabelHand.tif chips",
    )
    args = parser.parse_args()

    found_pairs = find_s1hand_pairs(args.data_dir)
    print(f"Found {len(found_pairs)} (S1Hand, LabelHand) chip pairs.")
    if found_pairs:
        ds = Sen1Floods11Dataset(found_pairs)
        sample = ds[0]
        pv = sample["pixel_values"]
        lb = sample["labels"]
        unique_labels = torch.unique(lb).tolist()
        print(f"Sample 0 image: { Path(sample['image_path']).name }")
        print(f"Tensor shape: {tuple(pv.shape)} | dtype: {pv.dtype}")
        print(f"Mask shape:   {tuple(lb.shape)} | unique values: {unique_labels}")