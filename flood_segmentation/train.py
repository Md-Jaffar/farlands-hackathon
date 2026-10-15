from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from dataset import Sen1Floods11Dataset, find_s1hand_pairs


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for deterministic training (Rule 7)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CombinedCEDiceLoss(nn.Module):
    """
    Combined Cross-Entropy + Soft Dice Loss for binary flood segmentation,
    strictly ignoring pixels where target == ignore_index (-1).
    """

    def __init__(
        self,
        ignore_index: int = -1,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        # Up-weight class 1 (flood) slightly to counter background imbalance
        self.register_buffer("class_weights", torch.tensor([1.0, 2.0], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Masked Cross-Entropy Loss
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights.to(logits.device),
            ignore_index=self.ignore_index,
        )

        # 2. Masked Soft Dice Loss on Flood Class (class 1)
        valid_mask = (targets != self.ignore_index).float()
        if valid_mask.sum() == 0:
            return ce_loss

        probs_flood = torch.softmax(logits, dim=1)[:, 1, :, :] * valid_mask
        target_flood = (targets == 1).float() * valid_mask

        intersection = (probs_flood * target_flood).sum(dim=(1, 2))
        denom = probs_flood.sum(dim=(1, 2)) + target_flood.sum(dim=(1, 2))
        dice_score = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def compute_confusion_metrics(
    preds: np.ndarray, targets: np.ndarray, ignore_index: int = -1
) -> dict[str, int]:
    """Accumulate TP, FP, FN, TN on valid (non-ignored) pixels."""
    valid = targets != ignore_index
    p = preds[valid]
    t = targets[valid]
    tp = int(np.sum((p == 1) & (t == 1)))
    fp = int(np.sum((p == 1) & (t == 0)))
    fn = int(np.sum((p == 0) & (t == 1)))
    tn = int(np.sum((p == 0) & (t == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def summarize_metrics(counts: dict[str, int]) -> dict[str, float]:
    """Calculate Flood IoU, Dice/F1, Precision, Recall, and Pixel Accuracy."""
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "flood_iou": round(float(iou), 4),
        "dice_f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "pixel_accuracy": round(float(accuracy), 4),
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    ignore_index: int = -1,
) -> dict[str, Any]:
    """Run held-out evaluation and return segmentation metrics."""
    model.eval()
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]  # (B, H, W)

        outputs = model(pixel_values=pixel_values)
        logits = F.interpolate(
            outputs.logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        targets = labels.numpy()

        step_counts = compute_confusion_metrics(preds, targets, ignore_index=ignore_index)
        for k in totals:
            totals[k] += step_counts[k]

    metrics = summarize_metrics(totals)
    metrics["confusion_matrix"] = totals
    return metrics


def train_segformer(
    data_dir: str,
    model_name: str = "nvidia/mit-b3",
    epochs: int = 8,
    batch_size: int = 2,
    lr: float = 1e-4,
) -> dict[str, Any]:
    set_seed(42)
    base_dir = Path(__file__).resolve().parent
    best_model_dir = base_dir / "model" / "best_model"
    metrics_dir = base_dir / "metrics"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_s1hand_pairs(data_dir)
    if not pairs:
        raise RuntimeError(f"No (*_S1Hand.tif, *_LabelHand.tif) pairs found in '{data_dir}'.")

    # Split: keep at least 1 held-out validation chip (or 20% if dataset is larger)
    if len(pairs) >= 4:
        val_count = max(1, len(pairs) // 4)
        train_pairs = pairs[:-val_count]
        val_pairs = pairs[-val_count:]
    else:
        train_pairs = pairs
        val_pairs = pairs

    print(
        f"Training on {len(train_pairs)} chip(s) | Evaluating on {len(val_pairs)} held-out chip(s)"
    )

    train_ds = Sen1Floods11Dataset(train_pairs, augment=True)
    val_ds = Sen1Floods11Dataset(val_pairs, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {model_name} on device: {device}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "background", 1: "flood"},
        label2id={"background": 0, "flood": 1},
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    criterion = CombinedCEDiceLoss(ignore_index=-1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_iou = -1.0
    best_metrics: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            logits = F.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        avg_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate_loader(model, val_loader, device=device)
        val_iou = val_metrics["flood_iou"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | Dice/F1: {val_metrics['dice_f1']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}"
        )

        if val_iou >= best_iou:
            best_iou = val_iou
            best_metrics = dict(val_metrics)
            model.save_pretrained(str(best_model_dir))

    # Write final evaluation report to metrics/metrics.json
    final_report = {
        "model_architecture": model_name,
        "dataset": "Sen1Floods11 (Assam, India - S1Hand / LabelHand)",
        "input_channels": ["VV_dB", "VH_dB", "VV_minus_VH_dB"],
        "resolution": [512, 512],
        "ignore_index": -1,
        "train_chips": [p[0].name for p in train_pairs],
        "held_out_val_chips": [p[0].name for p in val_pairs],
        "epochs_trained": epochs,
        "device": device,
        "metrics": best_metrics,
        "provenance": "MODEL_DERIVED",
    }

    metrics_path = metrics_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nSaved best checkpoint to: {best_model_dir}")
    print(f"Saved honest evaluation metrics to: {metrics_path}")
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SegFormer on Sen1Floods11 SAR chips")
    parser.add_argument("--data-dir", type=str, default="..", help="Root folder containing S1Hand/LabelHand chips")
    parser.add_argument("--model", type=str, default="nvidia/mit-b3", help="Pretrained SegFormer backbone")
    parser.add_argument("--epochs", type=int, default=8, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_segformer(
        data_dir=args.data_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )