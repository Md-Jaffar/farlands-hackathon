# FloodSense AI — Work Package 04: Flood Segmentation (M5 Vision)

Predicts flood inundation masks from Sentinel-1 SAR and RGB aerial imagery. 

## Capabilities
- **Milestone M0**: Physics-informed SAR backscatter threshold baseline.
- **Milestone M2**: Fine-tuned SegFormer (`nvidia/mit-b3`) for 2-class semantic segmentation.
- Extracts valid `ignore_index` nodata masks and accurately calculates real-world flood area in square meters.

## Commands

Run offline inference on a single image:
```bash
python predict.py --image ../Research_Pack/India_324254_S1Hand.tif --out examples/