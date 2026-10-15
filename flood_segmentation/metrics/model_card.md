# FloodSense AI: SegFormer-B3 Model Card (M5 Vision)

## Model Architecture
- **Backbone**: Hugging Face `nvidia/mit-b3` (SegFormer)
- **Input**: 3-channel synthesized Sentinel-1 SAR tensor `[VV, VH, VV - VH]` in dB scale.
- **Output**: Binary segmentation mask (0 = Background, 1 = Flood, -1 = Ignore/NoData).

## Known Limitations & Edge Cases (Judge Q&A Prep)
1. **Urban Bounce (Double Bounce):** Radar waves bounce off buildings and back to the satellite, appearing incredibly bright. Flooded urban streets can sometimes be missed because the building scatter overpowers the dark specular reflection of the water.
2. **Radar Shadows:** Steep terrain or tall buildings block the radar signal, creating dark "shadows" that the threshold baseline might falsely classify as water. The SegFormer model learns context to ignore these better than the baseline.
3. **Wind/Ripples:** High winds rough up the water surface, causing radar waves to bounce back to the sensor instead of reflecting away, which makes the water appear brighter and harder to detect.