# SAR-to-EO Translation using CycleGAN

## Project Overview

This project implements **CycleGAN-based SAR-to-EO translation** to convert Synthetic Aperture Radar (SAR) images into realistic Earth Observation (EO) optical images. The system generates high-quality optical imagery from all-weather SAR data across three different spectral configurations.

### Key Features
- Cross-modal translation from SAR to optical imagery
- Multi-spectral generation: RGB, NIR/SWIR/RedEdge, and RGB+NIR configurations
- All-weather capability for continuous monitoring
- Vegetation spectral integrity preservation

### Applications
- Disaster response and emergency monitoring
- Agricultural crop health assessment
- Environmental time series gap-filling
- Enhanced visual interpretation of radar data

## Quick Start

### Prerequisites
```bash
pip install torch torchvision rasterio matplotlib scikit-image opencv-python tqdm Pillow numpy
```

### Dataset Structure
```
SAR-to-EO/split_dataset/
├── stratified_split/
│   ├── train/ (SAR/, EO/)
│   ├── val/ (SAR/, EO/)
│   └── test/ (SAR/, EO/)
├── geographic_split/
└── random_split/
```

### Usage
```python
# Configure paths
base_path = r"path/to/your/SAR-to-EO/split_dataset"

# Train all configurations
results, metrics = train_all_configurations(
    base_path=base_path,
    split_type='stratified_split',
    num_epochs=10,
    batch_size=2
)

# Generate performance analysis
print_final_performance_summary(metrics)
plot_training_metrics(metrics)
```

## Model Architecture

### CycleGAN Implementation
- **Generators**: ResNet-based with 9 residual blocks
- **Discriminators**: PatchGAN (70×70 patches)
- **Loss Functions**: Adversarial + Cycle Consistency (λ=10.0)
- **Optimizer**: Adam (lr=0.0002, β₁=0.5, β₂=0.999)

### Data Processing
- **Normalization**: All images scaled to [-1, 1] range
- **Band Configurations**: 
  - RGB: Red, Green, Blue channels
  - NIR_SWIR_RedEdge: Near-IR, SWIR, Red Edge
  - RGB_NIR: RGB + Near-Infrared
- **Splitting Strategy**: Stratified, Geographic, and Random splits

## Results

### Performance Summary (10 Epochs)

| Configuration | Avg SSIM | Avg PSNR (dB) | NDVI Preservation | Generator Loss |
|---------------|-----------|---------------|-------------------|----------------|
| RGB | 0.2157 | 15.88 | **0.1290** | 5.2569 |
| NIR_SWIR_RedEdge | **0.4445** | **26.81** | 0.2156 | **4.9081** |
| RGB_NIR | 0.3101 | 18.12 | 0.2579 | 5.1048 |

### Key Findings
- **NIR_SWIR_RedEdge** achieved highest overall performance (PSNR: 26.81 dB)
- **SWIR channel** showed exceptional quality (52.06 dB PSNR)
- **RGB configuration** best preserved vegetation characteristics
- All configurations showed consistent improvement without overfitting

## Technical Stack

- **Deep Learning**: PyTorch, torchvision
- **Geospatial**: rasterio
- **Computer Vision**: OpenCV, scikit-image
- **Visualization**: matplotlib
- **Utilities**: numpy, Pillow, tqdm

## Future Work

- Extended training with 50+ epochs
- Perceptual loss integration
- Cloud detection capabilities
- Multi-temporal analysis
- Real-time inference optimization


## 5 SAR → EO sample results
<img width="1866" height="667" alt="Screenshot 2025-07-24 221238" src="https://github.com/user-attachments/assets/aa4b0314-2a61-41c7-b8b2-5f7907c94f82" />
<img width="1864" height="657" alt="Screenshot 2025-07-24 221255" src="https://github.com/user-attachments/assets/1d049bcd-b17b-4e0d-8a52-3f4c17cb449b" />
<img width="1843" height="670" alt="Screenshot 2025-07-24 221312" src="https://github.com/user-attachments/assets/b0243a11-78a9-43ad-8b88-614da5e5b0a9" />




## Citation

```bibtex
@misc{kumar2024sar2eo,
  title={SAR-to-EO Translation using CycleGAN for Multi-spectral Remote Sensing},
  author={Adarsh Kumar},
  year={2024},
  email={adarsh25nov@gmail.com}
}
```

## Contact

**Adarsh Kumar** - adarsh25nov@gmail.com

---

This project demonstrates the potential of generative adversarial networks in remote sensing, enabling realistic optical imagery generation from all-weather SAR data for enhanced Earth observation capabilities.
