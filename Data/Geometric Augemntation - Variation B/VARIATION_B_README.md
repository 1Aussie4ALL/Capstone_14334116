# Variation B - Geometric Augmentation (Anatomically Plausible)

## Overview
Variation B implements **geometric augmentations** that are anatomically plausible for MRI brain tumor images. Unlike Variation A (photometric), this focuses on spatial transformations that maintain the physiological realism of brain structures.

## What It Does
Variation B applies subtle geometric transformations to create augmented versions of the original dataset while preserving the anatomical integrity of brain MRI scans.

## Geometric Transformations

### 1. **Rotation** (p≈0.3)
- **Range**: ±7° (normal), ±10° (cap)
- **Purpose**: Simulates slight head tilting during scanning
- **Safety**: Small angles preserve brain structure orientation

### 2. **Translation** (p≈0.3)
- **Range**: ≤3% of image width/height
- **Purpose**: Simulates slight positioning variations
- **Safety**: Minimal displacement maintains ROI visibility

### 3. **Scaling/Zooming** (p≈0.3)
- **Range**: 0.95–1.05 (normal), 0.9–1.1 (cap)
- **Purpose**: Simulates different scanning distances
- **Safety**: Subtle scaling preserves relative proportions

### 4. **Horizontal Flip** (p=0.3)
- **Purpose**: Simulates left-right orientation variations
- **Safety**: Safe for tumor type classification (laterality not informative)

### 5. **Light Elastic Deformation** (p=0.2)
- **Parameters**: α ≈ 1–3 px, σ ≈ 6–8 px
- **Purpose**: Simulates minor tissue deformation
- **Safety**: Very mild warping maintains anatomical plausibility

## Guardrails & Safety Features

✅ **No Vertical Flips** - Anatomically implausible for brain scans  
✅ **Subtle Combined Transforms** - Prevents non-physiological distortions  
✅ **Probability Control** - Each augmentation has controlled application rate  
✅ **1-2 Augmentations Per Image** - Prevents over-augmentation  

## Data Source
- **Input**: Uses the same source data as Variation A (`data_used_for_variation_A_800`)
- **Output**: Creates `Variation B_results_800` with geometrically augmented images
- **Target**: 800 augmented images per cancer type (same as Variation A)

## Usage

### Quick Start
```bash
# Run the batch file (Windows)
run_variation_b.bat

# Or run directly with Python
python create_variation_b_dataset_800.py
```

### Requirements
Install dependencies first:
```bash
pip install -r requirements_variation_b.txt
```

## Output Structure
```
Variation B_results_800/
├── glioma/
│   ├── glioma_0001_Tr-gl_1310_variation_B.jpg
│   ├── glioma_0002_Tr-gl_0228_variation_B.jpg
│   └── ...
├── meningioma/
├── notumor/
├── pituitary/
└── not_mri/
```

## Summary Report
The script generates `Variation_B_800_Summary_Report.txt` containing:
- Processing statistics for each cancer type
- Total image counts
- Detailed augmentation parameters
- Safety measures implemented

## Why Geometric Augmentation?
- **Realistic Variations**: Simulates actual scanning conditions
- **Anatomical Preservation**: Maintains brain structure integrity
- **Data Diversity**: Increases training dataset variety
- **Clinical Relevance**: Reflects real-world MRI variations

## Comparison with Variation A
| Aspect | Variation A (Photometric) | Variation B (Geometric) |
|--------|---------------------------|-------------------------|
| **Type** | Color/brightness changes | Spatial transformations |
| **Focus** | Image appearance | Image geometry |
| **Safety** | No structural change | Anatomical preservation |
| **Use Case** | Lighting variations | Positioning variations |

## Notes
- **Seed Fixed**: Uses seed=42 for reproducible results
- **Error Handling**: Gracefully handles corrupted images
- **Progress Tracking**: Shows processing progress every 100 images
- **Memory Efficient**: Processes images one at a time
