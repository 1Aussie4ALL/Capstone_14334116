# Brain Tumour Detection Using Deep Learning

A deep learning research project investigating how different data augmentation strategies affect the performance, generalisation, and calibration of CNN-based brain tumour classification models using MRI scans.

This project was completed as part of my Software Engineering (Honours) degree, with a focus on applying AI and computer vision techniques to a real-world medical imaging problem.

---

## Project Overview

Brain tumour classification from MRI images is a challenging computer vision problem where model performance can be heavily influenced by the quality, diversity, and balance of training data.

This project systematically evaluates different image augmentation strategies to determine how they affect CNN-based brain tumour detection.

The system classifies MRI scans into four categories:

- Glioma
- Meningioma
- Pituitary Tumour
- No Tumour

Rather than evaluating accuracy alone, the project also considers model confidence, class-level performance, ROC-AUC, Precision-Recall performance, and calibration.

---

## Research Objective

The main objective of this project was to investigate:

> How do different data augmentation strategies influence the classification performance and confidence calibration of CNN-based brain tumour detection models?

Two major augmentation approaches were compared.

### Variation A — Photometric Augmentation

Changes the visual appearance of MRI images while largely preserving their spatial structure.

Examples include:

- Brightness adjustments
- Contrast changes
- Image intensity transformations

### Variation B — Geometric Augmentation

Introduces spatial transformations to improve the model's ability to recognise tumours under different orientations and positions.

Examples include:

- Rotation
- Translation
- Scaling
- Flipping

---

## Dataset

The final dataset combines MRI images from multiple publicly available brain tumour datasets, including:

- Figshare
- SARTAJ
- BR35H

Approximately **7,000 MRI images** were used across the four classification categories.

```text
MRI Dataset
│
├── Glioma
├── Meningioma
├── Pituitary
└── No Tumour
