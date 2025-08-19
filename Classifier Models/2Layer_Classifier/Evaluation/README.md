# 🔍 2-Layer MRI Classifier - Comprehensive Evaluation

## 📊 **What This Evaluation Provides**

### **1. Classification Reports (Validation + Test)**
- **Per-class metrics**: Precision, Recall, F1-score, Support
- **Macro averages**: Unweighted mean across all classes
- **Micro averages**: Overall metrics considering all samples
- **Weighted averages**: Class-balanced metrics

### **2. Normalized Confusion Matrices**
- **Training/Validation set**: Shows model performance on seen data
- **Test set**: Shows model performance on unseen data
- **Normalized by true labels**: Each row sums to 1.0

### **3. Precision-Recall Curves**
- **Per-class PR curves**: Better than ROC for imbalanced datasets
- **Average Precision (AP)**: Area under PR curve
- **PR-AUC scores**: Per-class performance metrics

### **4. ROC Curves**
- **Per-class ROC curves**: Traditional performance visualization
- **ROC-AUC scores**: Area under ROC curve per class
- **Macro/micro averages**: Overall ROC performance

### **5. Calibration Plots**
- **Reliability diagrams**: How well probabilities match actual frequencies
- **ECE (Expected Calibration Error)**: Calibration quality metric
- **Brier scores**: Probability prediction accuracy

### **6. Summary CSV**
One-row summary with key metrics:
- `accuracy`, `macro_F1`, `weighted_F1`
- `balanced_accuracy`, `macro_PR_AUC`, `macro_ROC_AUC`
- `ECE`, `Brier`

## 🚀 **How to Run**

### **Option 1: Double-click batch file**
```bash
run_evaluation.bat
```

### **Option 2: Command line**
```bash
cd 2Layer_Classifier/Evaluation
python evaluate_2layer_classifier.py
```

## 📁 **Output Structure**

After running, you'll get:
```
Evaluation_Results/
├── confusion_matrix_train_normalized.png
├── confusion_matrix_test_normalized.png
├── pr_curves_train.png
├── pr_curves_test.png
├── roc_curves_train.png
├── roc_curves_test.png
├── calibration_train.png
├── calibration_test.png
├── summary_train.csv
└── summary_test.csv
```

## 🔧 **Technical Details**

- **Model**: Uses `../Models/mri_2layer_classifier.h5`
- **Dataset**: Loads from `../../Dataset/`
- **Image size**: 128×128 pixels (same as training)
- **Classes**: 5 classes (4 MRI types + 1 non-MRI)

## 📈 **Interpreting Results**

### **High Performance Indicators:**
- **Accuracy > 90%**: Overall good performance
- **Macro F1 > 0.85**: Balanced performance across classes
- **PR-AUC > 0.90**: Good precision-recall balance
- **ECE < 0.1**: Well-calibrated probabilities
- **Brier < 0.1**: Accurate probability predictions

### **Class-Specific Analysis:**
- **High precision**: Low false positives
- **High recall**: Low false negatives
- **Balanced F1**: Good precision-recall trade-off

## 🎯 **Use Cases**

- **Model validation**: Assess training quality
- **Performance comparison**: Compare with other models
- **Error analysis**: Identify problematic classes
- **Publication**: Generate figures for papers
- **Deployment**: Ensure production readiness

---

**Ready to evaluate?** Just run `run_evaluation.bat`! 🚀
