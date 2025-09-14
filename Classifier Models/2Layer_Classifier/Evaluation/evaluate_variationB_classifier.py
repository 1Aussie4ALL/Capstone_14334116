import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_auc_score, roc_curve, brier_score_loss,
    precision_recall_fscore_support, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("🔍 Comprehensive VariationB MRI Classifier Evaluation")
print("=" * 60)
print("📊 This classifier combines original dataset + Variation B 800 dataset")
print("=" * 60)

# Configuration
IMG_SIZE = 128
MODEL_PATH = '../../VariationB_Enhanced/Models/mri_variationB_classifier.h5'
DATASET_PATH = '../../Dataset'

# Classes
MRI_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NON_MRI_CLASS = 'not_mri'
ALL_CLASSES = MRI_CLASSES + [NON_MRI_CLASS]


def load_and_preprocess_data(data_type='Testing'):
    """Load and preprocess dataset (Training or Testing)"""
    print(f"\n📥 Loading {data_type} dataset...")

    images = []
    labels = []

    # Load MRI images (tumor types)
    for mri_class in MRI_CLASSES:
        class_path = os.path.join(DATASET_PATH, data_type, mri_class)
        if os.path.exists(class_path):
            print(f"  Loading {mri_class}: ", end="")
            count = 0
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype('float32') / 255.0

                        images.append(img)
                        labels.append(mri_class)
                        count += 1
            print(f"{count} images")

    # Load non-MRI images
    not_mri_path = os.path.join(DATASET_PATH, data_type, 'not_mri')
    if os.path.exists(not_mri_path):
        print(f"  Loading not_mri: ", end="")
        count = 0
        for filename in os.listdir(not_mri_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(not_mri_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype('float32') / 255.0

                    images.append(img)
                    labels.append('not_mri')
                    count += 1
        print(f"{count} images")

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    print(f"✅ {data_type} dataset loaded: {len(X)} total images")

    # Convert labels to numeric indices
    label_to_index = {label: idx for idx, label in enumerate(ALL_CLASSES)}
    y_numeric = np.array([label_to_index[label] for label in y])

    return X, y_numeric, y, label_to_index


def load_model_and_predict():
    """Load the trained VariationB model"""
    global MODEL_PATH
    print(f"\n🤖 Loading VariationB model: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Trying alternative paths...")

        # Try alternative paths
        alternative_paths = [
            'mri_variationB_classifier.h5',
            '../Models/mri_variationB_classifier_final.h5',
            'mri_variationB_classifier_final.h5'
        ]

        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"✅ Found model at: {alt_path}")
                MODEL_PATH = alt_path
                break
        else:
            print("❌ Model not found in any expected location")
            return None

    try:
        model = load_model(MODEL_PATH)
        print("✅ VariationB model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def get_predictions_and_probabilities(model, X):
    """Get predictions and probabilities for all classes"""
    print("\n🔮 Making predictions...")

    probabilities = model.predict(X, verbose=0)
    predicted_classes = np.argmax(probabilities, axis=1)
    return probabilities, predicted_classes


def calculate_classification_metrics(y_true, y_pred, y_true_str, label_to_index):
    """Calculate comprehensive classification metrics"""
    print("\n📊 Calculating classification metrics...")

    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y_pred_str = [index_to_label[idx] for idx in y_pred]

    print("\n📈 Classification Report:")
    print(classification_report(y_true_str, y_pred_str, target_names=ALL_CLASSES, digits=4))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(ALL_CLASSES))
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    accuracy = np.mean(y_pred == y_true)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'micro': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1
        },
        'weighted': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        },
        'overall': {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc
        }
    }

    return metrics, y_pred_str


def plot_confusion_matrix(y_true, y_pred, save_path, title):
    """Plot normalized confusion matrix"""
    print(f"\n📊 Creating {title}...")

    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=ALL_CLASSES, yticklabels=ALL_CLASSES,
                cbar_kws={'label': 'Normalized Count'})
    plt.title(f'{title} - VariationB Classifier (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return cm


def plot_precision_recall_curves(probabilities, y_true, save_path):
    """Plot PR curves for each class"""
    print(f"\n📈 Creating Precision-Recall curves...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    pr_aucs = []
    for i, class_name in enumerate(ALL_CLASSES):
        y_binary = (y_true == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_binary, probabilities[:, i])
        pr_auc = average_precision_score(y_binary, probabilities[:, i])
        pr_aucs.append(pr_auc)

        axes[i].plot(recall, precision, label=f'{class_name} (AP={pr_auc:.3f})', linewidth=2)
        axes[i].set_xlabel('Recall', fontsize=12)
        axes[i].set_ylabel('Precision', fontsize=12)
        axes[i].set_title(f'{class_name} - PR Curve (VariationB)', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])

    axes[5].set_visible(False)

    plt.suptitle('VariationB Classifier - Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return pr_aucs


def plot_roc_curves(probabilities, y_true, save_path):
    """Plot ROC curves for each class"""
    print(f"\n📊 Creating ROC curves...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    roc_aucs = []
    for i, class_name in enumerate(ALL_CLASSES):
        y_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, probabilities[:, i])
        roc_auc = roc_auc_score(y_binary, probabilities[:, i])
        roc_aucs.append(roc_auc)

        axes[i].plot(fpr, tpr, label=f'{class_name} (AUC={roc_auc:.3f})', linewidth=2)
        axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        axes[i].set_xlabel('False Positive Rate', fontsize=12)
        axes[i].set_ylabel('True Positive Rate', fontsize=12)
        axes[i].set_title(f'{class_name} - ROC Curve (VariationB)', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])

    axes[5].set_visible(False)

    plt.suptitle('VariationB Classifier - ROC Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return roc_aucs


def plot_calibration_curves(probabilities, y_true, save_path):
    """Plot calibration curves and calculate ECE/Brier scores"""
    print(f"\n🎯 Creating calibration plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    ece_scores = []
    brier_scores = []

    for i, class_name in enumerate(ALL_CLASSES):
        y_binary = (y_true == i).astype(int)
        probs = probabilities[:, i]

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, probs, n_bins=10
        )

        brier = brier_score_loss(y_binary, probs)
        brier_scores.append(brier)

        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        ece_scores.append(ece)

        axes[i].plot(mean_predicted_value, fraction_of_positives, 's-',
                     label=f'{class_name}', markersize=8, linewidth=2)
        axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly calibrated')
        axes[i].set_xlabel('Mean Predicted Probability', fontsize=12)
        axes[i].set_ylabel('Fraction of Positives', fontsize=12)
        axes[i].set_title(f'{class_name} - Calibration (VariationB)\nECE={ece:.3f}, Brier={brier:.3f}',
                         fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])

    axes[5].set_visible(False)

    plt.suptitle('VariationB Classifier - Calibration Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return ece_scores, brier_scores


def create_summary_csv(metrics, pr_aucs, roc_aucs, ece_scores, brier_scores, save_path):
    """Create one-row summary CSV with all metrics"""
    print(f"\n💾 Creating summary CSV...")

    macro_pr_auc = np.mean(pr_aucs)
    macro_roc_auc = np.mean(roc_aucs)
    macro_ece = np.mean(ece_scores)
    macro_brier = np.mean(brier_scores)

    summary_data = {
        'accuracy': [metrics['overall']['accuracy']],
        'macro_F1': [metrics['macro']['f1']],
        'weighted_F1': [metrics['weighted']['f1']],
        'balanced_accuracy': [metrics['overall']['balanced_accuracy']],
        'macro_PR_AUC': [macro_pr_auc],
        'macro_ROC_AUC': [macro_roc_auc],
        'ECE': [macro_ece],
        'Brier': [macro_brier]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(save_path, index=False)

    print(f"✅ Summary saved to: {save_path}")
    print("\n📊 Summary Metrics:")
    for col in summary_df.columns:
        print(f"  {col}: {summary_df[col].iloc[0]:.4f}")

    return summary_df


def main():
    """Main evaluation function"""
    try:
        # Load model
        model = load_model_and_predict()
        if model is None:
            return

        # Load datasets
        X_train, y_train, y_train_str, label_to_index = load_and_preprocess_data('Training')
        X_test, y_test, y_test_str, label_to_index = load_and_preprocess_data('Testing')

        # Combine for validation (no separate val set)
        X_val = X_train
        y_val = y_train
        y_val_str = y_train_str

        print(f"\n📊 Dataset Summary:")
        print(f"  Training: {len(X_train)} images")
        print(f"  Testing: {len(X_test)} images")
        print(f"  Note: VariationB classifier was trained on original + Variation B 800 dataset")

        # Predictions
        print("\n🔮 Getting predictions...")
        train_probs, train_preds = get_predictions_and_probabilities(model, X_train)
        test_probs, test_preds = get_predictions_and_probabilities(model, X_test)

        # Evaluate Training/Validation
        print("\n" + "="*50)
        print("📊 TRAINING/VALIDATION SET EVALUATION (VariationB)")
        print("="*50)

        train_metrics, train_pred_str = calculate_classification_metrics(
            y_train, train_preds, y_train_str, label_to_index
        )

        # Evaluate Test set
        print("\n" + "="*50)
        print("📊 TEST SET EVALUATION (VariationB)")
        print("="*50)

        test_metrics, test_pred_str = calculate_classification_metrics(
            y_test, test_preds, y_test_str, label_to_index
        )

        # Output directory
        os.makedirs('Evaluation_Results_VariationB', exist_ok=True)

        # 1. Confusion Matrices
        plot_confusion_matrix(
            y_train, train_preds,
            'Evaluation_Results_VariationB/confusion_matrix_variationB_train_normalized.png',
            'Training/Validation Confusion Matrix - VariationB'
        )

        plot_confusion_matrix(
            y_test, test_preds,
            'Evaluation_Results_VariationB/confusion_matrix_variationB_test_normalized.png',
            'Test Confusion Matrix - VariationB'
        )

        # 2. PR Curves
        train_pr_aucs = plot_precision_recall_curves(
            train_probs, y_train,
            'Evaluation_Results_VariationB/pr_curves_variationB_train.png'
        )

        test_pr_aucs = plot_precision_recall_curves(
            test_probs, y_test,
            'Evaluation_Results_VariationB/pr_curves_variationB_test.png'
        )

        # 3. ROC Curves
        train_roc_aucs = plot_roc_curves(
            train_probs, y_train,
            'Evaluation_Results_VariationB/roc_curves_variationB_train.png'
        )

        test_roc_aucs = plot_roc_curves(
            test_probs, y_test,
            'Evaluation_Results_VariationB/roc_curves_variationB_test.png'
        )

        # 4. Calibration Plots
        train_ece, train_brier = plot_calibration_curves(
            train_probs, y_train,
            'Evaluation_Results_VariationB/calibration_variationB_train.png'
        )

        test_ece, test_brier = plot_calibration_curves(
            test_probs, y_test,
            'Evaluation_Results_VariationB/calibration_variationB_test.png'
        )

        # 5. Summary CSVs
        create_summary_csv(
            train_metrics, train_pr_aucs, train_roc_aucs, train_ece, train_brier,
            'Evaluation_Results_VariationB/summary_variationB_train.csv'
        )

        create_summary_csv(
            test_metrics, test_pr_aucs, test_roc_aucs, test_ece, test_brier,
            'Evaluation_Results_VariationB/summary_variationB_test.csv'
        )

        print("\n🎉 VariationB Classifier evaluation completed successfully!")
        print(f"📁 Results saved in: Evaluation_Results_VariationB/")

    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


