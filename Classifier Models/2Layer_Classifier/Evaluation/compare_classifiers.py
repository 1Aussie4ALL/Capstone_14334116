import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("🔍 Classifier Comparison: 2Layer vs VariationA")
print("=" * 60)
print("This script compares the performance of both classifiers")
print("=" * 60)

def load_summary_data():
    """Load summary data from both classifiers"""
    print("\n📥 Loading summary data...")
    
    # Try to load 2Layer classifier results
    layer2_train_path = 'Evaluation_Results/summary_train.csv'
    layer2_test_path = 'Evaluation_Results/summary_test.csv'
    
    # Try to load VariationA classifier results
    variationA_train_path = 'Evaluation_Results_VariationA/summary_variationA_train.csv'
    variationA_test_path = 'Evaluation_Results_VariationA/summary_variationA_test.csv'
    
    results = {}
    
    # Load 2Layer results
    if os.path.exists(layer2_train_path) and os.path.exists(layer2_test_path):
        print("✅ Found 2Layer classifier results")
        results['2Layer'] = {
            'train': pd.read_csv(layer2_train_path),
            'test': pd.read_csv(layer2_test_path)
        }
    else:
        print("⚠️  2Layer classifier results not found")
        print("   Expected: summary_train.csv and summary_test.csv")
    
    # Load VariationA results
    if os.path.exists(variationA_train_path) and os.path.exists(variationA_test_path):
        print("✅ Found VariationA classifier results")
        results['VariationA'] = {
            'train': pd.read_csv(variationA_train_path),
            'test': pd.read_csv(variationA_test_path)
        }
    else:
        print("⚠️  VariationA classifier results not found")
        print("   Expected: summary_variationA_train.csv and summary_variationA_test.csv")
    
    return results

def create_comparison_table(results):
    """Create a comparison table of metrics"""
    if not results:
        print("❌ No results to compare")
        return None
    
    print("\n📊 Creating comparison table...")
    
    # Create comparison DataFrame
    comparison_data = []
    
    for classifier_name, data in results.items():
        for dataset_type, df in data.items():
            if not df.empty:
                row = {
                    'Classifier': classifier_name,
                    'Dataset': dataset_type.capitalize(),
                    'Accuracy': df['accuracy'].iloc[0],
                    'Macro_F1': df['macro_F1'].iloc[0],
                    'Weighted_F1': df['weighted_F1'].iloc[0],
                    'Balanced_Accuracy': df['balanced_accuracy'].iloc[0],
                    'Macro_PR_AUC': df['macro_PR_AUC'].iloc[0],
                    'Macro_ROC_AUC': df['macro_ROC_AUC'].iloc[0],
                    'ECE': df['ECE'].iloc[0],
                    'Brier': df['Brier'].iloc[0]
                }
                comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📋 Comparison Table:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison table
        comparison_df.to_csv('Evaluation_Results/classifier_comparison.csv', index=False)
        print(f"\n💾 Comparison table saved to: Evaluation_Results/classifier_comparison.csv")
        
        return comparison_df
    else:
        print("❌ No comparison data available")
        return None

def plot_metric_comparison(comparison_df):
    """Plot comparison of key metrics"""
    if comparison_df is None or comparison_df.empty:
        return
    
    print("\n📈 Creating metric comparison plots...")
    
    # Key metrics to compare
    key_metrics = ['Accuracy', 'Macro_F1', 'Weighted_F1', 'Balanced_Accuracy', 'Macro_PR_AUC', 'Macro_ROC_AUC']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(key_metrics):
        ax = axes[i]
        
        # Filter data for this metric
        metric_data = comparison_df[['Classifier', 'Dataset', metric]]
        
        # Create grouped bar plot
        classifiers = metric_data['Classifier'].unique()
        datasets = metric_data['Dataset'].unique()
        
        x = np.arange(len(datasets))
        width = 0.35
        
        for j, classifier in enumerate(classifiers):
            classifier_data = metric_data[metric_data['Classifier'] == classifier]
            values = [classifier_data[classifier_data['Dataset'] == dataset][metric].iloc[0] 
                     if not classifier_data[classifier_data['Dataset'] == dataset].empty else 0 
                     for dataset in datasets]
            
            ax.bar(x + j*width, values, width, label=classifier, alpha=0.8)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, classifier in enumerate(classifiers):
            classifier_data = metric_data[metric_data['Classifier'] == classifier]
            for k, dataset in enumerate(datasets):
                if not classifier_data[classifier_data['Dataset'] == dataset].empty:
                    value = classifier_data[classifier_data['Dataset'] == dataset][metric].iloc[0]
                    ax.text(x[k] + j*width, value + 0.01, f'{value:.3f}', 
                           ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Classifier Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Evaluation_Results/classifier_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_improvements(comparison_df):
    """Calculate improvement percentages"""
    if comparison_df is None or comparison_df.empty:
        return
    
    print("\n📈 Calculating improvements...")
    
    # Get test set results for both classifiers
    layer2_test = comparison_df[(comparison_df['Classifier'] == '2Layer') & (comparison_df['Dataset'] == 'Test')]
    variationA_test = comparison_df[(comparison_df['Classifier'] == 'VariationA') & (comparison_df['Dataset'] == 'Test')]
    
    if layer2_test.empty or variationA_test.empty:
        print("⚠️  Cannot calculate improvements - missing test data")
        return
    
    # Calculate improvements
    improvements = {}
    metrics = ['Accuracy', 'Macro_F1', 'Weighted_F1', 'Balanced_Accuracy', 'Macro_PR_AUC', 'Macro_ROC_AUC']
    
    for metric in metrics:
        layer2_value = layer2_test[metric].iloc[0]
        variationA_value = variationA_test[metric].iloc[0]
        
        if layer2_value != 0:
            improvement = ((variationA_value - layer2_value) / layer2_value) * 100
            improvements[metric] = improvement
        else:
            improvements[metric] = 0
    
    # Create improvements DataFrame
    improvements_df = pd.DataFrame({
        'Metric': list(improvements.keys()),
        '2Layer_Value': [layer2_test[metric].iloc[0] for metric in metrics],
        'VariationA_Value': [variationA_test[metric].iloc[0] for metric in metrics],
        'Improvement_%': list(improvements.values())
    })
    
    print("\n📊 Performance Improvements (Test Set):")
    print(improvements_df.to_string(index=False, float_format='%.4f'))
    
    # Save improvements
    improvements_df.to_csv('Evaluation_Results/performance_improvements.csv', index=False)
    print(f"\n💾 Improvements saved to: Evaluation_Results/performance_improvements.csv")
    
    return improvements_df

def plot_improvements(improvements_df):
    """Plot performance improvements"""
    if improvements_df is None or improvements_df.empty:
        return
    
    print("\n📊 Creating improvement visualization...")
    
    # Create improvement plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Metric values comparison
    metrics = improvements_df['Metric']
    layer2_values = improvements_df['2Layer_Value']
    variationA_values = improvements_df['VariationA_Value']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, layer2_values, width, label='2Layer Classifier', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, variationA_values, width, label='VariationA Classifier', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Value')
    ax1.set_title('Metric Values Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement percentages
    improvements = improvements_df['Improvement_%']
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax2.bar(metrics, improvements, color=colors, alpha=0.8)
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Performance Improvements')
    ax2.set_xticklabels(metrics, rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if imp > 0 else -0.5), 
                f'{imp:+.2f}%', ha='center', va='bottom' if imp > 0 else 'top')
    
    plt.suptitle('VariationA vs 2Layer Classifier Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Evaluation_Results/performance_improvements_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main comparison function"""
    try:
        print("🔍 Starting classifier comparison...")
        
        # Load results
        results = load_summary_data()
        
        if not results:
            print("❌ No results found to compare")
            print("\n💡 Make sure to run both evaluations first:")
            print("   1. evaluate_2layer_classifier.py")
            print("   2. evaluate_variationA_classifier.py")
            return
        
        # Create comparison table
        comparison_df = create_comparison_table(results)
        
        if comparison_df is not None:
            # Plot metric comparisons
            plot_metric_comparison(comparison_df)
            
            # Calculate and plot improvements
            improvements_df = calculate_improvements(comparison_df)
            plot_improvements(improvements_df)
            
            print("\n🎉 Classifier comparison completed!")
            print(f"📁 All results saved in: Evaluation_Results/")
            
        else:
            print("❌ Comparison failed")
        
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
