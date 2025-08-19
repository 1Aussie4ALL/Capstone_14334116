import os

def create_summary_report():
    """Create a summary report of the Variation A dataset creation."""
    print("📊 Creating summary report...")
    
    data_used_path = "data_used_for_variation_A"
    results_path = "Variation A_results"
    
    cancer_types = ['glioma', 'meningioma', 'notumor', 'pituitary', 'not_mri']
    
    report = []
    report.append("=" * 60)
    report.append("VARIATION A DATASET CREATION SUMMARY")
    report.append("=" * 60)
    
    total_original = 0
    total_augmented = 0
    
    for cancer_type in cancer_types:
        data_used_dir = os.path.join(data_used_path, cancer_type)
        results_dir = os.path.join(results_path, cancer_type)
        
        original_count = len([f for f in os.listdir(data_used_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) if os.path.exists(data_used_dir) else 0
        augmented_count = len([f for f in os.listdir(results_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) if os.path.exists(results_dir) else 0
        
        total_original += original_count
        total_augmented += augmented_count
        
        report.append(f"{cancer_type.upper():<15}: {original_count:>3} original -> {augmented_count:>3} augmented")
    
    report.append("-" * 60)
    report.append(f"TOTAL{'':<10}: {total_original:>3} original -> {total_augmented:>3} augmented")
    report.append("=" * 60)
    
    # Save report
    with open("Variation_A_Summary_Report.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(report))
    
    # Print report
    for line in report:
        print(line)
    
    print(f"\n📄 Summary report saved to: Variation_A_Summary_Report.txt")
    print(f"📁 Original images: {data_used_path}")
    print(f"🎨 Augmented images: {results_path}")

if __name__ == "__main__":
    create_summary_report()
