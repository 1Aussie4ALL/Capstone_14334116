import os

def count_images_in_directory(directory_path):
    """Count images in a directory and its subdirectories"""
    if not os.path.exists(directory_path):
        return 0, {}
    
    total_count = 0
    class_counts = {}
    
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            class_name = item
            count = 0
            for filename in os.listdir(item_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    count += 1
            class_counts[class_name] = count
            total_count += count
    
    return total_count, class_counts

def main():
    """Compare original dataset vs combined dataset"""
    print("🔍 DATASET COMPARISON: Original vs VariationA")
    print("=" * 60)
    
    # Count original dataset
    original_path = "Dataset/Training"
    original_total, original_counts = count_images_in_directory(original_path)
    
    # Count Variation A dataset
    variation_a_path = "../Photometric Augmentation - Variation A/Variation A_results_800"
    variation_a_total, variation_a_counts = count_images_in_directory(variation_a_path)
    
    # Calculate combined totals
    combined_total = original_total + variation_a_total
    combined_counts = {}
    
    for class_name in original_counts.keys():
        original_count = original_counts.get(class_name, 0)
        variation_count = variation_a_counts.get(class_name, 0)
        combined_counts[class_name] = original_count + variation_count
    
    # Print results
    print(f"\n📁 ORIGINAL DATASET: {original_total:,} images")
    print("-" * 40)
    for class_name, count in original_counts.items():
        print(f"  {class_name:12}: {count:>6,} images")
    
    print(f"\n🎨 VARIATION A DATASET: {variation_a_total:,} images")
    print("-" * 40)
    for class_name, count in variation_a_counts.items():
        print(f"  {class_name:12}: {count:>6,} images")
    
    print(f"\n🚀 COMBINED DATASET: {combined_total:,} images")
    print("-" * 40)
    for class_name, count in combined_counts.items():
        print(f"  {class_name:12}: {count:>6,} images")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Original Dataset:     {original_total:>6,} images")
    print(f"  Variation A Dataset:  {variation_a_total:>6,} images")
    print(f"  Combined Total:       {combined_total:>6,} images")
    print(f"  Increase:             {variation_a_total:>6,} images (+{variation_a_total/original_total*100:.1f}%)")
    
    print(f"\n🎯 READY FOR VARIATIONA CLASSIFIER TRAINING!")
    print(f"   This will train on {combined_total:,} total images")
    print(f"   Expected training time: 4-8 hours")
    print(f"   Model output: mri_variationA_classifier.h5")

if __name__ == "__main__":
    main()
