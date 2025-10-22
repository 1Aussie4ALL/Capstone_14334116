import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter
import random
from scipy import ndimage
import matplotlib.pyplot as plt

def create_challenging_augmentations(image_path, output_dir="challenging_test_images"):
    """
    Create challenging augmented MRI scans that will really test all 3 models
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Convert to RGB for PIL operations
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"Creating challenging augmented images from: {base_name}")
    
    # 1. EXTREME BLUR - Test model robustness to image quality
    print("Creating extreme blur variations...")
    for blur_level in [15, 25, 35]:
        blurred = cv2.GaussianBlur(img, (blur_level, blur_level), 0)
        cv2.imwrite(f"{output_dir}/{base_name}_extreme_blur_{blur_level}.jpg", blurred)
    
    # 2. NOISE CORRUPTION - Test noise robustness
    print("Creating noise corruption variations...")
    for noise_level in [0.1, 0.2, 0.3]:
        noise = np.random.normal(0, noise_level * 255, img.shape).astype(np.uint8)
        noisy_img = cv2.add(img, noise)
        cv2.imwrite(f"{output_dir}/{base_name}_noise_{int(noise_level*100)}.jpg", noisy_img)
    
    # 3. ROTATION CHALLENGES - Test geometric robustness
    print("Creating rotation challenges...")
    for angle in [15, 30, 45, -15, -30, -45]:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h), borderValue=(0, 0, 0))
        cv2.imwrite(f"{output_dir}/{base_name}_rotated_{angle}.jpg", rotated)
    
    # 4. BRIGHTNESS/CONTRAST EXTREMES - Test photometric robustness
    print("Creating brightness/contrast extremes...")
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Very dark
    enhancer = ImageEnhance.Brightness(pil_img)
    dark_img = enhancer.enhance(0.3)
    cv2.imwrite(f"{output_dir}/{base_name}_very_dark.jpg", cv2.cvtColor(np.array(dark_img), cv2.COLOR_RGB2BGR))
    
    # Very bright
    bright_img = enhancer.enhance(2.0)
    cv2.imwrite(f"{output_dir}/{base_name}_very_bright.jpg", cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2BGR))
    
    # Low contrast
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    low_contrast = contrast_enhancer.enhance(0.3)
    cv2.imwrite(f"{output_dir}/{base_name}_low_contrast.jpg", cv2.cvtColor(np.array(low_contrast), cv2.COLOR_RGB2BGR))
    
    # High contrast
    high_contrast = contrast_enhancer.enhance(2.5)
    cv2.imwrite(f"{output_dir}/{base_name}_high_contrast.jpg", cv2.cvtColor(np.array(high_contrast), cv2.COLOR_RGB2BGR))
    
    # 5. CROPPING CHALLENGES - Test partial image recognition
    print("Creating cropping challenges...")
    h, w = img.shape[:2]
    
    # Center crop (75% of original)
    crop_size = int(min(h, w) * 0.75)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    cropped = img[start_h:start_h+crop_size, start_w:start_w+crop_size]
    cv2.imwrite(f"{output_dir}/{base_name}_center_crop_75.jpg", cropped)
    
    # Corner crop
    corner_crop = img[:crop_size, :crop_size]
    cv2.imwrite(f"{output_dir}/{base_name}_corner_crop.jpg", corner_crop)
    
    # 6. SCALING CHALLENGES - Test different resolutions
    print("Creating scaling challenges...")
    for scale in [0.5, 0.75, 1.5, 2.0]:
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f"{output_dir}/{base_name}_scale_{scale}.jpg", scaled)
    
    # 7. MIXED AUGMENTATIONS - Combine multiple challenges
    print("Creating mixed augmentation challenges...")
    
    # Blur + Noise
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    noise = np.random.normal(0, 0.15 * 255, blurred.shape).astype(np.uint8)
    mixed1 = cv2.add(blurred, noise)
    cv2.imwrite(f"{output_dir}/{base_name}_blur_noise_mixed.jpg", mixed1)
    
    # Rotation + Brightness
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 20, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h), borderValue=(0, 0, 0))
    pil_rotated = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_rotated)
    bright_rotated = enhancer.enhance(1.5)
    cv2.imwrite(f"{output_dir}/{base_name}_rotated_bright_mixed.jpg", cv2.cvtColor(np.array(bright_rotated), cv2.COLOR_RGB2BGR))
    
    # 8. EDGE CASES - Very challenging scenarios
    print("Creating edge case challenges...")
    
    # Heavily compressed (simulate poor quality)
    cv2.imwrite(f"{output_dir}/{base_name}_compressed.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 20])
    
    # Salt and pepper noise
    salt_pepper = img.copy()
    noise = np.random.random(img.shape[:2])
    salt_pepper[noise < 0.05] = 0  # Salt
    salt_pepper[noise > 0.95] = 255  # Pepper
    cv2.imwrite(f"{output_dir}/{base_name}_salt_pepper.jpg", salt_pepper)
    
    # Motion blur simulation
    kernel = np.zeros((9, 9))
    kernel[4, :] = np.ones(9) / 9
    motion_blurred = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(f"{output_dir}/{base_name}_motion_blur.jpg", motion_blurred)
    
    print(f"✅ Created {len(os.listdir(output_dir))} challenging test images in {output_dir}/")
    print("\n🎯 These images will test:")
    print("   • Model robustness to image quality")
    print("   • Geometric transformation handling")
    print("   • Photometric variation adaptation")
    print("   • Noise and corruption resistance")
    print("   • Partial image recognition")
    print("   • Resolution scaling capabilities")

def create_test_suite():
    """
    Create a comprehensive test suite from existing images
    """
    # Find existing test images
    test_images = []
    
    # Check various directories for test images
    directories_to_check = [
        "uploads",
        "../uploads", 
        "../../uploads",
        "../../Beta/uploads",
        "../../Classifier Models/WebAPP/uploads"
    ]
    
    for directory in directories_to_check:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(directory, file))
    
    if not test_images:
        print("❌ No test images found. Please add some MRI images to test with.")
        return
    
    print(f"Found {len(test_images)} test images:")
    for img in test_images:
        print(f"  • {img}")
    
    # Create challenging augmentations for each image
    for img_path in test_images[:3]:  # Limit to first 3 images to avoid too many files
        if os.path.exists(img_path):
            create_challenging_augmentations(img_path)
    
    print(f"\n🚀 Test suite created! Upload these images to test your 3 models:")
    print(f"   📁 Check the 'challenging_test_images' folder")
    print(f"   🧠 These will really stress-test Original, VariationA, and VariationB models!")

if __name__ == "__main__":
    print("🧠 Creating Challenging Augmented MRI Test Images")
    print("=" * 50)
    create_test_suite()
