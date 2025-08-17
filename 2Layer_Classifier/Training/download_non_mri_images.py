import os
import requests
from PIL import Image
import io
import time
import random

def download_image(url, save_path):
    """Download image from URL and save it"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Open and verify it's a valid image
        img = Image.open(io.BytesIO(response.content))
        img.verify()
        
        # Reopen and save
        img = Image.open(io.BytesIO(response.content))
        img.save(save_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {str(e)}")
        return False

def download_non_mri_images():
    """Download diverse non-MRI images for training"""
    print("🚀 Downloading additional non-MRI images...")
    
    # Create non_mri folder if it doesn't exist
    save_dir = "Dataset/Training/not_mri"
    os.makedirs(save_dir, exist_ok=True)
    
    # Diverse non-MRI image URLs (free stock photos)
    image_urls = [
        # Nature
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",
        
        # Buildings
        "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1514565131-fce0801e5785?w=400&h=400&fit=crop",
        
        # People
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=400&h=400&fit=crop",
        
        # Animals
        "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1558788353-fb6ae47327e9?w=400&h=400&fit=crop",
        
        # Food
        "https://images.unsplash.com/photo-1504674900240-9c9c0b1b0b1b?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400&h=400&fit=crop",
        
        # Technology
        "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1518186285589-2f7649de83e0?w=400&h=400&fit=crop",
        
        # Objects
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",
        
        # Vehicles
        "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=400&h=400&fit=crop"
    ]
    
    print(f"📥 Found {len(image_urls)} images to download")
    print(f"💾 Saving to: {save_dir}")
    
    successful_downloads = 0
    
    for i, url in enumerate(image_urls):
        # Generate unique filename
        filename = f"downloaded_non_mri_{i+1:03d}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        print(f"  Downloading {i+1}/{len(image_urls)}: {filename}")
        
        if download_image(url, save_path):
            successful_downloads += 1
            print(f"    ✅ Downloaded successfully")
        else:
            print(f"    ❌ Download failed")
        
        # Small delay to be respectful to servers
        time.sleep(0.5)
    
    print(f"\n🎉 Download completed!")
    print(f"✅ Successfully downloaded: {successful_downloads}/{len(image_urls)} images")
    print(f"📁 Images saved to: {save_dir}")
    
    # Count total non-MRI images
    total_images = len([f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"📊 Total non-MRI images now: {total_images}")

if __name__ == "__main__":
    try:
        download_non_mri_images()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Make sure you have internet connection and the 'requests' library installed")
        print("💡 Install with: pip install requests pillow")
