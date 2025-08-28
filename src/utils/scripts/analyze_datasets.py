#!/usr/bin/env python3
"""
Quick dataset analyzer to check COCO datasets before merging.
"""

import os
import json
import glob

def analyze_datasets(scenes_dir):
    """Analyze all COCO datasets in the scenes directory."""
    
    print("=== COCO Dataset Analysis ===\n")
    
    # Find all COCO datasets
    bbox_files = glob.glob(os.path.join(scenes_dir, '**', 'bbox.json'), recursive=True)
    
    if not bbox_files:
        print(f"No COCO datasets found in: {scenes_dir}")
        return
    
    total_images = 0
    total_annotations = 0
    all_categories = set()
    
    print(f"Found {len(bbox_files)} COCO datasets:\n")
    
    for i, bbox_file in enumerate(bbox_files, 1):
        try:
            with open(bbox_file, 'r') as f:
                data = json.load(f)
            
            images_count = len(data.get('images', []))
            annotations_count = len(data.get('annotations', []))
            categories = data.get('categories', [])
            
            total_images += images_count
            total_annotations += annotations_count
            
            # Collect category names
            for cat in categories:
                all_categories.add(cat['name'])
            
            # Check if images directory exists
            images_dir = os.path.join(os.path.dirname(bbox_file), 'images')
            images_exist = os.path.exists(images_dir)
            
            if images_exist:
                actual_image_files = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            else:
                actual_image_files = 0
            
            print(f"{i}. {bbox_file}")
            print(f"   Images in JSON: {images_count}")
            print(f"   Actual image files: {actual_image_files}")
            print(f"   Annotations: {annotations_count}")
            print(f"   Categories: {len(categories)}")
            print(f"   Images dir exists: {images_exist}")
            
            if images_count != actual_image_files:
                print(f"   ⚠️  WARNING: Mismatch between JSON entries and actual files!")
            
            print()
            
        except Exception as e:
            print(f"Error reading {bbox_file}: {e}\n")
    
    print("=== Summary ===")
    print(f"Total datasets: {len(bbox_files)}")
    print(f"Total images: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"Unique categories: {sorted(all_categories)}")
    print(f"Average images per dataset: {total_images / len(bbox_files):.1f}")
    print(f"Average annotations per image: {total_annotations / total_images:.1f}")
    
    print(f"\n=== Merge Simulation ===")
    print(f"If selecting 6,000 images randomly:")
    print(f"  - Training set: 4,500 images (75%)")
    print(f"  - Validation set: 1,500 images (25%)")
    
    if total_images < 6000:
        print(f"  ⚠️  WARNING: Only {total_images} images available, less than target 6,000!")
    else:
        print(f"  ✅ Sufficient images available ({total_images} >= 6,000)")

if __name__ == '__main__':
    scenes_dir = 'D:/Thesis SSD/Scenes'
    analyze_datasets(scenes_dir)
