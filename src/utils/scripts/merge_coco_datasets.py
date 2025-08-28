#!/usr/bin/env python3
"""
COCO Dataset Merger Script

This script merges multiple COCO datasets from the Scenes folder into a single dataset
with 6,000 randomly selected images, split into 75% train and 25% validation sets.

Usage: python merge_coco_datasets.py
"""

import os
import json
import glob
import random
import shutil
from pathlib import Path
from collections import defaultdict
import argparse

def load_coco_dataset(bbox_file_path):
    """Load a COCO dataset from a bbox.json file."""
    with open(bbox_file_path, 'r') as f:
        data = json.load(f)
    
    # Get the images directory path
    images_dir = os.path.join(os.path.dirname(bbox_file_path), 'images')
    
    return data, images_dir

def create_output_structure(output_dir):
    """Create the output directory structure."""
    output_dir = Path(output_dir)
    
    # Create main directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    
    # Create subdirectories
    for split_dir in [train_dir, val_dir]:
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    
    return train_dir, val_dir

def get_unique_id_mappings(all_datasets):
    """Create mappings for unique image and annotation IDs across all datasets."""
    image_id_mapping = {}
    annotation_id_mapping = {}
    
    current_image_id = 0
    current_annotation_id = 0
    
    for dataset_idx, (data, _) in enumerate(all_datasets):
        # Map image IDs
        for image in data['images']:
            old_id = image['id']
            new_id = current_image_id
            image_id_mapping[(dataset_idx, old_id)] = new_id
            current_image_id += 1
        
        # Map annotation IDs
        for annotation in data['annotations']:
            old_id = annotation['id']
            new_id = current_annotation_id
            annotation_id_mapping[(dataset_idx, old_id)] = new_id
            current_annotation_id += 1
    
    return image_id_mapping, annotation_id_mapping

def generate_unique_filename(original_filename, dataset_path):
    """Generate a unique filename based on the scene and conditions"""
    # Extract scene information from the dataset path
    path_parts = dataset_path.replace('\\', '/').split('/')
    
    scene_info = []
    for part in path_parts:
        if 'Scene' in part:
            scene_info.append(part.replace(' ', '').replace('-', '_'))
        elif part in ['Day', 'Night', 'Foggy', 'Normal', 'Rainy', 'Fog']:
            scene_info.append(part.lower())
    
    # Create prefix from scene information
    scene_prefix = '_'.join(scene_info) if scene_info else 'scene'
    
    # Get file extension
    name, ext = os.path.splitext(original_filename)
    
    # Create new filename: scene_prefix_originalname.ext
    new_filename = f"{scene_prefix}_{name}{ext}"
    
    return new_filename

def merge_coco_datasets(scenes_dir, output_dir, target_images=6000, train_ratio=0.75):
    """
    Merge multiple COCO datasets into train/val splits.
    
    Args:
        scenes_dir: Path to the Scenes directory containing COCO datasets
        output_dir: Path to output directory for merged datasets
        target_images: Total number of images to include (default: 6000)
        train_ratio: Ratio of images for training (default: 0.75)
    """
    
    print(f"Starting COCO dataset merger...")
    print(f"Target images: {target_images}")
    print(f"Train ratio: {train_ratio} ({int(target_images * train_ratio)} train, {int(target_images * (1-train_ratio))} val)")
    
    # Find all COCO datasets
    bbox_files = glob.glob(os.path.join(scenes_dir, '**', 'bbox.json'), recursive=True)
    print(f"Found {len(bbox_files)} COCO datasets:")
    
    # Load all datasets
    all_datasets = []
    all_images_info = []
    
    for i, bbox_file in enumerate(bbox_files):
        print(f"  Loading: {bbox_file}")
        data, images_dir = load_coco_dataset(bbox_file)
        all_datasets.append((data, images_dir))
        
        # Collect all image information with source dataset index
        for image in data['images']:
            all_images_info.append({
                'dataset_idx': i,
                'image_info': image,
                'source_path': os.path.join(images_dir, image['file_name'])
            })
    
    print(f"Total available images: {len(all_images_info)}")
    
    # Check if we have enough images
    if len(all_images_info) < target_images:
        print(f"Warning: Only {len(all_images_info)} images available, using all of them.")
        target_images = len(all_images_info)
    
    # Randomly select target number of images
    random.shuffle(all_images_info)
    selected_images = all_images_info[:target_images]
    
    # Split into train/val
    train_size = int(target_images * train_ratio)
    train_images = selected_images[:train_size]
    val_images = selected_images[train_size:]
    
    print(f"Selected {len(train_images)} images for training")
    print(f"Selected {len(val_images)} images for validation")
    
    # Create output structure
    train_dir, val_dir = create_output_structure(output_dir)
    
    # Get unique ID mappings
    image_id_mapping, annotation_id_mapping = get_unique_id_mappings(all_datasets)
    
    # Process train and validation sets
    for split_name, split_images, split_dir in [('train', train_images, train_dir), ('val', val_images, val_dir)]:
        print(f"\nProcessing {split_name} set...")
        
        # Collect selected image IDs for this split
        selected_image_ids = set()
        for img_info in split_images:
            dataset_idx = img_info['dataset_idx']
            original_image_id = img_info['image_info']['id']
            selected_image_ids.add((dataset_idx, original_image_id))
        
        # Create merged COCO annotation
        merged_coco = {
            'info': {
                'year': 2025,
                'version': '1.0.0',
                'description': f'Merged COCO Dataset - {split_name.capitalize()} Split',
                'contributor': 'COCO Dataset Merger',
                'url': 'Not Set',
                'date_created': '2025-08-26'
            },
            'licenses': [
                {
                    'id': 0,
                    'name': 'No License',
                    'url': 'Not Set'
                }
            ],
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Get categories from the first dataset (assuming all have same categories)
        merged_coco['categories'] = all_datasets[0][0]['categories']
        
        # Add selected images and their annotations
        for img_info in split_images:
            dataset_idx = img_info['dataset_idx']
            original_image = img_info['image_info']
            source_path = img_info['source_path']
            
            # Map to new unique ID
            new_image_id = image_id_mapping[(dataset_idx, original_image['id'])]
            
            # Generate unique filename based on scene information
            original_filename = original_image['file_name']
            dataset_path = bbox_files[dataset_idx]
            scene_filename = generate_unique_filename(original_filename, dataset_path)
            
            # Copy image file
            dest_path = split_dir / 'images' / scene_filename
            
            # Handle potential filename conflicts by adding counter
            counter = 0
            base_name = Path(scene_filename).stem
            extension = Path(scene_filename).suffix
            final_filename = scene_filename
            
            while dest_path.exists():
                counter += 1
                final_filename = f"{base_name}_{counter}{extension}"
                dest_path = split_dir / 'images' / final_filename
            
            # Update image info with new filename
            original_image = original_image.copy()  # Don't modify original
            original_image['file_name'] = final_filename
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                print(f"  Copied: {original_image['file_name']}")
            else:
                print(f"  Warning: Source image not found: {source_path}")
                continue
            
            # Add image to merged dataset
            image_entry = original_image.copy()
            image_entry['id'] = new_image_id
            merged_coco['images'].append(image_entry)
            
            # Add corresponding annotations
            dataset_data = all_datasets[dataset_idx][0]
            for annotation in dataset_data['annotations']:
                if annotation['image_id'] == original_image['id']:
                    # Create new annotation with mapped IDs
                    new_annotation = annotation.copy()
                    new_annotation['id'] = annotation_id_mapping[(dataset_idx, annotation['id'])]
                    new_annotation['image_id'] = new_image_id
                    merged_coco['annotations'].append(new_annotation)
        
        # Save merged COCO annotation file
        annotation_file = split_dir / 'annotations' / f'{split_name}_annotations.json'
        with open(annotation_file, 'w') as f:
            json.dump(merged_coco, f, indent=2)
        
        print(f"  Saved {len(merged_coco['images'])} images and {len(merged_coco['annotations'])} annotations")
        print(f"  Annotation file: {annotation_file}")
    
    print(f"\nDataset merging completed!")
    print(f"Output directory: {output_dir}")
    print(f"Train set: {train_dir}")
    print(f"Val set: {val_dir}")

def main():
    parser = argparse.ArgumentParser(description='Merge COCO datasets from Scenes folder')
    parser.add_argument('--scenes_dir', default='D:/Thesis SSD/Scenes', 
                        help='Path to Scenes directory (default: D:/Thesis SSD/Scenes)')
    parser.add_argument('--output_dir', default='D:/Thesis SSD/merged_dataset', 
                        help='Output directory for merged dataset (default: D:/Thesis SSD/merged_dataset)')
    parser.add_argument('--target_images', type=int, default=6000, 
                        help='Total number of images to include (default: 6000)')
    parser.add_argument('--train_ratio', type=float, default=0.75, 
                        help='Ratio of images for training (default: 0.75)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducible results (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Validate inputs
    if not os.path.exists(args.scenes_dir):
        print(f"Error: Scenes directory not found: {args.scenes_dir}")
        return
    
    if not (0 < args.train_ratio < 1):
        print(f"Error: Train ratio must be between 0 and 1, got: {args.train_ratio}")
        return
    
    if args.target_images <= 0:
        print(f"Error: Target images must be positive, got: {args.target_images}")
        return
    
    # Check if output directory exists and warn about potential overwrites
    if os.path.exists(args.output_dir):
        response = input(f"Output directory {args.output_dir} already exists. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Run the merger
    try:
        merge_coco_datasets(
            scenes_dir=args.scenes_dir,
            output_dir=args.output_dir,
            target_images=args.target_images,
            train_ratio=args.train_ratio
        )
    except Exception as e:
        print(f"Error during merging: {e}")
        return

if __name__ == '__main__':
    main()
