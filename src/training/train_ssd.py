import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
import json
import os
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Any
import argparse
import logging
from datetime import datetime
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class COCODataset(torch.utils.data.Dataset):
    """Custom COCO Dataset for SSD training"""
    
    def __init__(self, annotations_file: str, images_dir: str, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image_id
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # Filter images that have annotations
        self.image_ids = list(self.image_annotations.keys())
        
        logger.info(f"Loaded {len(self.image_ids)} images with annotations")
        logger.info(f"Categories: {[cat['name'] for cat in self.categories.values()]}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        annotations = self.image_annotations[img_id]
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        for ann in annotations:
            # Convert COCO format (x, y, w, h) to (x1, y1, x2, y2)
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Ensure boxes are valid
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_info['width'], x2)
            y2 = min(img_info['height'], y2)
            
            if x2 > x1 and y2 > y1:  # Valid box
                boxes.append([x1, y1, x2, y2])
                # Category IDs start from 0 in your dataset (no need to subtract 1)
                labels.append(ann['category_id'] + 1)  # Add 1 for background class at index 0
        
        if len(boxes) == 0:
            # If no valid boxes, create a dummy background box
            boxes = [[0, 0, 1, 1]]
            labels = [0]  # Background class
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        
        return image, target

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images, targets = zip(*batch)
    return list(images), list(targets)

def create_model(num_classes: int, pretrained: bool = True):
    """Create SSD300 model with custom number of classes"""
    from torchvision.models.detection.ssd import SSD300_VGG16_Weights
    
    # Load pretrained SSD300 with VGG16 backbone using new API
    if pretrained:
        weights = SSD300_VGG16_Weights.COCO_V1
    else:
        weights = None
    
    model = ssd300_vgg16(weights=weights)
    
    # Get the original classification head to extract parameters
    original_head = model.head.classification_head
    
    # Extract the input channels from the first layer of the classification head
    # The classification head has multiple layers for different feature maps
    in_channels = []
    num_anchors = []
    
    # Get the input channels and number of anchors from each layer
    for module in original_head.module_list:
        in_channels.append(module.in_channels)
        # Calculate num_anchors from the output channels (out_channels = num_anchors * num_classes)
        # Original COCO has 91 classes, so we can calculate num_anchors
        num_anchors_per_layer = module.out_channels // 91  # 91 classes in COCO
        num_anchors.append(num_anchors_per_layer)
    
    # Create new classification head for our custom classes
    # num_classes + 1 for background class
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1  # +1 for background
    )
    
    return model

def get_transforms(train: bool = True):
    """Get image transforms for training/validation"""
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def train_one_epoch(model, data_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Log progress
        if batch_idx % 10 == 0:
            logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{num_batches}, '
                       f'Loss: {losses.item():.4f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, data_loader, device):
    """Validate the model"""
    model.train()  # SSD needs to be in train mode for loss calculation
    total_loss = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for images, targets in data_loader:
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'ssd300_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f'Checkpoint saved: {checkpoint_path}')

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        default_config = {
            'num_epochs': 20,
            'batch_size': 8,
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'num_classes': 6,  # Car, Bus, Person, Truck, Motorcycle, Bicycle
            'train_annotations': 'merged_dataset/train/annotations/train_annotations.json',
            'train_images': 'merged_dataset/train/images',
            'val_annotations': 'merged_dataset/val/annotations/val_annotations.json',
            'val_images': 'merged_dataset/val/images',
            'checkpoint_dir': 'checkpoints',
            'output_dir': 'output'
        }
        
        # Save default config
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config

def main():
    parser = argparse.ArgumentParser(description='Train SSD300 on custom dataset')
    parser.add_argument('--config', type=str, default='configs/train_ssd.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f'Loaded config: num_epochs={config["num_epochs"]}, batch_size={config["batch_size"]}')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create datasets
    logger.info('Creating datasets...')
    
    train_dataset = COCODataset(
        annotations_file=config['train_annotations'],
        images_dir=config['train_images'],
        transforms=get_transforms(train=True)
    )
    
    val_dataset = COCODataset(
        annotations_file=config['val_annotations'],
        images_dir=config['val_images'],
        transforms=get_transforms(train=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Validation dataset size: {len(val_dataset)}')
    
    # Create model
    logger.info('Creating model...')
    model = create_model(num_classes=config['num_classes'])
    model.to(device)
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    logger.info('Starting training...')
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, config['num_epochs']):
        logger.info(f'Epoch {epoch}/{config["num_epochs"]-1}')
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        
        logger.info(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, config['checkpoint_dir'])
        
        # Save loss history
        loss_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs': list(range(len(train_losses)))
        }
        
        loss_file = os.path.join(config['output_dir'], 'loss_history.json')
        with open(loss_file, 'w') as f:
            json.dump(loss_history, f, indent=2)
    
    # Save final model
    final_model_path = os.path.join(config['output_dir'], 'ssd300_final.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f'Final model saved: {final_model_path}')
    
    logger.info('Training completed!')

if __name__ == '__main__':
    main()
