"""
CARLA SSD Detection System
Main module for integrating SSD object detection with CARLA simulator
"""

import carla
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
import numpy as np
import cv2
import time
import logging
from PIL import Image
import queue
import threading
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class SSDDetector:
    """SSD Object Detector for CARLA Integration"""
    
    def __init__(self, checkpoint_path: str, num_classes: int = 6, confidence_threshold: float = 0.5):
        """Initialize SSD detector for CARLA"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.class_names = ['background', 'Car', 'Bus', 'Person', 'Truck', 'Motorcycle', 'Bicycle']
        self.colors = {
            1: (0, 255, 0),    # Car - Green
            2: (255, 0, 0),    # Bus - Blue
            3: (0, 255, 255),  # Person - Yellow
            4: (255, 0, 255),  # Truck - Magenta
            5: (255, 255, 0),  # Motorcycle - Cyan
            6: (128, 0, 255)   # Bicycle - Purple
        }
        
        # Create and load model
        self.model = self._create_model(num_classes, checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logging.info(f"SSD Detector initialized on {self.device}")
        logging.info(f"Model loaded from: {checkpoint_path}")
    
    def _create_model(self, num_classes: int, checkpoint_path: str):
        """Create SSD300 model and load checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Create model with same architecture as training
        model = ssd300_vgg16(weights=None)
        
        # Get the original classification head to extract parameters
        original_head = model.head.classification_head
        
        # Extract parameters from original head
        in_channels = []
        num_anchors = []
        
        for module in original_head.module_list:
            in_channels.append(module.in_channels)
            num_anchors_per_layer = module.out_channels // 91  # 91 classes in COCO
            num_anchors.append(num_anchors_per_layer)
        
        # Create new classification head
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes + 1  # +1 for background
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def detect(self, image_array: np.ndarray):
        """Detect objects in image array"""
        # Convert numpy array to PIL Image
        image_pil = Image.fromarray(image_array)
        
        # Preprocess
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Process predictions
        pred = predictions[0]
        
        # Filter by confidence
        keep = pred['scores'] > self.confidence_threshold
        boxes = pred['boxes'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()
        labels = pred['labels'][keep].cpu().numpy()
        
        return boxes, scores, labels
    
    def draw_detections(self, image: np.ndarray, boxes, scores, labels):
        """Draw bounding boxes on image"""
        result_image = image.copy()
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get color and class name
            color = self.colors.get(label, (255, 255, 255))
            class_name = self.class_names[label] if label < len(self.class_names) else f"Class_{label}"
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_text = f"{class_name}: {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(result_image, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(result_image, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image
    
    def get_detection_summary(self, boxes, scores, labels):
        """Get summary of detections"""
        summary = {}
        for label in labels:
            class_name = self.class_names[label] if label < len(self.class_names) else f"Class_{label}"
            summary[class_name] = summary.get(class_name, 0) + 1
        return summary
