import torch
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
from PIL import Image, ImageDraw, ImageFont
import json
import os
import argparse
import numpy as np

def create_model(num_classes: int, checkpoint_path: str = None):
    """Create SSD300 model and load checkpoint"""
    from torchvision.models.detection.ssd import SSD300_VGG16_Weights
    
    # Create model with same architecture as training
    model = ssd300_vgg16(weights=None)  # No pretrained weights for evaluation
    
    # Get the original classification head to extract parameters
    original_head = model.head.classification_head
    
    # Extract the input channels from the first layer of the classification head
    in_channels = []
    num_anchors = []
    
    # Get the input channels and number of anchors from each layer
    for module in original_head.module_list:
        in_channels.append(module.in_channels)
        # Calculate num_anchors from the output channels (assume original was COCO with 91 classes)
        num_anchors_per_layer = module.out_channels // 91  # 91 classes in COCO
        num_anchors.append(num_anchors_per_layer)
    
    # Create new classification head for our custom classes
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1  # +1 for background
    )
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    return model

def predict_image(model, image_path: str, device, confidence_threshold: float = 0.5):
    """Make prediction on a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Transform image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Process predictions
    pred = predictions[0]
    
    # Filter by confidence
    keep = pred['scores'] > confidence_threshold
    boxes = pred['boxes'][keep].cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()
    labels = pred['labels'][keep].cpu().numpy()
    
    return image, boxes, scores, labels

def draw_predictions(image, boxes, scores, labels, class_names):
    """Draw bounding boxes and labels on image"""
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan']
    
    for box, score, label in zip(boxes, scores, labels):
        # Convert box coordinates
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        color = colors[label % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label and score
        if label < len(class_names):
            class_name = class_names[label]
        else:
            class_name = f"Class_{label}"
        
        text = f"{class_name}: {score:.2f}"
        
        # Get text size for background
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw background for text
        draw.rectangle([x1, y1-text_height-5, x1+text_width+5, y1], fill=color)
        draw.text((x1+2, y1-text_height-3), text, fill='white', font=font)
    
    return image

def main():
    parser = argparse.ArgumentParser(description='Evaluate SSD300 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default='prediction.jpg',
                       help='Output image path')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Class names (index 0 is background, so we start from index 1)
    class_names = ['background', 'Car', 'Bus', 'Person', 'Truck', 'Motorcycle', 'Bicycle']
    
    # Create and load model
    model = create_model(num_classes=6, checkpoint_path=args.checkpoint)
    model.to(device)
    
    # Make prediction
    image, boxes, scores, labels = predict_image(
        model, args.image, device, args.confidence
    )
    
    print(f"Found {len(boxes)} detections")
    for box, score, label in zip(boxes, scores, labels):
        class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
        print(f"  {class_name}: {score:.3f} at {box}")
    
    # Draw predictions
    result_image = draw_predictions(image, boxes, scores, labels, class_names)
    
    # Save result
    result_image.save(args.output)
    print(f"Result saved to {args.output}")

if __name__ == '__main__':
    main()
