import torch
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return model

def predict_batch(model, images, device, confidence_threshold=0.01):
    """Make predictions on a batch of images"""
    model.eval()
    
    # Transform images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Process images
    processed_images = []
    for img in images:
        processed_images.append(transform(img))
    
    # Move to device
    processed_images = [img.to(device) for img in processed_images]
    
    # Make predictions
    with torch.no_grad():
        predictions = model(processed_images)
    
    # Process predictions
    results = []
    for pred in predictions:
        # Filter by confidence
        keep = pred['scores'] > confidence_threshold
        boxes = pred['boxes'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()
        labels = pred['labels'][keep].cpu().numpy()
        
        results.append({
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        })
    
    return results

def convert_to_coco_format(predictions, image_ids, category_mapping):
    """Convert predictions to COCO format for evaluation"""
    coco_predictions = []
    
    for pred, image_id in zip(predictions, image_ids):
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']
        
        for box, score, label in zip(boxes, scores, labels):
            # Skip background class (label 0)
            if label == 0:
                continue
                
            # Convert from xyxy to xywh format
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Map label back to original category ID
            # Labels are 1-indexed (1=Car, 2=Bus, etc.), so subtract 1
            original_category_id = label - 1
            
            coco_prediction = {
                'image_id': int(image_id),
                'category_id': int(original_category_id),
                'bbox': [float(x1), float(y1), float(width), float(height)],
                'score': float(score)
            }
            coco_predictions.append(coco_prediction)
    
    return coco_predictions

def load_dataset_images(annotations_file, images_dir, max_images=None):
    """Load dataset images and metadata"""
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    images = []
    image_ids = []
    
    # Limit number of images if specified
    image_list = coco_data['images']
    if max_images:
        image_list = image_list[:max_images]
    
    for img_info in tqdm(image_list, desc="Loading images"):
        img_path = os.path.join(images_dir, img_info['file_name'])
        if os.path.exists(img_path):
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            images.append(image)
            image_ids.append(img_info['id'])
    
    return images, image_ids

def evaluate_model(model, annotations_file, images_dir, device, 
                  confidence_threshold=0.01, max_images=None, batch_size=8):
    """Evaluate model using COCO metrics"""
    
    # Load ground truth
    logger.info("Loading ground truth annotations...")
    coco_gt = COCO(annotations_file)
    
    # Load images
    logger.info("Loading images...")
    images, image_ids = load_dataset_images(annotations_file, images_dir, max_images)
    
    logger.info(f"Evaluating on {len(images)} images")
    
    # Get category mapping
    categories = coco_gt.dataset['categories']
    category_mapping = {cat['id']: cat['name'] for cat in categories}
    
    # Make predictions in batches
    logger.info("Making predictions...")
    all_predictions = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
        batch_images = images[i:i+batch_size]
        batch_image_ids = image_ids[i:i+batch_size]
        
        # Make predictions
        batch_predictions = predict_batch(model, batch_images, device, confidence_threshold)
        
        # Convert to COCO format
        coco_predictions = convert_to_coco_format(batch_predictions, batch_image_ids, category_mapping)
        all_predictions.extend(coco_predictions)
    
    logger.info(f"Generated {len(all_predictions)} predictions")
    
    # Save predictions to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(all_predictions, f, indent=2)
        predictions_file = f.name
    
    try:
        # Load predictions
        coco_dt = coco_gt.loadRes(predictions_file)
        
        # Run evaluation
        logger.info("Running COCO evaluation...")
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        # Filter to only evaluate on the images we processed
        coco_eval.params.imgIds = image_ids
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract key metrics
        metrics = {
            'mAP': coco_eval.stats[0],  # mAP @ IoU=0.50:0.95
            'mAP_50': coco_eval.stats[1],  # mAP @ IoU=0.50
            'mAP_75': coco_eval.stats[2],  # mAP @ IoU=0.75
            'mAP_small': coco_eval.stats[3],  # mAP for small objects
            'mAP_medium': coco_eval.stats[4],  # mAP for medium objects
            'mAP_large': coco_eval.stats[5],  # mAP for large objects
            'mAR_1': coco_eval.stats[6],  # mAR with 1 detection per image
            'mAR_10': coco_eval.stats[7],  # mAR with 10 detections per image
            'mAR_100': coco_eval.stats[8],  # mAR with 100 detections per image
            'mAR_small': coco_eval.stats[9],  # mAR for small objects
            'mAR_medium': coco_eval.stats[10],  # mAR for medium objects
            'mAR_large': coco_eval.stats[11],  # mAR for large objects
        }
        
        # Per-category evaluation
        logger.info("Per-category results:")
        per_category_metrics = {}
        
        for cat_id, cat_name in category_mapping.items():
            coco_eval_cat = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval_cat.params.imgIds = image_ids
            coco_eval_cat.params.catIds = [cat_id]
            coco_eval_cat.evaluate()
            coco_eval_cat.accumulate()
            
            # Check if we have valid evaluation results for this category
            if coco_eval_cat.eval['precision'].size > 0:
                # Get precision array for this category
                # Shape: [TxRxKxAxM] where T=IoU thresholds, R=recall thresholds, K=categories, A=areas, M=maxDets
                precision = coco_eval_cat.eval['precision']
                
                if precision.shape[2] > 0:  # Check if we have categories in results
                    # Calculate AP metrics for the first (and only) category in this evaluation
                    cat_idx = 0  # Index in the filtered evaluation (not the original cat_id)
                    
                    # AP@0.5 (IoU threshold index 0)
                    ap_50_vals = precision[0, :, cat_idx, 0, 2]  # area=all, maxDets=100
                    ap_50 = np.mean(ap_50_vals[ap_50_vals > -1])  # Exclude -1 values
                    
                    # AP@0.75 (IoU threshold index 5 if available)
                    if precision.shape[0] > 5:
                        ap_75_vals = precision[5, :, cat_idx, 0, 2]
                        ap_75 = np.mean(ap_75_vals[ap_75_vals > -1])
                    else:
                        ap_75 = 0.0
                    
                    # Overall AP (average across all IoU thresholds)
                    ap_avg_vals = precision[:, :, cat_idx, 0, 2]  # area=all, maxDets=100
                    valid_vals = ap_avg_vals[ap_avg_vals > -1]
                    ap_avg = np.mean(valid_vals) if len(valid_vals) > 0 else 0.0
                else:
                    ap_50 = ap_75 = ap_avg = 0.0
            else:
                ap_50 = ap_75 = ap_avg = 0.0
            
            per_category_metrics[cat_name] = {
                'AP': ap_avg,
                'AP_50': ap_50,
                'AP_75': ap_75
            }
            
            logger.info(f"{cat_name}: AP={ap_avg:.3f}, AP@50={ap_50:.3f}, AP@75={ap_75:.3f}")
        
        return metrics, per_category_metrics, coco_eval
        
    finally:
        # Clean up temporary file
        if os.path.exists(predictions_file):
            os.unlink(predictions_file)

def save_evaluation_report(metrics, per_category_metrics, output_file):
    """Save evaluation results to JSON file"""
    report = {
        'overall_metrics': metrics,
        'per_category_metrics': per_category_metrics,
        'evaluation_timestamp': str(np.datetime64('now'))
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate SSD300 model with COCO metrics')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to COCO annotations file')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to images directory')
    parser.add_argument('--output', type=str, default='evaluation_report.json',
                       help='Output evaluation report file')
    parser.add_argument('--confidence', type=float, default=0.01,
                       help='Confidence threshold for predictions')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to evaluate (for testing)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--num-classes', type=int, default=6,
                       help='Number of classes in the model')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create and load model
    logger.info('Loading model...')
    model = create_model(num_classes=args.num_classes, checkpoint_path=args.checkpoint)
    model.to(device)
    
    # Run evaluation
    metrics, per_category_metrics, coco_eval = evaluate_model(
        model=model,
        annotations_file=args.annotations,
        images_dir=args.images,
        device=device,
        confidence_threshold=args.confidence,
        max_images=args.max_images,
        batch_size=args.batch_size
    )
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"mAP @ IoU=0.50:0.95: {metrics['mAP']:.3f}")
    logger.info(f"mAP @ IoU=0.50:      {metrics['mAP_50']:.3f}")
    logger.info(f"mAP @ IoU=0.75:      {metrics['mAP_75']:.3f}")
    logger.info(f"mAR @ 100 dets:      {metrics['mAR_100']:.3f}")
    
    # Save report
    save_evaluation_report(metrics, per_category_metrics, args.output)
    
    logger.info("Evaluation completed!")

if __name__ == '__main__':
    main()
