@echo off
echo Installing pycocotools...
pip install pycocotools

echo Running COCO evaluation...
python evaluate_coco_metrics.py --checkpoint output/ssd300_final.pth --annotations merged_dataset/val/annotations/val_annotations.json --images merged_dataset/val/images --output validation_evaluation_report.json --confidence 0.01 --batch-size 8 --num-classes 6

echo Evaluation complete! Check validation_evaluation_report.json for detailed results.
pause
