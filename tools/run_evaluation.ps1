# PowerShell script for COCO evaluation

Write-Host "Installing pycocotools..." -ForegroundColor Green
pip install pycocotools

Write-Host "Running COCO evaluation..." -ForegroundColor Green
python evaluate_coco_metrics.py `
    --checkpoint output/ssd300_final.pth `
    --annotations merged_dataset/val/annotations/val_annotations.json `
    --images merged_dataset/val/images `
    --output validation_evaluation_report.json `
    --confidence 0.01 `
    --batch-size 8 `
    --num-classes 6

Write-Host "Evaluation complete! Check validation_evaluation_report.json for detailed results." -ForegroundColor Green
Read-Host "Press Enter to continue..."
