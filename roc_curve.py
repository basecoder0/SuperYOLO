import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import auc

# Configuration


# EXPERIMENTS = {
#     'roc': 'runs/test/test_RGB_noSR_300ep_img_512/best_predictions.json',
# }
# GROUND_TRUTH_DIR = "dataset/VEDAI/labels"
# IMAGES_DIR = "dataset/VEDAI/images"
# OUTPUT_DIR = "runs/roc/VEDAI_RGB_noSR_300ep_img_512"
# IMAGE_WIDTH = 512  # VEDAI original image width
# IMAGE_HEIGHT = 512  # VEDAI original image height



# EXPERIMENTS = {
#     'roc': 'runs/test/test_RGB_llvip_noSR_300ep_img_512/best_predictions.json',
# }
# GROUND_TRUTH_DIR = "dataset/LLVIP/labels"
# IMAGES_DIR = "dataset/LLVIP/images"
# OUTPUT_DIR = "runs/roc/LLVIP_RGB_noSR_300ep_img_512"
# IMAGE_WIDTH = 1280  # LLVIP original image width
# IMAGE_HEIGHT = 1024  # LLVIP original image height



EXPERIMENTS = {
    'roc': 'runs/test/test_RGB_m3fd_noSR_300ep_img_512/best_predictions.json',
}
GROUND_TRUTH_DIR = "dataset/M3FD/labels"
IMAGES_DIR = "dataset/M3FD/images"
OUTPUT_DIR = "runs/roc/M3FD_RGB_noSR_300ep_img_512"
IMAGE_WIDTH = 1024 # M3FD original image width
IMAGE_HEIGHT = 780  # M3FD original image height

TOTAL_IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT  # square pixels
IOU_THRESHOLD = 0.5  # Fixed IoU threshold

# Define multiple confidence thresholds to compare
CONFIDENCE_THRESHOLDS = [0.2, 0.7]

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def count_total_images(gt_dir):
    """Count total number of images in the dataset"""
    image_files = set()
    for filename in os.listdir(gt_dir):
        if filename.endswith('.txt'):
            base_name = os.path.splitext(filename)[0]
            if base_name.endswith('_co') or base_name.endswith('_ir'):
                base_name = base_name[:-3]
            image_files.add(base_name)
    return len(image_files)

def load_ground_truth(gt_dir, image_id):
    """Load ground truth boxes from YOLO txt file"""
    if image_id.endswith('_co') or image_id.endswith('_ir'):
        base_name = image_id[:-3]
    else:
        base_name = f"{image_id:05d}"
    gt_file = os.path.join(gt_dir, f"{base_name}.txt")

    gt_boxes = []
    if os.path.exists(gt_file):
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    x_center = float(parts[1]) * IMAGE_WIDTH
                    y_center = float(parts[2]) * IMAGE_HEIGHT
                    width = float(parts[3]) * IMAGE_WIDTH
                    height = float(parts[4]) * IMAGE_HEIGHT
                    gt_boxes.append({
                        'bbox': [x_center - width / 2, y_center - height / 2,
                                 x_center + width / 2, y_center + height / 2],
                        'used': False
                    })
    return gt_boxes

def bbox_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def evaluate_predictions(predictions, gt_dir, all_test_images, confidence_threshold=0.2, iou_threshold=0.5, no_threshold=False):
    """Evaluate predictions for detection only (ignores class_id)."""
    pred_by_image = defaultdict(list)
    
    # Filter predictions by confidence threshold
    if no_threshold:
        filtered_predictions = predictions
    else:
        filtered_predictions = [pred for pred in predictions if pred['score'] >= confidence_threshold]
    print(f"Filtered predictions: {len(filtered_predictions)}/{len(predictions)} "
          f"({len(filtered_predictions)/len(predictions)*100:.1f}%) with confidence ≥ {confidence_threshold}")
    
    # Group filtered predictions by image
    for pred in filtered_predictions:
        pred_by_image[pred['image_id']].append(pred)
    
    images_with_predictions = set(pred_by_image.keys())
    
    print(f"Images with predictions after filtering: {len(images_with_predictions)}")

    scores = []
    tp_flags = []
    total_gt = 0
    total_tp_possible = 0  # Track maximum possible TPs

    # Process ALL test images (not just those with predictions after filtering)
    for image_id in all_test_images:
        gt_boxes = load_ground_truth(gt_dir, image_id)
        total_gt += len(gt_boxes)
        total_tp_possible += len(gt_boxes)  # Each GT can produce at most one TP
        
        # Get predictions for this image (may be empty after filtering)
        preds = pred_by_image.get(image_id, [])
        
        # Skip images with no predictions after filtering (but still counted GT above)
        if not preds:
            continue

        # DEBUG: Print first image details
        if image_id == list(images_with_predictions)[0]:
            print(f"\nDEBUG - First image: {image_id}")
            print(f"  Sample prediction bbox (raw): {preds[0]['bbox']}")
            print(f"  Sample prediction bbox (converted): {bbox_to_xyxy(preds[0]['bbox'])}")
            if gt_boxes:
                print(f"  Sample GT bbox: {gt_boxes[0]['bbox']}")

        # Sort predictions by confidence (high to low)
        sorted_preds = sorted(preds, key=lambda x: -x['score'])
        
        for pred in sorted_preds:
            pred_box = bbox_to_xyxy(pred['bbox'])
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_boxes):
                if not gt['used']:
                    current_iou = calculate_iou(pred_box, gt['bbox'])
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_gt_idx = i

            # Use the specified IoU threshold            
            is_tp = best_iou >= iou_threshold
            scores.append(pred['score'])
            tp_flags.append(is_tp)

            if is_tp and best_gt_idx != -1:
                gt_boxes[best_gt_idx]['used'] = True
                

    return scores, tp_flags, total_gt, total_tp_possible, len(all_test_images)

def calculate_fa_per_pixel_metrics(scores, tp_flags, processed_images_count, total_tp_possible):
    """Calculate FA per square pixel and TPR for ROC curve."""
    if len(scores) == 0:
        print("No scores to calculate metrics!")
        return np.array([0]), np.array([0])
    
    # Sort by confidence descending
    sorted_indices = np.argsort(-np.array(scores))
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_tp_flags = np.array(tp_flags)[sorted_indices]
    
    # Calculate cumulative FPs and TPs
    fp_cumsum = np.cumsum([not x for x in sorted_tp_flags])
    tp_cumsum = np.cumsum(sorted_tp_flags)
    
    # Calculate FA per square pixel - USE PROCESSED IMAGES COUNT, NOT TOTAL IMAGES
    total_processed_area_pixels = TOTAL_IMAGE_AREA * processed_images_count
    fa_per_pixel = fp_cumsum / total_processed_area_pixels
    
    # Calculate TPR using total possible TPs (total GT objects)
    tpr = tp_cumsum / total_tp_possible if total_tp_possible > 0 else np.zeros_like(tp_cumsum)
    
    # Add (0,0) point for complete curve
    fa_per_pixel = np.concatenate(([0], fa_per_pixel))
    tpr = np.concatenate(([0], tpr))
    
    return fa_per_pixel, tpr

def calculate_normalized_auc(fa_per_pixel, tpr):
    """Calculate normalized AUC that works with FA per pixel values."""
    if len(fa_per_pixel) < 2:
        return 0.0
    
    # Normalize FA per pixel to [0, 1] range for AUC calculation
    max_fa = fa_per_pixel[-1] if fa_per_pixel[-1] > 0 else 1.0
    normalized_fa = fa_per_pixel / max_fa
    
    # Calculate AUC using normalized values
    roc_auc = auc(normalized_fa, tpr)
    
    return roc_auc

def plot_multiple_confidence_thresholds(conf_results, output_dir, iou_threshold):
    """Plot ROC curves for multiple confidence thresholds on the same figure"""
    ensure_dir(output_dir)
    
    valid_results = {k: v for k, v in conf_results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to plot!")
        return
        
    colors = ['blue', 'red']  # Different colors for different confidence thresholds
    line_styles = ['-', '--']  # Different line styles

    plt.figure(figsize=(12, 8))
    
    for (conf_thresh, result_data), color, line_style in zip(
        valid_results.items(), colors, line_styles):
        
        # Unpack the result data
        fa_per_pixel, tpr, total_tp_possible, processed_images, tp_count = result_data
        
        if len(fa_per_pixel) > 0 and len(tpr) > 0:
            # Calculate normalized AUC
            roc_auc = calculate_normalized_auc(fa_per_pixel, tpr)
            
            # Plot the curve
            label = f'Conf ≥ {conf_thresh} (AUC = {roc_auc:.3f})' if conf_thresh != 'no_threshold' else f'No Threshold (AUC = {roc_auc:.3f})'
            plt.plot(fa_per_pixel, tpr, color=color, linestyle=line_style,
                     label=label,
                     linewidth=2)
            
            print(f"Confidence {conf_thresh}: Normalized AUC = {roc_auc:.3f}")
            print(f"Confidence {conf_thresh}: Max FA/pixel = {fa_per_pixel[-1]:.2e}")
            print(f"Confidence {conf_thresh}: Max TPR = {tpr[-1]:.3f}")
            print(f"Confidence {conf_thresh}: Total possible TPs = {total_tp_possible}")
            print(f"Confidence {conf_thresh}: Processed images = {processed_images}")
                
        else:
            print(f"Skipping Confidence {conf_thresh} - no valid data points")

    plt.xscale('log')
    plt.xlabel('False Alarms per Square Pixel (log scale)')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - Different Confidence Thresholds\n(IoU threshold: {iou_threshold})')
    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Set reasonable x-axis limits for FA per pixel
    plt.xlim([1e-9, 1e-5])

    roc_path = os.path.join(output_dir, f'roc_multiple_confidence_thresholds_iou_{iou_threshold}.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multiple confidence thresholds plot to: {roc_path}")

def main(opt):
    ensure_dir(OUTPUT_DIR)
        
    # Count total images for FA per pixel calculation
    total_images = count_total_images(GROUND_TRUTH_DIR)
    print(f"Total images in dataset: {total_images}")
    print(f"IoU Threshold: {IOU_THRESHOLD}")
    print(f"Confidence Thresholds to compare: {CONFIDENCE_THRESHOLDS if not opt.no_threshold else 'No thresholding (all predictions)'}")

    for exp_name, pred_file in EXPERIMENTS.items():
        print(f"\n{'='*50}")
        print(f"Processing experiment: {exp_name}")
        print(f"{'='*50}")

        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file {pred_file} not found. Skipping {exp_name}.")
            continue
            
        try:
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
            print(f"Loaded {len(predictions)} predictions from {pred_file}")
        except Exception as e:
            print(f"Error loading predictions from {pred_file}: {e}")
            continue

        # Get all unique test images from predictions (this is the test set)
        all_test_images = sorted(set(pred['image_id'] for pred in predictions))
        print(f"Test set contains {len(all_test_images)} unique images")

        # Evaluate with multiple confidence thresholds
        conf_results = {}

        if opt.no_threshold:
            print("\n--- Evaluating with NO confidence threshold (all predictions) ---")
            scores, tp_flags, total_gt, total_tp_possible, processed_images_count = evaluate_predictions(
                predictions, GROUND_TRUTH_DIR, all_test_images, 
                confidence_threshold=0.0,  # No threshold
                iou_threshold=IOU_THRESHOLD,
                no_threshold=opt.no_threshold
            )
            if len(scores) > 0:
                fa_per_pixel, tpr = calculate_fa_per_pixel_metrics(scores, tp_flags, processed_images_count, total_tp_possible)
                tp_count = sum(tp_flags)
                conf_results['no_threshold'] = (fa_per_pixel, tpr, total_tp_possible, processed_images_count, tp_count)
                
                print(f"Statistics (No Threshold):")
                print(f"  Total predictions: {len(scores)}")
                print(f"  True positives: {tp_count}")
                print(f"  False positives: {len(scores) - tp_count}")
                print(f"  Total GT objects: {total_gt}")
                print(f"  Total possible TPs: {total_tp_possible}")
                print(f"  Processed images: {processed_images_count}")
                print(f"  Max FA/pixel: {fa_per_pixel[-1]:.2e}")
                print(f"  Max TPR: {tpr[-1]:.3f}")
                
                precision = tp_count / len(scores) if len(scores) > 0 else 0
                recall = tp_count / total_tp_possible if total_tp_possible > 0 else 0
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
            else:
                conf_results['no_threshold'] = None
                print("No predictions found with no thresholding")
        else:        
            for confidence_threshold in CONFIDENCE_THRESHOLDS:
                print(f"\n--- Evaluating with confidence threshold: {confidence_threshold} ---")
                
                # Evaluate with current confidence threshold
                scores, tp_flags, total_gt, total_tp_possible, processed_images_count = evaluate_predictions(
                    predictions, GROUND_TRUTH_DIR, all_test_images, 
                    confidence_threshold=confidence_threshold,
                    iou_threshold=IOU_THRESHOLD,
                )
                if len(scores) > 0:
                    # Calculate FA per pixel metrics
                    fa_per_pixel, tpr = calculate_fa_per_pixel_metrics(scores, tp_flags, processed_images_count, total_tp_possible)
                    
                    # Calculate true positive count
                    tp_count = sum(tp_flags)
                    
                    # Store all necessary data for plotting
                    conf_results[confidence_threshold] = (fa_per_pixel, tpr, total_tp_possible, processed_images_count, tp_count)
                    
                    # Print statistics
                    print(f"Statistics (Conf ≥ {confidence_threshold}):")
                    print(f"  Total predictions: {len(scores)}")
                    print(f"  True positives: {tp_count}")
                    print(f"  False positives: {len(scores) - tp_count}")
                    print(f"  Total GT objects: {total_gt}")
                    print(f"  Total possible TPs: {total_tp_possible}")
                    print(f"  Processed images: {processed_images_count}")
                    print(f"  Max FA/pixel: {fa_per_pixel[-1]:.2e}")
                    print(f"  Max TPR: {tpr[-1]:.3f}")
                    
                    # Calculate precision and recall
                    precision = tp_count / len(scores) if len(scores) > 0 else 0
                    recall = tp_count / total_tp_possible if total_tp_possible > 0 else 0
                    print(f"  Precision: {precision:.3f}")
                    print(f"  Recall: {recall:.3f}")
                        
                else:
                    conf_results[confidence_threshold] = None
                    print(f"No predictions found with confidence threshold {confidence_threshold}")

        # Plot results for all confidence thresholds
        plot_multiple_confidence_thresholds(conf_results, OUTPUT_DIR, IOU_THRESHOLD)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-threshold', action='store_true', help='Disable confidence thresholding')
    opt = parser.parse_args()
    main(opt)