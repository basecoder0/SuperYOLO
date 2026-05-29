import argparse
import json
import os
import shutil
import glob
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



# EXPERIMENTS = {
#     'roc': 'runs/test/test_RGB_m3fd_noSR_300ep_img_512/best_predictions.json',
# }
# GROUND_TRUTH_DIR = "dataset/M3FD/labels"
# IMAGES_DIR = "dataset/M3FD/images"
# OUTPUT_DIR = "runs/roc/M3FD_RGB_noSR_300ep_img_512"
# IMAGE_WIDTH = 1024 # M3FD original image width
# IMAGE_HEIGHT = 780  # M3FD original image height

DATASET_NAME = "VEDAI"  # Change this to VEDAI, LLVIP, or M3FD as needed

EXPERIMENTS = {
    'roc1': 'runs/test/test_MF_vedai_noSR_300ep/best_predictions.json',
    'roc2': 'runs/test/test_PIX_FUS_vedai_noSR_300ep/best_predictions.json',
    'roc3': 'runs/test/test_FEAT_FUS_mdf4_vedai_noSR_300ep/best_predictions.json',

    'roc4': 'runs/train/MF_vedai_noSR_300ep/best_train_predictions.json',
    'roc5': 'runs/train/PIX_FUS_vedai_noSR_300ep/best_train_predictions.json',
    'roc6': 'runs/train/FEAT_FUS_mdf4_vedai_noSR_300ep/best_train_predictions.json',
}

GROUND_TRUTH_DIR =  {
    'roc1': f"dataset/{DATASET_NAME}/labels",
    'roc2': f"dataset/{DATASET_NAME}/labels",
    'roc3': f"dataset/{DATASET_NAME}/labels",
    'roc4': f"dataset/{DATASET_NAME}/labels",
    'roc5': f"dataset/{DATASET_NAME}/labels",
    'roc6': f"dataset/{DATASET_NAME}/labels",
}
# GROUND_TRUTH_DIR =  {
#     'roc1': "dataset/VEDAI/labels",
#     'roc2': "dataset/LLVIP/labels",
#     'roc3': "dataset/M3FD/labels",
# }
IMAGES_DIR = {
    'roc1': f"dataset/{DATASET_NAME}/images",
    'roc2': f"dataset/{DATASET_NAME}/images",
    'roc3': f"dataset/{DATASET_NAME}/images",
    'roc4': f"dataset/{DATASET_NAME}/images",
    'roc5': f"dataset/{DATASET_NAME}/images",
    'roc6': f"dataset/{DATASET_NAME}/images",
}
# IMAGES_DIR = {
#     'roc1': "dataset/VEDAI/images",
#     'roc2': "dataset/LLVIP/images",
#     'roc3': "dataset/M3FD/images",
# }
OUTPUT_DIR = "runs/roc/comparison_all_datasets"

IMAGE_WIDTH = {
    'roc1': 512,
    'roc2': 512,
    'roc3': 512,
    'roc4': 512,
    'roc5': 512,
    'roc6': 512,
}
IMAGE_HEIGHT = {
    'roc1': 512,
    'roc2': 512,
    'roc3': 512,
    'roc4': 512,
    'roc5': 512,
    'roc6': 512,
}

# IMAGE_WIDTH = {
#     'roc1': 512,  # VEDAI original image width
#     'roc2': 1280,  # LLVIP original image width
#     'roc3': 1024,  # M3FD original image width
# }
# IMAGE_HEIGHT = {
#     'roc1': 512,  # VEDAI original image height
#     'roc2': 1024,  # LLVIP original image height
#     'roc3': 780,  # M3FD original image height
# }

DATASET_MAP = {
    'roc1': DATASET_NAME,
    'roc2': DATASET_NAME,
    'roc3': DATASET_NAME,
    'roc4': DATASET_NAME,
    'roc5': DATASET_NAME,
    'roc6': DATASET_NAME,
}

# DATASET_MAP = {
#     'roc1': 'VEDAI',
#     'roc2': 'LLVIP',
#     'roc3': 'M3FD'
# }

EXPERIMENT_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'cyan',]
ADDITIONAL_COLORS = ['orange', 'purple', 'cyan', 'magenta', 'brown', 'pink']
FIXED_FA_THRESHOLDS = [1e-8, 1e-7, 1e-6]
MAX_IMAGES_PER_THRESHOLD = 100 # Maximum number of example images to save per FA threshold
FA_THRESHOLD_DIRS = {
    'roc1': {
        'parent_dir': f"{OUTPUT_DIR}/{DATASET_NAME}/",
        'modality_dir_MF': "MF/",

    },
    'roc2': {
        'parent_dir': f"{OUTPUT_DIR}/{DATASET_NAME}/",
        'modality_dir_MF': "PIX_FUS/",
    },
    'roc3': {
        'parent_dir': f"{OUTPUT_DIR}/{DATASET_NAME}/",
        'modality_dir_MF': "FEAT_FUS_mdf1/",
    },
    'roc4': {
        'parent_dir': f"{OUTPUT_DIR}/{DATASET_NAME}/",
        'modality_dir_MF': "FEAT_FUS_mdf2/",
    },
    'roc5': {
        'parent_dir': f"{OUTPUT_DIR}/{DATASET_NAME}/",
        'modality_dir_MF': "FEAT_FUS_mdf3/",
    },
    'roc6': {
        'parent_dir': f"{OUTPUT_DIR}/{DATASET_NAME}/",
        'modality_dir_MF': "FEAT_FUS_mdf4/",
    },

}

# TOTAL_IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT  # square pixels
TOTAL_IMAGE_AREA = {key: IMAGE_WIDTH[key] * IMAGE_HEIGHT[key] for key in IMAGE_WIDTH}  # square pixels

IOU_THRESHOLD = 0.5  # Fixed IoU threshold

# Define multiple confidence thresholds to compare
CONFIDENCE_THRESHOLDS = [0.2, 0.7]
SCALAR = 2  # Scale factor for larger plots

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

def load_ground_truth(gt_dir, image_id, name):
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
                    x_center = float(parts[1]) * IMAGE_WIDTH[name]
                    y_center = float(parts[2]) * IMAGE_HEIGHT[name]
                    width = float(parts[3]) * IMAGE_WIDTH[name]
                    height = float(parts[4]) * IMAGE_HEIGHT[name]
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

def evaluate_predictions(predictions, gt_dir, name, all_test_images, confidence_threshold=0.2, iou_threshold=0.5, no_threshold=False):
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
    image_ids = []
    total_gt = 0
    total_tp_possible = 0  # Track maximum possible TPs

    # Process ALL test images (not just those with predictions after filtering)
    for image_id in all_test_images:
        gt_boxes = load_ground_truth(gt_dir, image_id, name)
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
            image_ids.append(image_id)

            if is_tp and best_gt_idx != -1:
                gt_boxes[best_gt_idx]['used'] = True

    return scores, tp_flags, total_gt, total_tp_possible, len(all_test_images), image_ids

def calculate_fa_per_pixel_metrics(scores, name, tp_flags, processed_images_count, total_tp_possible, image_ids):
    """Calculate FA per square pixel and TPR for ROC curve."""
    if len(scores) == 0:
        print("No scores to calculate metrics!")
        return np.array([0]), np.array([0])

    # Sort by confidence descending
    sorted_indices = np.argsort(-np.array(scores))
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_tp_flags = np.array(tp_flags)[sorted_indices]
    sorted_image_ids = np.array(image_ids)[sorted_indices]

    # Calculate cumulative FPs and TPs
    fp_cumsum = np.cumsum([not x for x in sorted_tp_flags])
    tp_cumsum = np.cumsum(sorted_tp_flags)

    # Calculate FA per square pixel - USE PROCESSED IMAGES COUNT, NOT TOTAL IMAGES
    total_processed_area_pixels = TOTAL_IMAGE_AREA[name] * processed_images_count
    fa_per_pixel = fp_cumsum / total_processed_area_pixels

    # Calculate TPR using total possible TPs (total GT objects)
    tpr = tp_cumsum / total_tp_possible if total_tp_possible > 0 else np.zeros_like(tp_cumsum)

    # Add (0,0) point for complete curve
    fa_per_pixel = np.concatenate(([0], fa_per_pixel))
    tpr = np.concatenate(([0], tpr))
    sorted_image_ids = np.concatenate(([None], sorted_image_ids))

    return fa_per_pixel, tpr, sorted_image_ids

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

def find_operating_point(fa_per_pixel, tpr, target_fa):
    """Find TPR at a specific FA per pixel threshold"""
    if len(fa_per_pixel) == 0:
        return None, None

    # Find closest FA value
    idx = np.argmin(np.abs(fa_per_pixel - target_fa))

    return fa_per_pixel[idx], tpr[idx]


def get_images_at_threshold(fa_per_pixel, sorted_image_ids, target_fa):
    """
    Get all images that contributed predictions up to target FA threshold

    Returns:
        - Set of unique image IDs
        - Count of predictions at this point
        - Index in the sorted predictions array
    """
    # Find the index where fa_per_pixel is closest to target_fa
    idx = np.argmin(np.abs(fa_per_pixel - target_fa))

    # Get all image IDs from start up to this index
    images_at_threshold = sorted_image_ids[:idx+1]

    # Get unique images
    unique_images = set(images_at_threshold)

    return unique_images, len(images_at_threshold), idx

def save_fa_threshold_images(unique_images, name, output_dir, target_fa, dataset_name, test_dir=None, max_images=10):
    """
    Copy predicted images from test_batch directory that contributed to predictions at a specific FA threshold

    Args:
        unique_images: Set of unique image IDs
        name: Dataset name key (roc1, roc2, etc.)
        output_dir: Base output directory
        target_fa: Target FA per pixel threshold
        dataset_name: Human-readable dataset name (VEDAI, LLVIP, M3FD)
        test_dir: Directory containing test prediction images with _pred suffix
        max_images: Maximum number of images to save (default: 10)
    """

    display_name = DATASET_MAP.get(name, dataset_name)

    # Limit the number of images to save
    images_to_save = list(unique_images)
    if None in images_to_save:
        images_to_save.remove(None)

    if len(images_to_save) > max_images:
        # Take first max_images (they are already sorted by confidence through the ROC process)
        images_to_save = images_to_save[:max_images]
        print(f"  Saving {max_images} of {len(unique_images)} images for {display_name} at FA/pixel={target_fa:.1e}")
    else:
        print(f"  Saving all {len(images_to_save)} images for {display_name} at FA/pixel={target_fa:.1e}")

    copied_count = 0
    for image_id in images_to_save:
        if image_id is None:
            continue

        # Determine modality from directory name
        if 'RGB' in test_dir:
            modality = 'RGB'
            base_name = image_id
        elif 'IR' in test_dir:
            modality = 'IR'
            base_name = image_id
        elif 'MF' in test_dir:
            modality = 'MF'
            base_name = image_id
        elif 'PIX_FUS' in test_dir:
            modality = 'PIX_FUS'
            base_name = image_id
        elif 'FEAT_FUS_mdf1' in test_dir:
            modality = 'FEAT_FUS_mdf1'
            base_name = image_id
        elif 'FEAT_FUS_mdf2' in test_dir:
            modality = 'FEAT_FUS_mdf2'
            base_name = image_id
        elif 'FEAT_FUS_mdf3' in test_dir:
            modality = 'FEAT_FUS_mdf3'
            base_name = image_id
        elif 'FEAT_FUS_mdf4' in test_dir:
            modality = 'FEAT_FUS_mdf4'
            base_name = image_id
        else:
            modality = 'NA'  # Default to NA
            base_name = image_id

        # Create directory structure: VEDAI/RGB/1e-07/
        threshold_dir = os.path.join(output_dir, display_name, modality, f"{target_fa:.0e}")
        ensure_dir(threshold_dir)

        # Find predicted image file with _pred suffix from test directory
        img_file = None
        if test_dir and os.path.exists(test_dir):
            # Try multiple patterns to handle different naming conventions
            # Pattern 1: test_batch*_<image_id>_pred.png (VEDAI, M3FD)
            pattern1 = os.path.join(test_dir, f"test_batch*_{base_name}_pred.png")
            # Pattern 2: test_batch*_<image_id>.jpg_pred.png (LLVIP)
            pattern2 = os.path.join(test_dir, f"test_batch*_{base_name}.jpg_pred.png")

            matching_files = glob.glob(pattern1)
            if not matching_files:
                matching_files = glob.glob(pattern2)

            if matching_files:
                img_file = matching_files[0]  # Take first match

        if img_file and os.path.exists(img_file):
            # Copy the predicted image file directly
            filename = os.path.basename(img_file)
            save_path = os.path.join(threshold_dir, filename)
            shutil.copy2(img_file, save_path)
            copied_count += 1
            print(f"  Copied: {filename} -> {save_path}")
        else:
            print(f"  Warning: Predicted image for {base_name} not found in {test_dir}")

    print(f"  Total copied: {copied_count}/{len(images_to_save)} images")

def plot_multiple_confidence_thresholds(conf_results, output_dir, iou_threshold):
    """Plot ROC curves for multiple confidence thresholds on the same figure"""
    ensure_dir(output_dir)
    plt.figure(figsize=(12*SCALAR, 8*SCALAR))

    for i, (name, conf_result) in enumerate(conf_results.items()):

        valid_results = {k: v for k, v in conf_result.items() if v is not None}

        if not valid_results:
            print("No valid results to plot!")
            return

        colors = [EXPERIMENT_COLORS[i], ADDITIONAL_COLORS[i % len(ADDITIONAL_COLORS)]]  # Different colors for different confidence thresholds
        line_styles = ['-', '--']  # Different line styles

        for (conf_thresh, result_data), color, line_style in zip(
            valid_results.items(), colors, line_styles):

            # Unpack the result data
            fa_per_pixel, tpr, total_tp_possible, processed_images, tp_count, sorted_image_ids, dataset_name = result_data

            if len(fa_per_pixel) > 0 and len(tpr) > 0:
                # Calculate normalized AUC
                roc_auc = calculate_normalized_auc(fa_per_pixel, tpr)
                print("plotting name:", name)
                # Plot the curve
                label = f'{name} - Conf ≥ {conf_thresh} (AUC = {roc_auc:.3f})' if conf_thresh != 'no_threshold' else f'{name} - No Threshold (AUC = {roc_auc:.3f})'
                plt.plot(fa_per_pixel, tpr, color=color, linestyle=line_style,
                        label=label,
                        linewidth=2)
                for fixed_fa in FIXED_FA_THRESHOLDS:
                    unique_images, num_predictions, idx = get_images_at_threshold(fa_per_pixel, sorted_image_ids, target_fa=fixed_fa)
                    # Get test directory from experiment path using dataset_name (which is the key like 'roc1')
                    test_dir = os.path.dirname(EXPERIMENTS[dataset_name])
                    save_fa_threshold_images(unique_images, dataset_name, output_dir, target_fa=fixed_fa,
                                           dataset_name=dataset_name, test_dir=test_dir,
                                           max_images=MAX_IMAGES_PER_THRESHOLD)
                    fixed_fa_value, fixed_tpr_value = find_operating_point(fa_per_pixel, tpr, target_fa=fixed_fa)
                    if fixed_fa_value is not None and fixed_tpr_value is not None:
                        plt.scatter(fixed_fa_value, fixed_tpr_value, color=color, edgecolor='black', zorder=5, s=100,
                                    label=f'FA/pixel={fixed_fa:.1e}, TPR={fixed_tpr_value:.3f}')


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

def display_statistics(conf_results, name, confidence_threshold, predictions, all_test_images, opt):
    # Evaluate with current confidence threshold
    scores, tp_flags, total_gt, total_tp_possible, processed_images_count, image_ids = evaluate_predictions(
        predictions, GROUND_TRUTH_DIR[name], name, all_test_images,
        confidence_threshold=confidence_threshold,
        iou_threshold=IOU_THRESHOLD,
    )
    if len(scores) > 0:
        # Calculate FA per pixel metrics
        fa_per_pixel, tpr, sorted_image_ids = calculate_fa_per_pixel_metrics(scores, name, tp_flags, processed_images_count, total_tp_possible, image_ids)

        # Calculate true positive count
        tp_count = sum(tp_flags)

        # Store all necessary data for plotting
        conf_results[confidence_threshold] = (fa_per_pixel, tpr, total_tp_possible, processed_images_count, tp_count, sorted_image_ids, name)

        # Print statistics
        if opt.no_threshold:
            print(f"Statistics (No Threshold):")
        else:
            print(f"Statistics (Conf ≥ {confidence_threshold}):")
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
        if opt.no_threshold:
            conf_results['no_threshold'] = None
            print("No predictions found with no thresholding")
        else:
            conf_results[confidence_threshold] = None
            print(f"No predictions found with confidence threshold {confidence_threshold}")
    return conf_results

def main(opt):
    all_experiment_results = {}

    for name, pred_file in EXPERIMENTS.items():

        ensure_dir(OUTPUT_DIR)

        # Count total images for FA per pixel calculation
        total_images = count_total_images(GROUND_TRUTH_DIR[name])
        print(f"Total images in dataset: {total_images}")
        print(f"IoU Threshold: {IOU_THRESHOLD}")
        print(f"Confidence Thresholds to compare: {CONFIDENCE_THRESHOLDS if not opt.no_threshold else 'No thresholding (all predictions)'}")

        # Clean directory name for output: example :runs/test/test_RGB_m3fd_noSR_300ep_img_512/best_predictions.json -> test_RGB_m3fd_noSR_300ep_img_512
        exp_name = pred_file.replace('.json', '').replace('runs/test/','').replace('/best_predictions','')

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
            # Display statistics for no thresholding
            conf_results = display_statistics(
                conf_results,
                name=name,
                confidence_threshold=0.0,
                predictions=predictions,
                all_test_images=all_test_images,
                opt=opt
            )
        else:
            for confidence_threshold in CONFIDENCE_THRESHOLDS:
                print(f"\n--- Evaluating with confidence threshold: {confidence_threshold} ---")

                # Display statistics for current confidence threshold
                conf_results = display_statistics(
                    conf_results,
                    name=name,
                    confidence_threshold=confidence_threshold,
                    predictions=predictions,
                    all_test_images=all_test_images,
                    opt=opt
                )
        all_experiment_results[exp_name] = conf_results

    # Plot results for all confidence thresholds
    plot_multiple_confidence_thresholds(all_experiment_results, OUTPUT_DIR, IOU_THRESHOLD)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-threshold', action='store_true', help='Disable confidence thresholding')
    opt = parser.parse_args()
    main(opt)
