"""
Dataset Transformation Script for SuperYOLO
Converts VOC XML annotations to YOLO format and creates k-fold splits

Supported datasets: LLVIP, M3FD
Usage: python transform_llvip.py --dataset LLVIP --folds 10

NOTE: M3FD Annotation directory muse be renamed to 'Annotations' for the script to work correctly.
NOTE: This script does not handle copying images, it only renames them and organizes them into a single directory. 
Make sure to backup your data before running the script as it will move files.
"""


import argparse
import os
import xml.etree.ElementTree as ET

from pathlib import Path

BASE_PATH = 'dataset/'
ANNOTATIONS = 'Annotations'
IMAGES = 'images'
LABELS = 'labels'
DIRECTORIES = {ANNOTATIONS: 'Annotations', IMAGES: 'images', LABELS: 'labels'}
FILE_FORMATS = ['.jpg', '.jpeg', '.png']
CLASS_MAPPINGS = {
    'LLVIP': {'person': 0},
    'M3FD': {'People': 0, 'Car': 1, 'Motorcycle': 2, 'Bus': 3, 'Truck': 4, 'Lamp': 5}
}

SUBDIRECTORIES = {
    'LLVIP': ['visible/train', 'visible/test', 'infrared/train', 'infrared/test'],
    'M3FD': ['Ir', 'Vis']
}

def convert_voc_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height):
    """
    Convert VOC bounding box to YOLO format
    
    Args:
        xmin, ymin, xmax, ymax: Box corners in pixels
        img_width, img_height: Image dimensions in pixels
    
    Returns:
        x_center, y_center, width, height: Normalized YOLO values (0-1)
    """
    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height

    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    return x_center, y_center, width, height


def convert_annotations(dataset_name):
    directory = Path(BASE_PATH + f'/{dataset_name}/{DIRECTORIES[ANNOTATIONS]}')
    for file in directory.iterdir():
        if file.suffix == '.xml':
            tree = ET.parse(file)
            root = tree.getroot()
            image_size = root.find('size')
            img_width = int(image_size.find('width').text)
            img_height = int(image_size.find('height').text)
            yolo_lines = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in CLASS_MAPPINGS[dataset_name]:
                    class_id = CLASS_MAPPINGS[dataset_name][class_name]
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    x_center, y_center, width, height = convert_voc_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height)
                    yolo_lines.append(f'{class_id} {x_center} {y_center} {width} {height}\n')
            save_to_ouput_directory(file, yolo_lines, dataset_name)


def create_output_directories(dataset_name):
    if not os.path.exists(BASE_PATH + f'/{dataset_name}/{DIRECTORIES[LABELS]}'):
        os.makedirs(BASE_PATH + f'/{dataset_name}/{DIRECTORIES[LABELS]}')
    if not os.path.exists(BASE_PATH + f'/{dataset_name}/{DIRECTORIES[IMAGES]}'):
        os.makedirs(BASE_PATH + f'/{dataset_name}/{DIRECTORIES[IMAGES]}')

def save_to_ouput_directory(filename, yolo_lines, dataset_name):
    output_path = Path(BASE_PATH + f'/{dataset_name}/{DIRECTORIES[LABELS]}') / (filename.stem + '.txt')
    with open(output_path, 'w') as f:
        f.writelines(yolo_lines)

def create_k_folds(all_images, target_fold_size, k=10, seed=42):
    """Split images into k folds for cross-validation"""
    import random
    random.seed(seed)  # Reproducibility
    
    images = all_images.copy()
    random.shuffle(images)
    k = max(1, len(images) // target_fold_size)  # Adjust k based on target fold size
    fold_size = len(images) // k
    print(f'Total images: {len(images)}, Target fold size: {target_fold_size} -> Creating {k} folds of ~{fold_size} images each')
    folds = []
    
    for i in range(k):
        start = i * fold_size
        # Last fold gets any remainder
        end = start + fold_size if i < k-1 else len(images)
        folds.append(images[start:end])
    
    return folds

def organize_directories(dataset_name):
    """
    Organize the Training and Test image files into one image directory    
    """
    images_dir = Path(BASE_PATH) / dataset_name / 'images'
    all_images = []
    
    # Check if images are already organized
    if images_dir.exists():
        for file in images_dir.iterdir():
            if file.suffix in FILE_FORMATS and '_co' in file.stem:
                # Extract stem without the _co suffix
                img_stem = file.stem.replace('_co', '')
                all_images.append(img_stem)
        
        if len(all_images) > 0:
            print(f'Found {len(all_images)} already organized images')
            return all_images
    
    # Otherwise organize from subdirectories
    subdirs = SUBDIRECTORIES.get(dataset_name, [])
    
    for subdir in subdirs:
        # Check if 'visible' or 'Vis' is IN the path
        if 'visible' in subdir or 'Vis' in subdir:
            suffix = 'co'
        elif 'infrared' in subdir or 'ir' in subdir or 'Ir' in subdir:
            suffix = 'ir'
        else:
            suffix = 'co'  # Default fallback
        source_path = Path(BASE_PATH) / dataset_name / subdir
        if not source_path.exists():
            continue
        for file in source_path.iterdir():
            if file.suffix in FILE_FORMATS:
                target = Path(BASE_PATH) / dataset_name / 'images' / (file.stem + '_' + suffix + file.suffix)
                if suffix == 'co':  # Only count once per image pair
                    all_images.append(file.stem)
                
                if not target.exists():
                    file.rename(target)

    print(f'Organized {len(all_images)} images')
    return all_images

def generate_paths(dataset_name, all_images, fold_size, num_folds=10):
    folds = create_k_folds(all_images, target_fold_size=fold_size, k=num_folds)
    img_path = f'dataset/{dataset_name}/images/'  # Relative path
    
    actual_num_folds = len(folds)  # Use actual number of folds created
    print(f'Generating paths for {actual_num_folds} folds')
    
    # Create sequential train/test pairs instead of k-fold CV
    # This keeps training sets at ~1k images instead of combining all folds
    num_pairs = actual_num_folds // 2  # Each pair uses 2 folds (1 for train, 1 for test)
    
    for i in range(num_pairs):
        fold_num = f'{i+1:02d}'
        
        # Use one fold for training, next fold for testing
        train_images = folds[i * 2]
        test_images = folds[i * 2 + 1] if (i * 2 + 1) < actual_num_folds else folds[0]
        
        # Write training fold
        train_file = Path(BASE_PATH) / dataset_name / f'fold{fold_num}_write.txt'
        with open(train_file, 'w') as f:
            for img_stem in train_images:
                f.write(f'{img_path}{img_stem}\n')
        
        # Write test fold
        test_file = Path(BASE_PATH) / dataset_name / f'fold{fold_num}test_write.txt'
        with open(test_file, 'w') as f:
            for img_stem in test_images:
                f.write(f'{img_path}{img_stem}\n')
        
        print(f'Created fold{fold_num}: {len(train_images)} train, {len(test_images)} test')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='', help='Dataset name (e.g., LLVIP, M3FD)')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    parser.add_argument('--target-fold-size', type=int, default=1000, help='Target number of images per fold (adjusts number of folds)')
    opt = parser.parse_args()

    if opt.dataset not in CLASS_MAPPINGS:
        print(f"Dataset '{opt.dataset}' not recognized. Please choose from: {list(CLASS_MAPPINGS.keys())}")
        exit(1)

    create_output_directories(opt.dataset)
    convert_annotations(opt.dataset)
    all_images = organize_directories(opt.dataset)
    generate_paths(opt.dataset, all_images, num_folds=opt.folds, fold_size=opt.target_fold_size)