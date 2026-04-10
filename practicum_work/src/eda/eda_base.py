## анализ целостности данных и распределения классов

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image 
import cv2

from .conf import DATA_SRC_ROOT, EDA_RESULTS, CLASS_NAMES


def load_mask_annotations(mask_dir):
    """Загрузка информации о масках из директории"""
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    mask_info = {}
    
    for mask_file in mask_files:
        if mask_file.endswith('.png'):
            img_id = mask_file.replace('.png', '')
            mask_path = os.path.join(mask_dir, mask_file)
            
            mask = np.array(Image.open(mask_path))
            unique_classes = np.unique(mask)
            
            mask_info[img_id] = {
                'file_name': mask_file,
                'path': mask_path,
                'shape': mask.shape,
                'classes': unique_classes
            }
    
    return mask_info

def calculate_class_counts(mask_info, image_ids):
    """Подсчет числа изображений с экземпляром каждого классов"""
    class_counts = {}
    for mask_id, info in mask_info.items():
        if mask_id in image_ids:
            for cls in info['classes']:
                if cls not in class_counts:
                    class_counts[cls] = 0
                class_counts[cls] += 1
    return class_counts

def check_img_and_mask_sizes(mask_info, image_ids, image_files, img_dir):
    """Проверка совпадения размеров масок и изображений"""
    size_mismatch = []
    
    for mask_id, info in mask_info.items():
        if mask_id in image_ids:
            img_file = None
            for f in image_files:
                if f.startswith(mask_id):
                    img_file = f
                    break
            
            if img_file:
                img_path = os.path.join(img_dir, img_file)
                img = Image.open(img_path)
                if img.size != info['shape'][::-1]:
                    size_mismatch.append(mask_id)
                    
    return size_mismatch

def print_dataset_stats(dataset_name, num_images, num_masks, valid_pairs, 
                        images_without_masks, masks_without_images, size_mismatch):
    """Вывод статистики датасета"""
    print(f"{dataset_name}: images={num_images}; masks={num_masks}; valid_pairs:{valid_pairs}")
    
    def print_msg_and_first5_mask_id(masks_id, message_err, message_good):
        if masks_id:
            print(f"  - {message_err}: {len(masks_id)}")
            if len(masks_id) <= 5:
                for img_id in masks_id:
                    print(f"    * {img_id}")
            else:
                print(f"    (showing first 5 of {len(masks_id)})")
                for img_id in list(masks_id)[:5]:
                    print(f"    * {img_id}")
        else:
            print(message_good)
    
    print_msg_and_first5_mask_id(images_without_masks, "Images without masks", "All images have masks")
    print_msg_and_first5_mask_id(images_without_masks, "Masks without images", "All masks have images")
    print_msg_and_first5_mask_id(images_without_masks, "Masks with size mismatch", "All images and masks have same sizes")

def analyze_and_check_dataset(dataset_name):
    """Анализ и проверка целостности части датасета"""
    print(f"Start analyze {dataset_name}")

    img_dir = os.path.join(DATA_SRC_ROOT, 'img', dataset_name)
    mask_dir = os.path.join(DATA_SRC_ROOT, 'labels', dataset_name)
    
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_ids = set([f.split('.')[0] for f in image_files])
    
    mask_info = load_mask_annotations(mask_dir)
    mask_ids = set(mask_info.keys())
    
    images_without_masks = image_ids - mask_ids
    masks_without_images = mask_ids - image_ids
    image_mask_pairs = mask_ids & image_ids
    
    class_counts = calculate_class_counts(mask_info, image_ids)
    
    size_mismatch = check_img_and_mask_sizes(mask_info, image_ids, image_files, img_dir)
    
    print_dataset_stats(dataset_name, len(image_ids), len(mask_ids), len(image_mask_pairs),
                       images_without_masks, masks_without_images, size_mismatch)
    
    class_distribution = pd.DataFrame({
        'class_id': list(class_counts.keys()),
        'class_name': [CLASS_NAMES.get(c, f"unknown class {c}") for c in class_counts.keys()],
        'count': list(class_counts.values())
    }).sort_values('class_id', ascending=True).reset_index(drop=True)
    print(f"Class_distribution:\n{class_distribution.to_string(index=False)}\n")
    
    return {
        'dataset_name': dataset_name,
        'image_files': image_files,
        'image_ids': image_ids,
        'mask_info': mask_info,
        'class_counts': class_counts,
        'class_distribution': class_distribution,
        'images_without_masks': images_without_masks,
        'masks_without_images': masks_without_images,
        'size_mismatch': size_mismatch,
        'summary': {
            'num_images': len(image_ids),
            'num_masks': len(mask_ids),
            'valid_pairs': len(image_mask_pairs),
            'missing_masks': len(images_without_masks),
            'missing_images': len(masks_without_images),
            'size_mismatch': len(size_mismatch)
        }
    }

def visualize_integrity_summary(train_results, val_results, test_results):
    """Визуализация сводки по целостности данных"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    datasets = ['Train', 'Validation', 'Test']
    results_list = [train_results, val_results, test_results]
    
    images_count = [r['summary']['num_images'] if r else 0 for r in results_list]
    masks_count = [r['summary']['num_masks'] if r else 0 for r in results_list]
    valid_pairs = [r['summary']['valid_pairs'] if r else 0 for r in results_list]
    
    x = np.arange(len(datasets))
    width = 0.25
    
    axes[0].bar(x - width, images_count, width, label='Images', color='skyblue')
    axes[0].bar(x, masks_count, width, label='Masks', color='lightcoral')
    axes[0].bar(x + width, valid_pairs, width, label='Valid Pairs', color='lightgreen')
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Dataset Overview')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].legend()
    
    for i, (img, mask, pair) in enumerate(zip(images_count, masks_count, valid_pairs)):
        axes[0].text(i - width, img, str(img), ha='center', va='bottom', fontsize=9)
        axes[0].text(i, mask, str(mask), ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width, pair, str(pair), ha='center', va='bottom', fontsize=9)
    
    missing_masks = [r['summary']['missing_masks'] if r else 0 for r in results_list]
    missing_images = [r['summary']['missing_images'] if r else 0 for r in results_list]
    size_mismatch = [r['summary']['size_mismatch'] if r else 0 for r in results_list]
    
    axes[1].bar(x - width/2, missing_masks, width, label='Missing Masks', color='red')
    axes[1].bar(x + width/2, missing_images, width, label='Missing Images', color='orange')
    
    if any(size_mismatch):
        axes[1].scatter(x, size_mismatch, color='red', s=100, marker='x', 
                       label='Size Mismatch', zorder=5)
    
    axes[1].set_xlabel('Dataset')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Data Count')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(EDA_RESULTS, 'count_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nGraph saved: {save_path}")

def save_all_statistics(train_results, val_results, test_results):
    """Сохранение всей статистики в CSV файлы"""
    stats_data = []
    for results in [train_results, val_results, test_results]:
        if results:
            summary_stat = results['summary'].copy()
            summary_stat['dataset'] = results['dataset_name']
            stats_data.append(summary_stat)
    
    stats_df = pd.DataFrame(stats_data)
    stats_path = os.path.join(EDA_RESULTS, 'dataset_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Dataset statistics saved: {stats_path}")
    
    all_classes = []
    for results in [train_results, val_results, test_results]:
        if results and results['class_distribution'] is not None:
            dist = results['class_distribution'].copy()
            dist['dataset'] = results['dataset_name']
            all_classes.append(dist)
    
    if all_classes:
        combined_dist = pd.concat(all_classes, ignore_index=True)
        dist_path = os.path.join(EDA_RESULTS, 'class_distribution.csv')
        combined_dist.to_csv(dist_path, index=False)
        print(f"Class distribution saved: {dist_path}")
    
    problematic = []
    for results in [train_results, val_results, test_results]:
        if results:
            for img_id in results['images_without_masks']:
                problematic.append({
                    'dataset': results['dataset_name'],
                    'image_id': img_id,
                    'issue': 'missing_mask',
                    'type': 'image'
                })
            for mask_id in results['masks_without_images']:
                problematic.append({
                    'dataset': results['dataset_name'],
                    'image_id': mask_id,
                    'issue': 'missing_image',
                    'type': 'mask'
                })
            for img_id in results['size_mismatch']:
                problematic.append({
                    'dataset': results['dataset_name'],
                    'image_id': img_id,
                    'issue': 'size_mismatch',
                    'type': 'both'
                })
    
    if problematic:
        problems_df = pd.DataFrame(problematic)
        problems_path = os.path.join(EDA_RESULTS, 'problematic_samples.csv')
        problems_df.to_csv(problems_path, index=False)
        print(f"Found {len(problematic)} integrity issues.")
        print(f"Problematic samples saved: {problems_path}")
    else:
        print("All data is intact and ready to use!")


if __name__ == "__main__":
    train_results = analyze_and_check_dataset('train')
    val_results = analyze_and_check_dataset('val')
    test_results = analyze_and_check_dataset('test')

    save_all_statistics(train_results, val_results, test_results)

    visualize_integrity_summary(train_results, val_results, test_results)