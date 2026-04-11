## визуализация разметки

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image 
import cv2

from conf import DATA_SRC_ROOT, EDA_RESULTS, CLASS_NAMES, CLASS_COLORS


def visualize_samples_with_masks(img_dir, mask_dir, dataset_name, num_samples):
    """Визуализация num_samples примеров с масками в два ряда"""
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg'))]
    if len(img_files) == 0:
        print(f"There no files in {img_dir}")
        return
    img_files = random.sample(img_files, min(num_samples, len(img_files)))
    
    num_samples = len(img_files)
    num_samples = num_samples if num_samples % 2 == 0 else num_samples - 1
    num_rows = num_samples // 2
    img_files = img_files[:num_samples]
    
    _, axes = plt.subplots(num_rows, 4, figsize=(16, 4 * num_rows))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_file in enumerate(img_files):
        row = idx // 2
        col = (idx % 2) * 2
        
        img_path = os.path.join(img_dir, img_file)
        mask_name = os.path.splitext(img_file)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = np.array(Image.open(mask_path))
        
        colormap = np.array(CLASS_COLORS, dtype=np.uint8)
        alpha = 0.6
        overlay = cv2.addWeighted(img, alpha, colormap[mask], 1 - alpha, 0)
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'Original: {img_file}', fontsize=10)
        axes[row, col].axis('off')
        
        axes[row, col + 1].imshow(overlay)
        axes[row, col + 1].set_title(f'With Mask Overlay', fontsize=10)
        axes[row, col + 1].axis('off')
    
    for row in range(num_rows):
        for col in range(2, 4):
            axes[row, col].axis('off')
    
    legend_patches = []
    for class_id, class_name in CLASS_NAMES.items():
        if class_id != 0:
            color = [c/255 for c in CLASS_COLORS[class_id]]
            legend_patches.append(mpatches.Patch(color=color, label=class_name))
    
    axes[0, 0].legend(handles=legend_patches, loc='upper left', fontsize=10, 
                      title='Classes', title_fontsize=11, framealpha=0.7)
    
    title = f"{dataset_name} examples"
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(EDA_RESULTS, f'{dataset_name.lower()}_samples.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

def visualize_all_dataset_samples():
    """Визуализация примеров для всех датасетов"""
    for dataset in ['train', 'val', 'test']:
        img_dir = os.path.join(DATA_SRC_ROOT, 'img', dataset)
        mask_dir = os.path.join(DATA_SRC_ROOT, 'labels', dataset)
        # img_dir = os.path.join(DATA_CLEARED_ROOT, 'img', dataset)
        # mask_dir = os.path.join(DATA_CLEARED_ROOT, 'labels', dataset)
        visualize_samples_with_masks(img_dir, mask_dir, dataset, num_samples=10)


if __name__ == "__main__":
    visualize_all_dataset_samples()