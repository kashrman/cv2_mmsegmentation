# Доразметка данных с помощью YOLO + SAM

import os
import shutil
import cv2
import numpy as np
from ultralytics.data.annotator import auto_annotate

from .conf import DATA_ADD_ROOT, DATA_SRC_ROOT, DATA_ERR_ROOT
from .visualize_samples_with_masks import visualize_samples_with_masks
from .data_filter import DataFilter


err_img_dir = f'{DATA_ERR_ROOT}/img/train'
add_masks_input_dir = f'{DATA_ADD_ROOT}/img/train'
add_masks_yolo_sam_dir = f'{DATA_ADD_ROOT}/yolo_sam_labels/train'
add_masks_output_dir = f'{DATA_ADD_ROOT}/labels/train'

# os.makedirs(add_masks_labels_dir, exist_ok=True) # ultralytics сам все сделает
os.makedirs(add_masks_output_dir, exist_ok=True)

def move_err_to_add_dir():
    if os.path.exists(err_img_dir):
        print(f"Move from {err_img_dir} to {add_masks_input_dir}")
        shutil.move(err_img_dir, add_masks_input_dir)
    else:
        print(f"Source directory not found: {err_img_dir}")

def auto_annotate_yolo_sam():
    auto_annotate(
        data=add_masks_input_dir,
        det_model="yolo26l.pt",
        sam_model="sam_b.pt", # "sam_l.pt" "mobile_sam.pt"
        device="cuda",
        output_dir=add_masks_yolo_sam_dir
    )

def convert_format():
    # для сохранения только котов/собак и перевод в нужный формат
    CLASS_ID_TO_MASK = {
        15: 2,  # cat → 2
        16: 1,  # dog → 1
    }

    # Порядок отрисовки: сначала фон (меньший приоритет), потом передний план
    DRAW_ORDER = [1, 2]  # dog первым, cat поверх (при перекрытии побеждает cat)

    for img_name in os.listdir(add_masks_input_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        img_path = os.path.join(add_masks_input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read: {img_path}")
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(add_masks_yolo_sam_dir, f'{base_name}.txt')

        polygons_by_class = {pc: [] for pc in DRAW_ORDER}

        if not os.path.exists(label_path):
            print(f"No label for {img_name}, saving empty mask")
        else:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    class_id = int(parts[0])
                    if class_id not in CLASS_ID_TO_MASK:
                        continue

                    pixel_class = CLASS_ID_TO_MASK[class_id]

                    coords = parts[1:]
                    if len(coords) < 6:  # минимум 3 точки для полигона
                        print(f"  Skipping degenerate polygon in {label_path}")
                        continue

                    seg = np.array(coords, dtype=float).reshape(-1, 2)
                    seg[:, 0] *= w
                    seg[:, 1] *= h
                    seg = np.round(seg).astype(np.int32)

                    polygons_by_class[pixel_class].append(seg)

            cats = len(polygons_by_class[2])
            dogs = len(polygons_by_class[1])
            print(f"{img_name}: dogs={dogs}, cats={cats}")

        # Рисуем в порядке DRAW_ORDER: последний рисуется поверх
        for pixel_class in DRAW_ORDER:
            for seg in polygons_by_class[pixel_class]:
                cv2.fillPoly(mask, [seg], pixel_class)

        mask_path = os.path.join(add_masks_output_dir, f'{base_name}.png')
        cv2.imwrite(mask_path, mask)
        print(f"  Saved: {mask_path}")


if __name__ == "__main__":
    
    move_err_to_add_dir()
    
    auto_annotate_yolo_sam()
    
    convert_format()
    
    visualize_samples_with_masks(add_masks_input_dir, add_masks_output_dir, 'train_after_add_yolo_sam_labels', num_samples=25)

