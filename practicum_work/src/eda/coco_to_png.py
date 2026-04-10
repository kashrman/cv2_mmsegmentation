import os
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from pycocotools.coco import COCO


def split_filename(name: str) -> str:
    """Извлекает "basename" типа:
        000000028253_7169_jpg.rf.oRA37gajJ1LYTmdmjGEM.jpg -> 000000028253_7169.jpg
    """
    stem = Path(name).stem
    core = stem.split(".rf.")[0].split("_jpg")[0]
    return core

def main():
    root = Path("data/roboflow/train")
    json_path = root / "_annotations.coco.json"
    images_dir = root
    masks_dir = root / "masks"

    # Читаем COCO
    coco = COCO(str(json_path))

    # Вытащим имя папки/датасета из `images` (если есть `train/val/test`)
    img_dirs = {}
    for img in coco.imgs.values():
        fname = img["file_name"]
        # в Roboflow имя файла в JSON часто не совпадает с именем на диске
        disk_name = img.get("file_name", None) or split_filename(fname)
        img_dirs[img["id"]] = images_dir / disk_name

    masks_dir.mkdir(exist_ok=True)

    for img_id in coco.imgs:
        img_info = coco.imgs[img_id]
        h, w = img_info["height"], img_info["width"]

        # Маска: индекс класса на пиксель
        mask = np.zeros((h, w), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_id = int(ann["category_id"])  # 1=cat, 2=dog ... 
            #в roboflow перепутано
            if cat_id == 2:
                cat_id = 1
            elif cat_id == 1:
                cat_id = 2
            segm = ann["segmentation"]

            for poly in segm:
                poly_pts = np.array(poly).reshape(-1, 2)
                cv2.fillPoly(mask, [poly_pts.astype(np.int32)], cat_id)

        # Пример: 000000028253_7169_jpg.rf.oRA37gajJ1LYTmdmjGEM.jpg -> masks/000000028253_7169.png
        img_fname = img_info["file_name"]
        img_stem = Path(split_filename(img_fname)).stem
        mask_path = masks_dir / f"{img_stem}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(mask).save(mask_path)
        print(f"Saved mask: {mask_path}")


if __name__ == "__main__":
    main()