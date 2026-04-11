
## запуск повторной интерактивной проверки качества разметки после доразметки YOLO+SAM

import os

from conf import DATA_SRC_ROOT, DATA_ERR_ROOT, DATA_ADD_ROOT
from data_filter import DataFilter

add_masks_input_dir = f'{DATA_ADD_ROOT}/img/train'
add_masks_yolo_sam_dir = f'{DATA_ADD_ROOT}/yolo_sam_labels/train'
add_masks_output_dir = f'{DATA_ADD_ROOT}/labels/train'
err_img_dir = f'{DATA_ERR_ROOT}/img/train'

if __name__ == "__main__":

    DataFilter('train', add_masks_input_dir, add_masks_output_dir).start()

    # список изображений с ошибками, которые так и не удалось разметить
    if os.path.exists(err_img_dir):
        print("List of error masks images:")
        files = os.listdir(err_img_dir)
        for fname in sorted(files):
            print(f"  {fname}")
        print(f"Total: {len(files)}")
    else:
        print(f"Source directory not found: {err_img_dir}")