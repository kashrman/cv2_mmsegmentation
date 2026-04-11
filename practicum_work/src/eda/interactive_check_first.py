
## запуск интерактивной проверки качества разметки всех наборов

import os

from conf import DATA_SRC_ROOT, DATA_ERR_ROOT
from data_filter import DataFilter


if __name__ == "__main__":

    DataFilter('val', f'{DATA_SRC_ROOT}/img/val', f'{DATA_SRC_ROOT}/labels/val').start()

    DataFilter('val', f'{DATA_SRC_ROOT}/img/test', f'{DATA_SRC_ROOT}/labels/test').start()

    DataFilter('train', f'{DATA_SRC_ROOT}/img/train', f'{DATA_SRC_ROOT}/labels/train').start()

    err_img_dir = f'{DATA_ERR_ROOT}/img/train'

    # Вывод списка изображений с ошибками
    if os.path.exists(err_img_dir):
        print("List of error masks images:")
        files = os.listdir(err_img_dir)
        for fname in sorted(files):
            print(f"  {fname}")
        print(f"Total: {len(files)}")
    else:
        print(f"Source directory not found: {err_img_dir}")