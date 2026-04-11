# класс для интерактивной проверки качества разметки

import os
import shutil
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io

from conf import DATA_CLEARED_ROOT, DATA_ERR_ROOT, CLASS_COLORS


class DataFilter:
    """Интерактивный фильтр данных с генерацией превью на лету"""
    
    def __init__(self, split, img_src, mask_src):
        self.split = split
        self.img_src = img_src
        self.mask_src = mask_src
      
        self.cleared_dir_img = os.path.join(DATA_CLEARED_ROOT, 'img', split)
        self.cleared_dir_labels = os.path.join(DATA_CLEARED_ROOT, 'labels', split)
        self.err_dir_img = os.path.join(DATA_ERR_ROOT, 'img', split)
        self.err_dir_labels = os.path.join(DATA_ERR_ROOT, 'labels', split)

        os.makedirs(self.cleared_dir_img, exist_ok=True)
        os.makedirs(self.cleared_dir_labels, exist_ok=True)
        os.makedirs(self.err_dir_img, exist_ok=True)
        os.makedirs(self.err_dir_labels, exist_ok=True)
        
        self.image_files = sorted([f for f in os.listdir(self.img_src) if f.endswith(('.jpg'))])
        self.current_idx = 0
        self.results = []
        
        self._create_widgets()
    
    # ------------------------------------------------------------------
    # Генерация превью
    # ------------------------------------------------------------------
    
    def _make_combined_figure(self, img_file):
        """Создаёт figure с оригиналом и наложенной маской (без сохранения на диск)"""
        img_id = img_file.split('.')[0]
        img_path  = os.path.join(self.img_src,  img_file)
        mask_path = os.path.join(self.mask_src, f"{img_id}.png")
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(img)
        axes[0].set_title('Original', fontsize=12)
        axes[0].axis('off')
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            colormap = np.array(CLASS_COLORS, dtype=np.uint8)
            overlay = cv2.addWeighted(img, 0.5, colormap[mask], 0.5, 0)
            axes[1].imshow(overlay)
            axes[1].set_title('With Mask', fontsize=12)
        else:
            axes[1].imshow(img)
            axes[1].set_title('With Mask (NOT FOUND)', fontsize=12, color='red')
        
        axes[1].axis('off')
        plt.suptitle(
            f"{self.split.upper()} | {img_file}",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        return fig
    
    # ------------------------------------------------------------------
    # Виджеты
    # ------------------------------------------------------------------
    
    def _create_widgets(self):
        self.image_output = widgets.Output()
        self.counter_label = widgets.Label(
            value=self._counter_text()
        )
        self.progress_bar = widgets.IntProgress(
            value=0, min=0, max=len(self.image_files),
            style={'bar_color':'#4CAF50'}
        )
        
        self.good_btn = widgets.Button(
            description='V CORRECT',
            button_style='success',
            layout=widgets.Layout(width='180px', height='40px')
        )
        self.bad_btn = widgets.Button(
            description='X INCORRECT',
            button_style='danger',
            layout=widgets.Layout(width='180px', height='40px')
        )
        self.skip_btn = widgets.Button(
            description='-> SKIP',
            button_style='warning',
            layout=widgets.Layout(width='180px', height='40px')
        )
        
        self.good_btn.on_click(self._on_good)
        self.bad_btn.on_click(self._on_bad)
        self.skip_btn.on_click(self._on_skip)
    
    def _counter_text(self):
        return f"Image {self.current_idx + 1} of {len(self.image_files)}"
    
    def _set_buttons_enabled(self, enabled: bool):
        """Блокируем кнопки во время отрисовки, чтобы не было двойных кликов"""
        for btn in (self.good_btn, self.bad_btn, self.skip_btn):
            btn.disabled = not enabled
    
    # ------------------------------------------------------------------
    # Отображение
    # ------------------------------------------------------------------
    
    def _show_current(self):
        self._set_buttons_enabled(False)
        
        with self.image_output:
            clear_output(wait=True)
            
            if self.current_idx >= len(self.image_files):
                self._show_summary()
                return
            
            img_file = self.image_files[self.current_idx]
            fig = self._make_combined_figure(img_file)
            plt.show()
            plt.close(fig)
        
        self.counter_label.value = self._counter_text()
        self.progress_bar.value = self.current_idx
        self._set_buttons_enabled(True)
    
    def _show_summary(self):
        """Итоговая статистика (внутри image_output)"""
        df = pd.DataFrame(self.results)
        print("\n" + "=" * 50)
        print(f"  FILTERING COMPLETED: {self.split.upper()}")
        print("=" * 50)
        print(f"  Total   : {len(df)}")
        print(f"  Correct : {len(df[df.status == 'correct'])}")
        print(f"  Incorrect: {len(df[df.status == 'incorrect'])}")
        print(f"  Skipped : {len(df[df.status == 'skipped'])}")
    
    # ------------------------------------------------------------------
    # Перемещение файлов
    # ------------------------------------------------------------------
    
    def _move_files(self, destination_img_dir, destination_labels_dir):
        img_file = self.image_files[self.current_idx]
        img_id   = img_file.split('.')[0]
        
        src_img  = os.path.join(self.img_src,  img_file)
        src_mask = os.path.join(self.mask_src, f"{img_id}.png")

        dst_img  = os.path.join(destination_img_dir, img_file)
        dst_mask = os.path.join(destination_labels_dir, f"{img_id}.png")
        
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        if os.path.exists(src_mask):
            shutil.move(src_mask, dst_mask)
    
    # ------------------------------------------------------------------
    # Обработчики кнопок
    # ------------------------------------------------------------------
    
    def _on_good(self, _):
        if self.current_idx >= len(self.image_files):
            return
        self._move_files(self.cleared_dir_img, self.cleared_dir_labels)
        self.results.append({'file': self.image_files[self.current_idx], 'status': 'correct'})
        self.current_idx += 1
        self._show_current()
    
    def _on_bad(self, _):
        if self.current_idx >= len(self.image_files):
            return
        self._move_files(self.err_dir_img, self.err_dir_labels)
        self.results.append({'file': self.image_files[self.current_idx], 'status': 'incorrect'})
        self.current_idx += 1
        self._show_current()
    
    def _on_skip(self, _):
        if self.current_idx >= len(self.image_files):
            return
        self.results.append({'file': self.image_files[self.current_idx], 'status': 'skipped'})
        self.current_idx += 1
        self._show_current()
    
    # ------------------------------------------------------------------
    # Запуск
    # ------------------------------------------------------------------
    
    def start(self):
        if not self.image_files:
            print(f"No images found in {self.img_src}")
            return
        
        print(f"Filtering {self.split.upper()} — {len(self.image_files)} images")
        
        display(widgets.VBox([
            self.counter_label,
            self.progress_bar,
            self.image_output,
            widgets.HBox(
                [self.good_btn, self.bad_btn, self.skip_btn],
                layout=widgets.Layout(justify_content='center', margin='8px 0')
            )
        ]))
        
        self._show_current()