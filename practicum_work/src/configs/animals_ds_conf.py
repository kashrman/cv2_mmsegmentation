#############################
### dataset settings
#############################

dataset_type = 'AnimalsDataset'
data_root = "data/cleared_final"
batch_size=8
num_workers=4

# ==== Определяем обучающий пайплайн данных ======
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     # custom аугментации
#     dict(type='PhotoMetricDistortion'),  
#     # dict(type='RandomRotFlip', degree=(-20, 20)),
#     # dict(type='RandomCutOut', prob=0.4, n_holes=(7, 15), cutout_ratio=(0.1, 0.15)), #, seg_fill_in=0
#     # dict(type='Albu', transforms=[dict(type="GridDistortion", num_steps=10, p=0.33)]),
#     # ===
#     dict(type='PackSegInputs')
# ]
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
    
#     # 1. Геометрия — меняем форму/положение объекта
#     dict(type='RandomRotFlip', degree=(-45, 45)),
#     dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),

#     # 2. Цвет/яркость — меняем внешний вид без изменения формы
#     dict(type='PhotoMetricDistortion'),

#     # 3. Тяжёлые аугментации через Albu
#     dict(type='Albu', transforms=[

#         # Группа A: искажения формы (одно из трёх)
#         dict(type='OneOf', transforms=[
#             dict(type='GridDistortion', num_steps=5, distort_limit=0.3, p=1.0),
#             dict(type='ElasticTransform', alpha=120, sigma=6, p=1.0),
#             dict(type='OpticalDistortion', distort_limit=0.3, shift_limit=0.3, p=1.0),
#         ], p=0.5),

#         # Группа B: размытие / шум (одно из четырёх)
#         dict(type='OneOf', transforms=[
#             dict(type='GaussianBlur', blur_limit=(3, 7), p=1.0),
#             dict(type='MotionBlur', blur_limit=7, p=1.0),
#             dict(type='GaussNoise', var_limit=(10, 50), p=1.0),
#             dict(type='ISONoise', p=1.0),
#         ], p=0.5),

#         # Группа C: цвет/контраст (одно из пяти)
#         dict(type='OneOf', transforms=[
#             dict(type='CLAHE', clip_limit=4.0, p=1.0),
#             dict(type='RandomBrightnessContrast', brightness_limit=0.3, contrast_limit=0.3, p=1.0),
#             dict(type='HueSaturationValue', hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
#             dict(type='RGBShift', r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
#             dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
#         ], p=0.5),

#         # Группа D: выпадение регионов (одно из трёх)
#         dict(type='OneOf', transforms=[
#             dict(type='CoarseDropout', max_holes=8, max_height=32, max_width=32, 
#                  fill_value=0, p=1.0),
#             dict(type='GridDropout', ratio=0.3, p=1.0),
#             dict(type='PixelDropout', dropout_prob=0.05, p=1.0),
#         ], p=0.4),

#     ]),

#     # 4. CutOut из mmseg — работает и с маской
#     dict(type='RandomCutOut', prob=0.3, n_holes=(3, 7), cutout_ratio=(0.05, 0.1)),

#     dict(type='PackSegInputs')
# ]

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
    
#     dict(type='RandomRotFlip', degree=(-30, 30)),  # уменьшил угол
    
#     # Важно: RandomResize для малых объектов
#     # dict(type='RandomResize', scale=(256, 256), ratio_range=(0.8, 1.2), keep_ratio=True),
    
#     dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
#     dict(type='PhotoMetricDistortion'),
    
#     # Упрощаем Albu для малого датасета
#     dict(type='Albu', transforms=[
#         dict(type='OneOf', transforms=[
#             dict(type='GaussianBlur', blur_limit=(3, 5), p=1.0),
#             dict(type='GaussNoise', var_limit=(10, 30), p=1.0),
#         ], p=0.3),
        
#         dict(type='OneOf', transforms=[
#             dict(type='RandomBrightnessContrast', brightness_limit=0.2, contrast_limit=0.2, p=1.0),
#             dict(type='HueSaturationValue', hue_shift_limit=10, sat_shift_limit=20, p=1.0),
#         ], p=0.4),
        
#         # Убрал CoarseDropout для малых объектов - они слишком маленькие
#     ]),
    
#     # Уменьшил CutOut
#     dict(type='RandomCutOut', prob=0.2, n_holes=(2, 5), cutout_ratio=(0.03, 0.08)),
    
#     dict(type='PackSegInputs')
# ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # custom аугментации
    dict(type='RandomFlip', prob=0.4, direction='horizontal'),
    dict(type='RandomRotate', prob=0.4, degree=(-30, 30)),
    # dict(type='RandomCrop', crop_size=(224, 224)),
    # dict(type='RandomCutOut', prob=0.3, n_holes=(7, 15), cutout_ratio=(0.05, 0.1)),
    dict(type='PhotoMetricDistortion',
         brightness_delta=15,
         contrast_range=(0.9, 1.1),
         saturation_range=(0.9, 1.1),
         hue_delta=5
    ),
    dict(type='Albu', transforms=[dict(type="GridDistortion", num_steps=10, p=0.1)]),
    # ===
    dict(type='PackSegInputs')
]
train_dataset = dict(
    type=dataset_type,
    data_root=data_root, #data_root, #
    data_prefix=dict(
        img_path='img/train',
        seg_map_path='labels/train'),
    pipeline=train_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
    drop_last=True  
)


# ==== Определяем валидационный пайплайн данных ======
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs")
]
test_pipeline = val_pipeline

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/val',
        seg_map_path='labels/val'),
    pipeline=val_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset
)


# ==== Определяем тестовый пайплайн данных ======
test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/test',
        seg_map_path='labels/test'),
    pipeline=test_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset
)


# Здесь же в пайплайне данных создаются объекты для подсчета метрик
# train_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'], classwise=True) - не используется
val_evaluator = dict(num_classes=3, type='IoUMetric', iou_metrics=['mDice'], classwise=True)
test_evaluator = val_evaluator

# Для инференса на валидации
# test_dataloader = val_dataloader