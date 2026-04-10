batch_size = 8
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/cleared_final'
dataset_type = 'AnimalsDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=50,
        rule='greater',
        save_best='mDice',
        type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(draw=True, interval=100, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
epoch_num = 300
input_size = (
    256,
    256,
)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=None,
    backbone=dict(
        act_cfg=dict(type='ReLU'),
        base_channels=64,
        conv_cfg=None,
        dec_dilations=(
            1,
            1,
            1,
            1,
        ),
        dec_num_convs=(
            2,
            2,
            2,
            2,
        ),
        downsamples=(
            True,
            True,
            True,
            True,
        ),
        enc_dilations=(
            1,
            1,
            1,
            1,
            1,
        ),
        enc_num_convs=(
            2,
            2,
            2,
            2,
            2,
        ),
        in_channels=3,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=5,
        strides=(
            1,
            1,
            1,
            1,
            1,
        ),
        type='UNet',
        upsample_cfg=dict(type='InterpConv'),
        with_cp=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=64,
        in_index=4,
        loss_decode=[
            dict(
                gamma=3.0,
                loss_name='loss_focal',
                loss_weight=1.0,
                type='FocalLoss',
                use_sigmoid=True),
            dict(loss_name='loss_dice', loss_weight=1.0, type='DiceLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=3,
        num_convs=1,
        type='FCNHead'),
    pretrained=None,
    test_cfg=dict(crop_size=256, mode='whole', stride=170),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_workers = 4
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.08),
    type='OptimWrapper')
optimizer = dict(lr=0.001, type='AdamW', weight_decay=0.08)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=300,
        eta_min=1e-06,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='img/test', seg_map_path='labels/test'),
        data_root='data/cleared_final',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='AnimalsDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataset = dict(
    data_prefix=dict(img_path='img/test', seg_map_path='labels/test'),
    data_root='data/cleared_final',
    img_suffix='.jpg',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs'),
    ],
    seg_map_suffix='.png',
    type='AnimalsDataset')
test_evaluator = dict(
    classwise=True, iou_metrics=[
        'mDice',
    ], num_classes=3, type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_epochs=300, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='img/train', seg_map_path='labels/train'),
        data_root='data/cleared_final',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(direction='horizontal', prob=0.4, type='RandomFlip'),
            dict(degree=(
                -30,
                30,
            ), prob=0.4, type='RandomRotate'),
            dict(
                brightness_delta=15,
                contrast_range=(
                    0.9,
                    1.1,
                ),
                hue_delta=5,
                saturation_range=(
                    0.9,
                    1.1,
                ),
                type='PhotoMetricDistortion'),
            dict(
                transforms=[
                    dict(num_steps=10, p=0.1, type='GridDistortion'),
                ],
                type='Albu'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='AnimalsDataset'),
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_dataset = dict(
    data_prefix=dict(img_path='img/train', seg_map_path='labels/train'),
    data_root='data/cleared_final',
    img_suffix='.jpg',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(direction='horizontal', prob=0.4, type='RandomFlip'),
        dict(degree=(
            -30,
            30,
        ), prob=0.4, type='RandomRotate'),
        dict(
            brightness_delta=15,
            contrast_range=(
                0.9,
                1.1,
            ),
            hue_delta=5,
            saturation_range=(
                0.9,
                1.1,
            ),
            type='PhotoMetricDistortion'),
        dict(
            transforms=[
                dict(num_steps=10, p=0.1, type='GridDistortion'),
            ],
            type='Albu'),
        dict(type='PackSegInputs'),
    ],
    seg_map_suffix='.png',
    type='AnimalsDataset')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(direction='horizontal', prob=0.4, type='RandomFlip'),
    dict(degree=(
        -30,
        30,
    ), prob=0.4, type='RandomRotate'),
    dict(
        brightness_delta=15,
        contrast_range=(
            0.9,
            1.1,
        ),
        hue_delta=5,
        saturation_range=(
            0.9,
            1.1,
        ),
        type='PhotoMetricDistortion'),
    dict(
        transforms=[
            dict(num_steps=10, p=0.1, type='GridDistortion'),
        ],
        type='Albu'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='img/val', seg_map_path='labels/val'),
        data_root='data/cleared_final',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='AnimalsDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_dataset = dict(
    data_prefix=dict(img_path='img/val', seg_map_path='labels/val'),
    data_root='data/cleared_final',
    img_suffix='.jpg',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs'),
    ],
    seg_map_suffix='.png',
    type='AnimalsDataset')
val_evaluator = dict(
    classwise=True, iou_metrics=[
        'mDice',
    ], num_classes=3, type='IoUMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            exp_name='sprint6_unet_v20',
            run_name='unet-s5-d16',
            tracking_uri='http://localhost:5000',
            type='MLflowVisBackend'),
    ])
work_dir = 'practicum_work/artifacts/mmsegmentation_work_dir'
