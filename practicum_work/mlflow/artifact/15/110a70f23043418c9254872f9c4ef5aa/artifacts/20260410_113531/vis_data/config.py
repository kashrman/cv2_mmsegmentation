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
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/cleared'
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
epoch_num = 150
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=None,
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=32,
        in_channels=3,
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            2,
            2,
            2,
            2,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
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
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            32,
            64,
            160,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(
                class_weight=[
                    0.01,
                    1.0,
                    1.0,
                ],
                loss_weight=1.0,
                type='CrossEntropyLoss',
                use_sigmoid=False),
            dict(
                class_weight=[
                    0.01,
                    1.0,
                    1.0,
                ],
                loss_weight=0.5,
                type='DiceLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=3,
        type='SegformerHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_workers = 4
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ), lr=6e-05, type='AdamW', weight_decay=0.01)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.1, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=150,
        eta_min=1e-07,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='img/test', seg_map_path='labels/test'),
        data_root='data/cleared',
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
    data_root='data/cleared',
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
train_cfg = dict(max_epochs=150, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='img/train', seg_map_path='labels/train'),
        data_root='data/cleared',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(degree=(
                -20,
                20,
            ), type='RandomRotFlip'),
            dict(
                cutout_ratio=(
                    0.05,
                    0.1,
                ),
                n_holes=(
                    7,
                    15,
                ),
                prob=0.4,
                type='RandomCutOut'),
            dict(type='PhotoMetricDistortion'),
            dict(
                transforms=[
                    dict(num_steps=10, p=0.33, type='GridDistortion'),
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
    data_root='data/cleared',
    img_suffix='.jpg',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(direction='horizontal', prob=0.5, type='RandomFlip'),
        dict(degree=(
            -20,
            20,
        ), type='RandomRotFlip'),
        dict(
            cutout_ratio=(
                0.05,
                0.1,
            ),
            n_holes=(
                7,
                15,
            ),
            prob=0.4,
            type='RandomCutOut'),
        dict(type='PhotoMetricDistortion'),
        dict(
            transforms=[
                dict(num_steps=10, p=0.33, type='GridDistortion'),
            ],
            type='Albu'),
        dict(type='PackSegInputs'),
    ],
    seg_map_suffix='.png',
    type='AnimalsDataset')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(degree=(
        -20,
        20,
    ), type='RandomRotFlip'),
    dict(
        cutout_ratio=(
            0.05,
            0.1,
        ),
        n_holes=(
            7,
            15,
        ),
        prob=0.4,
        type='RandomCutOut'),
    dict(type='PhotoMetricDistortion'),
    dict(
        transforms=[
            dict(num_steps=10, p=0.33, type='GridDistortion'),
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
        data_root='data/cleared',
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
    data_root='data/cleared',
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
            exp_name='segformer_b0_v1',
            run_name='segformer_b0',
            tracking_uri='http://localhost:5000',
            type='MLflowVisBackend'),
    ])
work_dir = 'practicum_work/artifacts/mmsegmentation_work_dir'
