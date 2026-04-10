_base_ = [
    '../../../mmsegmentation/configs/_base_/models/deeplabv3plus_r50-d8.py',
    'animals_ds_conf.py',
    '../../../mmsegmentation/configs/_base_/default_runtime.py',
]

#############################
### Модель
#############################
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(256, 256)
    ),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
        strides=(2, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 3, 4),
        norm_cfg=dict(type='BN', requires_grad=True)
    ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=320,
        in_index=3,
        channels=256,
        dilations=(1, 12, 24, 36),
        c1_in_channels=24,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[0.2, 1.0, 1.0]  # фон весит меньше
            ),
            dict(
                type='DiceLoss',
                loss_weight=0.5,
                class_weight=[0.2, 1.0, 1.0]
            )
        ]
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=24,
        in_index=1,
        channels=48,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=0.2, class_weight=[0.2, 1.0, 1.0]),
            dict(type='DiceLoss', loss_weight=0.1, class_weight=[0.2, 1.0, 1.0])
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

#############################
### Оптимизатор 
#############################
epoch_num = 300
batch_size = 8

optimizer = dict(
    type='AdamW',
    lr=0.0001,  
    weight_decay=0.01
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'backbone': dict(lr_mult=0.5)  # backbone учится медленнее
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=epoch_num,
        by_epoch=True
    )
]

# Логирование
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch_num)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=50, save_best='mDice', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=100, draw=True)
)

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='MLflowVisBackend',
            exp_name='deeplabv3_plus_v1',
            tracking_uri='http://localhost:5000',
            run_name='deeplabv3_plus_v1',
        )
    ]
)