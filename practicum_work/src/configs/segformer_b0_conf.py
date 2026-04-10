_base_ = [
    '../../../mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    'animals_ds_conf.py',
    '../../../mmsegmentation/configs/_base_/default_runtime.py',
]

#############################
### Переопределяем параметры модели для 3 классов
#############################
model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(256, 256)  # ваш размер изображений
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=3,  # background, cat, dog
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            # dict(
            #     type='CrossEntropyLoss',
            #     use_sigmoid=False,
            #     loss_weight=1.0,
            #     class_weight=[0.01, 1.0, 1.0]  # background весит меньше
            # ),
            # dict(
            #     type='DiceLoss',
            #     loss_weight=0.5,
            # )
            dict(  
                type='FocalLoss',  
                loss_name='loss_focal',  
                use_sigmoid=True,  # Only sigmoid focal loss supported now  
                gamma=3.0,  
                loss_weight=3.0, class_weight=[0.01, 1.0, 1.0]  
            ),  
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
        ]
    ),
    auxiliary_head=None,  # SegFormer не использует aux_head
    test_cfg=dict(mode='whole')
)

#############################
### Оптимизатор
#############################
epoch_num = 500

# SegFormer требует AdamW с маленьким lr
optimizer = dict(
    type='AdamW',
    lr=0.0001,  # 6e-5 для B0
    weight_decay=0.2
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'head': dict(lr_mult=10.0),  # голова учится быстрее
    #         'pos_block': dict(decay_mult=0.0),
    #         'norm': dict(decay_mult=0.0)
    #     }
    # )
)

# Плавный scheduler с прогревом
param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=0.1,
    #     by_epoch=False,
    #     begin=0,
    #     end=500  # прогрев на 500 итераций
    # ),
    dict(
        type='PolyLR',
        eta_min=1e-7,
        power=0.8,
        begin=0,
        end=epoch_num,
        by_epoch=True
    )
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch_num)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    # timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=50, save_best='mDice', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=100, draw=True)
)


visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),      # сохраняем логи локально
        # dict(
        #     type='MLflowVisBackend',
        #     exp_name='segformer_b0_v1',
        #     tracking_uri='http://localhost:5000',
        #     run_name='segformer_b0',
        # )
    ]
)