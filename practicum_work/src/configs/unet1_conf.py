#

_base_ = [
    '../../../mmsegmentation/configs/_base_/models/fcn_unet_s5-d16.py', 
    'animals_ds_conf.py', # dataset settings, включая аугментации и метрики
    '../../../mmsegmentation/configs/_base_/default_runtime.py', 
    # '../../../mmsegmentation/configs/_base_/schedules/***.py' # вместо отдельно файла добавил сразу сюда
]

#############################
### расписание обучение, вместо отдельного файла 'mmsegmentation/configs/_base_/schedules/***.py'
#############################
epoch_num = 500

optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
param_scheduler = [ dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=0, end=epoch_num, by_epoch=True) ]


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

##############################
### Основной конфиг
##############################
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        # отключил для инференса
        # dict(
        #     type='MLflowVisBackend',
        #     exp_name='sprint6_unet_v20',
        #     tracking_uri='http://localhost:5000',
        #     run_name='unet-s5-d16',
        # )
    ]
)

input_size = (256, 256)
data_preprocessor = dict(size=input_size)

model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode="whole"),
    decode_head=dict(
        num_classes=3,
        # dropout_ratio=0.2,
        loss_decode=[
            dict(type='FocalLoss', loss_name='loss_focal', use_sigmoid=True, gamma=3.0, loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
        ]
    ),
    auxiliary_head=None
)
