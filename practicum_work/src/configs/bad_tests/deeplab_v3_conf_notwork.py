
_base_ = [
    '../../../mmsegmentation/configs/_base_/models/deeplabv3_unet_s5-d16.py', 
    'animals_ds_conf.py', # dataset settings, включая аугментации и метрики
    '../../../mmsegmentation/configs/_base_/default_runtime.py', 
    # '../../../mmsegmentation/configs/_base_/schedules/***.py' # вместо отдельно файла добавил сразу сюда
]

#############################
### расписание обучение, вместо отдельного файла 'mmsegmentation/configs/_base_/schedules/***.py'
#############################
epoch_num = 300

# optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
param_scheduler = [ dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=0, end=epoch_num, by_epoch=True) ]

# Определяем спецфику обучающего и тренироворочного циклов
# У mmsegmentation много вариаций, как организовать обучающий цикл.
# Мы используем наиболее привычную, когда одна эпоха — это один проход по датасету
# Есть также обучающие циклы на основе итераций, в них одна эпоха — это какое-то число итераций 
# они используется с семплером, который бесконечно зацикливает датасет 

# Если вы будете смотреть на стандратные конфиги, там обычно именно такой подход
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch_num)
# С валидационным и тестовым попроще: берём стандартные реализации
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


##############################
### Основной конфиг
##############################
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),      # сохраняем логи локально
        dict(
            type='MLflowVisBackend',
            exp_name='deeplabv3_unet_v2',
            tracking_uri='http://localhost:5000',
            run_name='deeplabv3_unet_s5-d16',
        )
    ]
)

# Определим размер входа 
input_size = (256, 256)
data_preprocessor = dict(size=input_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode="whole"),
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(  
                type='FocalLoss',  
                loss_name='loss_focal',  
                use_sigmoid=True,  # Only sigmoid focal loss supported now  
                gamma=5.0,  
                loss_weight=3.0, class_weight=[0.01, 1.0, 1.0]  
            ),  
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
        ]
    ),
    # decode_head=dict(
    #     num_classes=3,
    #     loss_decode=[
    #         dict(type='CrossEntropyLoss', loss_weight=1.0), #, class_weight=[0.1, 0.45, 0.45]
    #         dict(type='DiceLoss', loss_weight=0.5)
    #     ]
    # ),
    auxiliary_head=None
    # auxiliary_head=dict(
    #     num_classes=3,
    #     loss_decode=[
    #         dict(  
    #             type='FocalLoss',  
    #             loss_name='loss_focal',  
    #             use_sigmoid=True,  # Only sigmoid focal loss supported now  
    #             gamma=3.0,  
    #             loss_weight=1.0  
    #         ),  
    #         dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
    #     ]
    # )
)
