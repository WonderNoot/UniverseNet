# dataset settings
dataset_type = 'MyDataset'
data_root = '/content/UniverseNet/datasets/dataset-coco(1280x1080)/'
classes = ('ILS_vertical_filled', 'ILS_horizontal_filled', 'red_light', 'yellow_light', 'ILS_horizontal_empty', 'ILS_vertical_empty')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 960)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
train_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes))
    )
val_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes))
    )
test_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes))
    )

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annotation_data.json',
        img_prefix=data_root + 'train/image_data/',
        pipeline=train_pipeline,
),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotation_data.json',
        img_prefix=data_root + 'val/image_data/',
        pipeline=val_pipeline,
),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/annotation_data.json',
        img_prefix=data_root + 'test/image_data/',
        pipeline=test_pipeline,
))
evaluation = dict(interval=1, metric='bbox')