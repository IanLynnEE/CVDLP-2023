_base_ = 'conditional-detr_r50_8xb2-50e_coco.py'

data_root = 'dataset/'
classes = ('creatures', 'fish', 'jellyfish', 'penguin',
           'puffin', 'shark', 'starfish', 'stingray')
max_epochs = 30
default_hooks = dict(checkpoint=dict(interval=1, type='CheckpointHook'))

param_scheduler = [
    dict(end=max_epochs, milestones=[
        15
    ], type='MultiStepLR'),
]

model = dict(
    bbox_head=dict(
        embed_dims=256,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=len(classes),
        type='ConditionalDETRHead'))
load_from = 'checkpoints/conditional-detr_r50_8xb2-50e_coco_20221121_180202-c83a1dc0.pth'

# At keep_ratio=True, the value of scale is (min, max), not (w, h).
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (400, 1333),
                        (500, 1333),
                        (600, 1333),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(384, 600),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root=data_root,
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        type='CocoDataset'))

val_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, type='Resize', scale=(800, 1333)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]
val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/val.json',
        backend_args=None,
        data_prefix=dict(img='valid/'),
        data_root=data_root,
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=val_pipeline,
        type='CocoDataset'))
val_evaluator = dict(
    ann_file=data_root + 'annotations/val.json',
    format_only=False,
    metric='bbox',
    outfile_prefix='./work_dirs/valid',
    type='CocoMetric')

test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        data_root=data_root,
        metainfo=dict(classes=classes),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, type='Resize', scale=(800, 1333)),
            dict(type='PackDetInputs')
        ],
        test_mode=True,
        type='CocoDataset'))
test_evaluator = dict(
    ann_file=data_root + 'annotations/test.json',
    backend_args=None,
    format_only=True,
    metric='bbox',
    outfile_prefix='./work_dirs/test',
    type='CocoMetric')
