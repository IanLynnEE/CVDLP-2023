_base_ = ['co_dino_5scale_r50_8xb2_1x_coco.py']

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth'
max_epochs = 32
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        by_epoch=True,
        end=32,
        gamma=0.1,
        milestones=[8, 16, 24]
    ),
]
default_hooks = dict(checkpoint=dict(max_keep_ckpts=8))

data_root = 'dataset/'
classes = ('creatures', 'fish', 'jellyfish', 'penguin',
           'puffin', 'shark', 'starfish', 'stingray')
num_classes = len(classes)
load_pipeline = None

_base_.model.roi_head[0]['bbox_head']['num_classes'] = len(classes)
_base_.model.bbox_head[0]['num_classes'] = len(classes)

model = dict(
    query_head=dict(num_classes=len(classes)),
    roi_head=[_base_.model.roi_head[0]],
    bbox_head=[_base_.model.bbox_head[0]],
)

# Image size in Coco is 640x480. We have 576x1024 and 768x1024 in our dataset.
train_cfg = dict(max_epochs=max_epochs)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice', transforms=[
        [
            dict(type='RandomChoiceResize', keep_ratio=True, scales=[
                (576, 1333), (608, 1333), (640, 1333), (672, 1333),
                (704, 1333), (736, 1333), (768, 1333), (800, 1333),
                (832, 1333),
            ]),
        ],
        [
            dict(type='RandomCrop', crop_type='absolute_range',
                 crop_size=(384, 600), allow_negative_crop=True),
            dict(type='RandomChoiceResize', keep_ratio=True, scales=[
                (576, 1333), (608, 1333), (640, 1333), (672, 1333),
                (704, 1333), (736, 1333), (768, 1333), (800, 1333),
                (832, 1333),
            ]),
        ]
    ]),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/val.json',
        data_prefix=dict(img='valid/'),
        data_root=data_root,
        metainfo=dict(classes=classes),
    )
)
val_evaluator = dict(
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    outfile_prefix='./work_dirs/valid',
)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs'),
]
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        data_root=data_root,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
    )
)
test_evaluator = dict(
    ann_file=data_root + 'annotations/test.json',
    format_only=True,
    metric='bbox',
    outfile_prefix='./work_dirs/test',
)
