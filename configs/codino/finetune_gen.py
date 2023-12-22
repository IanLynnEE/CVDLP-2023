_base_ = ['co_dino_5scale_r50_8xb2_1x_coco.py']

load_from = './work_dirs/co_dino_5scale_r50_1xb1_48e/epoch_33.pth'
resume = True
max_epochs = 48
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=33,
        by_epoch=True,
        end=48,
        gamma=0.1,
        milestones=[8, 16, 32]
    ),
]
default_hooks = dict(checkpoint=dict(max_keep_ckpts=8))

data_root = 'outputs/gen_complex/'
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
        _delete_=True,
        datasets=[
            dict(
                data_root=data_root,
                data_prefix=dict(img=''),
                ann_file='selected_ann.json',
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                metainfo=dict(classes=classes),
                pipeline=train_pipeline,
                type='CocoDataset'
            ), dict(
                data_root=data_root,
                data_prefix=dict(img=''),
                ann_file='original_selected_ann.json',
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                metainfo=dict(classes=classes),
                pipeline=train_pipeline,
                type='CocoDataset'
            ),
        ],
        type='ConcatDataset',
    ),
)

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/val.json',
        data_prefix=dict(img='valid/'),
        data_root='dataset',
        metainfo=dict(classes=classes),
    )
)
val_evaluator = dict(
    ann_file='dataset/annotations/val.json',
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
