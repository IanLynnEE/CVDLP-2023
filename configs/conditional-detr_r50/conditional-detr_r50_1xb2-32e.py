_base_ = 'conditional-detr_r50_8xb2-50e_coco.py'

load_from = 'checkpoints/conditional-detr_r50_8xb2-50e_coco_20221121_180202-c83a1dc0.pth'

data_root = 'dataset/'
classes = ('creatures', 'fish', 'jellyfish', 'penguin',
           'puffin', 'shark', 'starfish', 'stingray')
max_epochs = 32
default_hooks = dict(checkpoint=dict(by_epoch=True, max_keep_ckpts=3))

param_scheduler = [
    dict(type='MultiStepLR', end=max_epochs, milestones=[16],),
]

model = dict(
    bbox_head=dict(
        type='ConditionalDETRHead',
        embed_dims=256,
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_cls=dict(
            type='FocalLoss',
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            use_sigmoid=True),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        num_classes=len(classes),))

# At keep_ratio=True, the value of scale is (min, max), not (w, h).
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
train_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        data_root=data_root,
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)))

val_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        ann_file='annotations/val.json',
        data_prefix=dict(img='valid/'),
        data_root=data_root,
        metainfo=dict(classes=classes),
        test_mode=True))
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    format_only=False,
    metric='bbox',
    outfile_prefix='./work_dirs/valid')

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]
test_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        data_root=data_root,
        metainfo=dict(classes=classes),
        pipeline=test_pipeline,
        test_mode=True))
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    backend_args=None,
    format_only=True,
    metric='bbox',
    outfile_prefix='./work_dirs/test')
