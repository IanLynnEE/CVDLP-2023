_base_ = 'conditional-detr_r50_8xb2-50e_coco.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/'\
    'swin_large_patch4_window12_384_22k.pth'
load_from = 'checkpoints/conditional-detr_r50_8xb2-50e_coco_20221121_180202-c83a1dc0.pth'

data_root = 'dataset/'
classes = ('creatures', 'fish', 'jellyfish', 'penguin',
           'puffin', 'shark', 'starfish', 'stingray')
max_epochs = 32
default_hooks = dict(checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=3))

param_scheduler = [
    dict(end=max_epochs, milestones=[
        16
    ], type='MultiStepLR'),
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        # Please only add indices that would be used in FPN.
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        _delete_=True,
        type='ChannelMapper',
        in_channels=[192, 384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=1),
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

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                            (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                            (736, 2048), (768, 2048), (800, 2048), (832, 2048),
                            (864, 2048), (896, 2048), (928, 2048), (960, 2048),
                            (992, 2048), (1024, 2048), (1056, 2048),
                            (1088, 2048), (1120, 2048), (1152, 2048),
                            (1184, 2048), (1216, 2048), (1248, 2048),
                            (1280, 2048), (1312, 2048), (1344, 2048),
                            (1376, 2048), (1408, 2048), (1440, 2048),
                            (1472, 2048), (1504, 2048), (1536, 2048)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                            (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                            (736, 2048), (768, 2048), (800, 2048), (832, 2048),
                            (864, 2048), (896, 2048), (928, 2048), (960, 2048),
                            (992, 2048), (1024, 2048), (1056, 2048),
                            (1088, 2048), (1120, 2048), (1152, 2048),
                            (1184, 2048), (1216, 2048), (1248, 2048),
                            (1280, 2048), (1312, 2048), (1344, 2048),
                            (1376, 2048), (1408, 2048), (1440, 2048),
                            (1472, 2048), (1504, 2048), (1536, 2048)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
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
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
    dict(type='PackDetInputs')
]
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        data_root=data_root,
        metainfo=dict(classes=classes),
        pipeline=test_pipeline,
        test_mode=True,
        type='CocoDataset'))
test_evaluator = dict(
    ann_file=data_root + 'annotations/test.json',
    backend_args=None,
    format_only=True,
    metric='bbox',
    outfile_prefix='./work_dirs/test',
    type='CocoMetric')
