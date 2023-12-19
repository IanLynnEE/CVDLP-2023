_base_ = 'co_dino_5scale_r50_1xb1_48e.py'

test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/val.json',
        data_prefix=dict(img='valid/'),
    )
)
test_evaluator = dict(
    ann_file=_base_.data_root + 'annotations/val.json',
    outfile_prefix='./work_dirs/valid',
)
