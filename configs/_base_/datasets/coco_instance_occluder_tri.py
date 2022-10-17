dataset_type = 'TriOccluderCocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_front_mask=True, with_back_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='TriOccluderDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_front_masks', 'gt_back_masks']),
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
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    # please fill in the COCO annotation and image path
    train=dict(
        type=dataset_type,
        ann_file= "xxx/COCO/COCO2017/annotations/instances_train2017.json",
        img_prefix="xxx/COCO/COCO2017/train2017/",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file="xxx/COCO/COCO2017/annotations/instances_val2017.json",
        img_prefix="xxx/COCO/COCO2017/val2017/",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file="xxx/COCO/COCO2017/annotations/instances_val2017.json",
        img_prefix="xxx/COCO/COCO2017/val2017/",
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
