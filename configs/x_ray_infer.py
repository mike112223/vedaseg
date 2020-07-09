# work dir
root_workdir = 'workdir'

# seed
seed = 0

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='INFO'),
        # dict(type='FileHandler', level='INFO'),
    ),
)

# 2. data
# test_cfg = dict(
#     scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#     bias=[0.5, 0.25, 0.0, -0.25, -0.5, -0.75],
#     flip=True,
# )

test_cfg = dict()

img_norm_cfg = dict(mean=(123.675, 116.280, 103.530), std=(58.395, 57.120, 57.375))
ignore_label = 255

dataset_type = 'CocoDataset'
dataset_root = 'data/tianchi/jinnan2_round2_train_20190401/'
data = dict(
    train=dict(
        dataset=dict(
            type=dataset_type,
            ann_file='trainaug.json',
            data_root=dataset_root,
            img_prefix='restricted',
            spec_class=3,
        ),
        transforms=[
            dict(type='RandomCrop', height=513, width=513, image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ToTensor'),
        ],
        loader=dict(
            type='DataLoader',
            batch_size=8,
            num_workers=4,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        ),
    ),
    val=dict(
        dataset=dict(
            type=dataset_type,
            ann_file='val.json',
            data_root=dataset_root,
            img_prefix='restricted',
            spec_class=3,
        ),
        transforms=[
            dict(type='PadIfNeeded', size_divisor=32, scale_bias=1, image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ToTensor'),
        ],
        loader=dict(
            type='DataLoader',
            batch_size=1,
            num_workers=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        ),
    ),
)

# 3. model
nclasses = 2
model = dict(
    # model/encoder
    encoder=dict(
        backbone=dict(
            type='ResNet',
            arch='resnet50',
        ),
    ),
    # model/decoder
    decoder=dict(
        type='GFPN',
        # model/decoder/blocks
        neck=[
            # model/decoder/blocks/block1
            dict(
                type='JunctionBlock',
                fusion_method='concat',
                top_down=dict(
                    from_layer='c5',
                    upsample=dict(
                        type='Upsample',
                        scale_factor=2,
                        scale_bias=-1,
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                lateral=dict(from_layer='c4'),
                post=dict(
                    type='ConvModules',
                    in_channels=3072,  # 2048 + 1024
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='Relu', inplace=True),
                    num_convs=2,
                ),
                to_layer='p4',
            ),  # 16
            # model/decoder/blocks/block2
            dict(
                type='JunctionBlock',
                fusion_method='concat',
                top_down=dict(
                    from_layer='p4',
                    upsample=dict(
                        type='Upsample',
                        scale_factor=2,
                        scale_bias=-1,
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                lateral=dict(from_layer='c3'),
                post=dict(
                    type='ConvModules',
                    in_channels=768,  # 256 + 512
                    out_channels=128,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='Relu', inplace=True),
                    num_convs=2,
                ),
                to_layer='p3',
            ),  # 8
            # model/decoder/blocks/block3
            dict(
                type='JunctionBlock',
                fusion_method='concat',
                top_down=dict(
                    from_layer='p3',
                    upsample=dict(
                        type='Upsample',
                        scale_factor=2,
                        scale_bias=-1,
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                lateral=dict(from_layer='c2'),
                post=dict(
                    type='ConvModules',
                    in_channels=384,  # 128 + 256
                    out_channels=64,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='Relu', inplace=True),
                    num_convs=2,
                ),
                to_layer='p2',
            ),  # 4
            # model/decoder/blocks/block4
            dict(
                type='JunctionBlock',
                fusion_method='concat',
                top_down=dict(
                    from_layer='p2',
                    upsample=dict(
                        type='Upsample',
                        scale_factor=2,
                        scale_bias=-1,
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                lateral=dict(from_layer='c1'),
                post=dict(
                    type='ConvModules',
                    in_channels=128,  # 64 + 64
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='Relu', inplace=True),
                    num_convs=2,
                ),
                to_layer='p1',
            ),  # 2
            # model/decoder/blocks/block5
            dict(
                type='JunctionBlock',
                top_down=dict(
                    from_layer='p1',
                    upsample=dict(
                        type='Upsample',
                        scale_factor=2,
                        scale_bias=-1,
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                lateral=None,
                post=dict(
                    type='ConvModules',
                    in_channels=32,
                    out_channels=16,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='Relu', inplace=True),
                    num_convs=2,
                ),
                to_layer='p0',
            ),  # 1
        ]),
    # model/decoer/head
    head=dict(
        type='Head',
        in_channels=16,
        out_channels=nclasses,
        num_convs=0,
        upsample=dict(
            type='Upsample',
            scale_factor=1,
            scale_bias=0,
            mode='bilinear',
            align_corners=True
        ),
    )
)

## 3.1 resume
resume = None

# 4. criterion
criterion = dict(type='CrossEntropyLoss', ignore_index=ignore_label)

# 5. optim
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)

# 6. lr scheduler
max_epochs = 100
lr_scheduler = dict(
    type='PolyLR',
    warm_up=50,
    max_epochs=max_epochs)

# 7. runner
runner = dict(
    type='Runner',
    max_epochs=max_epochs,
    trainval_ratio=1,
    snapshot_interval=5,
)

# 8. device
gpu_id = '0'
