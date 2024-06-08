_base_ = [
    '../_base_/models/clip_rc.py',
    '../_base_/datasets/voc12_20_aug_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

img_size = 512
in_channels = 512
out_indices = [11]

region_level_bridge_size = 16

base_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
novel_class = [15, 16, 17, 18, 19]
both_class = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
]

pretrained = 'ViT-B-16.pt'

model = dict(
    type='CLIPRC',
    pretrained=pretrained,
    pretrained_text=pretrained,
    backbone=dict(
        type='CLIPVisionTransformerWithRLB',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=img_size,
        out_indices=out_indices,
        # setting of vpt
        num_tokens=10,
        prompt_dim=768,
        total_d_layer=11,
        # setting of RLB
        region_level_bridge_size=region_level_bridge_size,
        style='pytorch'),
    text_encoder=dict(type='CLIPTextEncoder',
                      context_length=77,
                      embed_dim=512,
                      transformer_width=512,
                      transformer_heads=8,
                      transformer_layers=12,
                      style='pytorch'),
    decode_head=dict(
        type='ATMSingleHeadSeg',
        img_size=img_size,
        in_channels=in_channels,
        seen_idx=base_class,
        all_idx=both_class,
        channels=in_channels,
        num_layers=3,
        num_classes=len(both_class),  # useless, pass to decode_head
        num_heads=8,
        use_proj=False,
        use_stages=len(out_indices),
        embed_dims=in_channels,
        loss_decode=dict(
            type='SegLoss',
            dec_layers=3,
            mask_weight=100.0,
            dice_weight=1.0,
            loss_weight=1.0,
            recovery_loss_weight=2.0,
        ),
    ),
    test_cfg=dict(mode='slide',
                  crop_size=(img_size, img_size),
                  stride=(426, 426)),
    base_class=base_class,
    novel_class=novel_class,
    both_class=both_class,
    self_training=True,
    ft_backbone=False,
    exclude_key='prompt',
    load_text_embedding=
    'configs/_base_/datasets/text_embedding/voc12_single.npy')

lr_config = dict(policy='poly',
                 power=0.9,
                 min_lr=1e-6,
                 by_epoch=False,
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)

optimizer = dict(type='AdamW',
                 lr=0.00002,
                 weight_decay=0.01,
                 paramwise_cfg=dict(
                     custom_keys={
                         'backbone': dict(lr_mult=10.0),
                         'text_encoder': dict(lr_mult=0.0),
                         'norm': dict(decay_mult=0.),
                         'ln': dict(decay_mult=0.),
                         'head': dict(lr_mult=10.),
                     }))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)
