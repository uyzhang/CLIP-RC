img_size = 512
in_channels = 512
out_indices = [11]

model = dict(type='CLIPRC',
             pretrained='ViT-B-16.pt',
             backbone=dict(type='CLIPVisionTransformerWithRLB',
                           layers=12,
                           style='pytorch'),
             text_encoder=dict(type='CLIPTextEncoder',
                               context_length=77,
                               style='pytorch'),
             decode_head=dict(
                 type='ATMSingleHeadSeg',
                 img_size=img_size,
                 in_channels=in_channels,
                 channels=in_channels,
                 num_layers=3,
                 num_heads=8,
                 use_stages=len(out_indices),
                 embed_dims=in_channels // 2,
                 loss_decode=dict(type='SegLoss',
                                  dec_layers=3,
                                  loss_weight=1.0),
             ),
             train_cfg=dict(),
             test_cfg=dict(mode='slide',
                           crop_size=(512, 512),
                           stride=(426, 426)))
