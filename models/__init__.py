from models.segmentor.clip_rc import CLIPRC

from models.backbone.text_encoder import CLIPTextEncoder
from models.backbone.img_encoder_rlb import CLIPVisionTransformerWithRLB
from models.decode_heads.decode_seg import ATMSingleHeadSeg
from models.losses.seg_loss import SegLoss

from configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
from configs._base_.datasets.dataloader.coco_stuff import ZeroCOCOStuffDataset
from configs._base_.datasets.dataloader.pascal_context import ZeroPascalContextDataset60