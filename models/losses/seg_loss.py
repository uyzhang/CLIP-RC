import torch
import torch.nn as nn
from mmseg.models.builder import LOSSES
from .criterion import SegPlusCriterion


@LOSSES.register_module()
class SegLoss(nn.Module):
    '''
    modified from https://github.com/ZiqinZhou66/ZegCLIP/blob/main/models/losses/atm_loss.py 
    '''

    def __init__(self,
                 dec_layers,
                 mask_weight=20.0,
                 dice_weight=1.0,
                 loss_weight=1.0,
                 recovery_loss_weight=1.0):
        super(SegLoss, self).__init__()
        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight}
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({
                k + f"_{i}": v
                for k, v in weight_dict.items()
            })
        weight_dict.update(aux_weight_dict)

        self.criterion = SegPlusCriterion(
            weight_dict=weight_dict,
            losses=["masks"],
        )
        self.loss_weight = loss_weight

        # recovery loss
        self.recovery_loss = nn.L1Loss()
        self.recovery_loss_weight = recovery_loss_weight

    def forward(
        self,
        outputs,
        label,
        ignore_index=255,
    ):
        """Forward function."""

        self.ignore_index = ignore_index
        targets = self.prepare_targets(label)
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[
                    k] * self.loss_weight
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        # recovery loss
        ori_q = outputs['ori_q'].detach()
        ori_lateral = outputs['ori_lateral'].detach()
        q = outputs['q']
        lateral = outputs['lateral']
        losses['recovery_loss_q'] = self.recovery_loss(
            q, ori_q) * self.recovery_loss_weight
        losses['recovery_loss_k'] = self.recovery_loss(
            lateral, ori_lateral) * self.recovery_loss_weight

        return losses

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            # gt_cls
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != self.ignore_index]
            masks = []
            for cls in gt_cls:
                masks.append(targets_per_image == cls)
            if len(gt_cls) == 0:
                masks.append(targets_per_image == self.ignore_index)

            masks = torch.stack(masks, dim=0)
            new_targets.append({
                "labels": gt_cls,
                "target_masks": masks,
                "masks": targets_per_image,
            })
        return new_targets
