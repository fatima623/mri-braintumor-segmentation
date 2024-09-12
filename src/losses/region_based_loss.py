import torch
from src.losses.dice_loss import DiceLoss
from torch import nn



class RegionBasedDiceLoss3D(nn.Module):

    def __init__(self, classes: int, sigmoid_normalization: bool=True):

        super(RegionBasedDiceLoss3D, self).__init__()

        self.dice_loss = DiceLoss(classes=classes, sigmoid_normalization=sigmoid_normalization,
                                               eval_regions=False)

        self.dice_loss_region_based = DiceLoss(classes=classes, sigmoid_normalization=sigmoid_normalization,
                                               eval_regions=True)


