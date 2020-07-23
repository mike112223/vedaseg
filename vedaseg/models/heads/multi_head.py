import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

from vedaseg.utils import build_from_cfg
from ..utils import build_module, ConvModules
from .registry import HEADS
from ..weight_init import init_weights

logger = logging.getLogger()


@HEADS.register_module
class MultiHead(nn.Module):
    """Head

    Args:
    """
    def __init__(self,
                 in_channels,
                 out_mask_channels,
                 out_cls_channels,
                 inter_channels=None,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Relu', inplace=True),
                 num_convs=0,
                 upsample=None,
                 dropouts=None,
                 custom_init=None):
        super().__init__()

        if num_convs > 0:
            mask_layers = [
                ConvModules(in_channels,
                            inter_channels,
                            3,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            num_convs=num_convs,
                            dropouts=dropouts),
                nn.Conv2d(inter_channels, out_mask_channels, 1)
            ]
        else:
            mask_layers = [nn.Conv2d(in_channels, out_mask_channels, 1)]
        if upsample:
            upsample_layer = build_module(upsample)
            mask_layers.append(upsample_layer)

        if num_convs > 0:
            cls_layers = [
                ConvModules(in_channels,
                            inter_channels,
                            3,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            num_convs=num_convs,
                            dropouts=dropouts),
                nn.Conv2d(inter_channels, out_cls_channels, 1)
            ]
        else:
            cls_layers = [nn.Conv2d(in_channels, out_cls_channels, 1)]

        self.mask_block = nn.Sequential(*mask_layers)
        self.cls_block = nn.Sequential(*cls_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        logger.info('Head init weights')
        init_weights(self.modules())
        if custom_init is not None:
            custom_conv = -1
            if upsample:
                custom_conv = -2
            logger.info(f'Head init weights:{custom_init}')
            nn.init.constant_(self.mask_block[custom_conv].weight, 0)
            self.mask_block[custom_conv].bias = nn.Parameter(
                torch.Tensor(custom_init))  # noqa

    def forward(self, x):

        mask_feat = self.mask_block(x)
        cls_feat = self.cls_block(x)
        cls_feat = self.avgpool(cls_feat)
        cls_feat = torch.flatten(cls_feat, 1)

        return mask_feat, cls_feat
