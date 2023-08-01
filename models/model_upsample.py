from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from .correlation_package.correlation import Correlation

from .modules_sceneflow import get_grid, WarpingLayer_SF
from .modules_sceneflow import initialize_msra, upsample_outputs_as, upsample_outputs_as_level
from .modules_sceneflow import upconv
from .modules_sceneflow import FeatureExtractor, MonoSceneFlowDecoder, ContextNetwork

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing


class ModelUpsample(nn.Module):
    """
    Upsample the layers of the FPN to the original image size.

    """
    def __init__(self, num_layers=5, reduce_dim=True):
        super(ModelUpsample, self).__init__()
        self.num_layers = num_layers
        self.upconv_layers = nn.ModuleList()
        self.reduce_dim = reduce_dim

        if not self.reduce_dim:
            self.upconv_layers = nn.ModuleList((
                upconv(256, 192, 3, 2),
                upconv(192, 128, 3, 2),
                upconv(128, 96, 3, 2),
                upconv(96, 64, 3, 2),
                upconv(64, 32, 3, 2),
                upconv(32, 16, 3, 2),
            ))

        else:
            self.ch_reduce = nn.Sequential(
                nn.Conv2d(81, 16, 1, 1, 0),
                nn.LeakyReLU(0.1, inplace=True),
            )


            for l in range(self.num_layers):
                if l == self.num_layers - 1:
                    self.upconv_layers.append(upconv(16, 16, 3, 4))
                else:
                    self.upconv_layers.append(upconv(16, 16, 3, 2))
            # print(f'Created upconv_layers: {self.upconv_layers}')

    def forward(self, inputs):
        outputs = []
        for i, data in enumerate(inputs):
            if self.reduce_dim:
                data = self.ch_reduce(data)
            for j in range(i, self.num_layers):
                data = self.upconv_layers[j](data)
            outputs.append(data)
        return outputs
