from __future__ import absolute_import, division, print_function
from . import model_monosceneflow as msf


import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

import segmentation_models_pytorch as smp

from .correlation_package.correlation import Correlation

from .modules_sceneflow import get_grid, WarpingLayer_SF
from .modules_sceneflow import initialize_msra, upsample_outputs_as
from .modules_sceneflow import upconv
from .modules_sceneflow import FeatureExtractor, MonoSceneFlowDecoder, ContextNetwork

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing

from collections import OrderedDict
from argparse import Namespace
import copy


class ModelSegmentation(nn.Module):
    """Model that combines monosceneflow and segmentation.

    The output from monosceneflow is used as input for the segmentation model,
    which classifies each pixel into one of three classes.
    """
    def __init__(self, in_channels):
        super(ModelSegmentation, self).__init__()




        # create segmentation model
        self.segmentation_model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            # use `imagenet` pre-trained weights for encoder initialization
            encoder_weights="imagenet",
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=in_channels,
            # model output channels (number of classes in your dataset)
            classes=4,
        )
        logging.info(f'Created segmentation model.')



    def forward(self, inputs):
        seg_out = self.segmentation_model(inputs)
        return seg_out


class ModelConvSegmentation(nn.Module):
    def __init__(self, in_channels=107, out_channels=3):
        super(ModelConvSegmentation, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.conv2 = nn.Conv2d(16, out_channels, 3, 1, 1)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        return out
