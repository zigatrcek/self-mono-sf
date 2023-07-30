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
    def __init__(self, args: Namespace):
        super(ModelSegmentation, self).__init__()




        # create segmentation model
        self.segmentation_model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            # use `imagenet` pre-trained weights for encoder initialization
            encoder_weights="imagenet",
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=args.in_channels,
            # model output channels (number of classes in your dataset)
            classes=4,
        )
        logging.info(f'Created segmentation model.')



    def forward(self, inputs):
        seg_out = self.segmentation_model(inputs)
        return seg_out
