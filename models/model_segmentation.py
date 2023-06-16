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


class ModelSegmentation(nn.Module):
    def __init__(self, args):
        self.mono_sf = msf.MonoSceneFlow(args)
        self.mono_sf.eval()

        self.segmentation_model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            # use `imagenet` pre-trained weights for encoder initialization
            encoder_weights="imagenet",
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=1,
            # model output channels (number of classes in your dataset)
            classes=3,
        )

    def forward(self, inputs):
        # inputs: [B, 3, H, W]
        # outputs: [B, 3, H, W]
        with torch.no_grad():
            mono_sf_out = self.mono_sf(inputs)
        # TODO: get relevant outputs
        seg_in = []
   
        seg_out = self.segmentation_model(seg_in)

        return seg_out
