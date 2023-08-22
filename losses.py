from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf

from models.forwardwarp_package.forward_warp import forward_warp
from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, reconstructImg, reconstructPts, projectSceneFlow2Flow
from utils.monodepth_eval import compute_errors, compute_d1_all
from models.modules_sceneflow import WarpingLayer_Flow

import logging
import traceback
import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt


###############################################
# Basic Module
###############################################

def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)


def _elementwise_l1(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=1, dim=1, keepdim=True)


def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)


def _SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

    return tf.pad(SSIM_img, pad=(1, 1, 1, 1), mode='constant', value=0)


def _apply_disparity(img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(
        batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(
        batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    # Disparity is passed in NCHW format with 1 channel
    x_shifts = disp[:, 0, :, :]
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = tf.grid_sample(img, 2 * flow_field - 1, mode='bilinear',
                            padding_mode='zeros', align_corners=True)

    return output


def _generate_image_left(img, disp):
    return _apply_disparity(img, -disp)


def _adaptive_disocc_detection(flow):

    # init mask
    b, _, h, w, = flow.size()
    mask = torch.ones(b, 1, h, w, dtype=flow.dtype,
                      device=flow.device).float().requires_grad_(False)
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1)
    disocc_map = (disocc > 0.5)

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(
            b, 1, h, w, dtype=torch.bool, device=flow.device).requires_grad_(False)

    return disocc_map


def _adaptive_disocc_detection_disp(disp):

    # # init
    b, _, h, w, = disp.size()
    mask = torch.ones(b, 1, h, w, dtype=disp.dtype,
                      device=disp.device).float().requires_grad_(False)
    flow = torch.zeros(b, 2, h, w, dtype=disp.dtype,
                       device=disp.device).float().requires_grad_(False)
    flow[:, 0:1, :, :] = disp * w
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1)
    disocc_map = (disocc > 0.5)

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(
            b, 1, h, w, dtype=torch.bool, device=disp.device).requires_grad_(False)

    return disocc_map


def _gradient_x(img):
    img = tf.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx


def _gradient_y(img):
    img = tf.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy


def _gradient_x_2nd(img):
    img_l = tf.pad(img, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]
    img_r = tf.pad(img, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]
    gx = img_l + img_r - 2 * img
    return gx


def _gradient_y_2nd(img):
    img_t = tf.pad(img, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :]
    img_b = tf.pad(img, (0, 0, 0, 1), mode="replicate")[:, :, 1:, :]
    gy = img_t + img_b - 2 * img
    return gy


def _smoothness_motion_2nd(sf, img, beta=1):
    sf_grad_x = _gradient_x_2nd(sf)
    sf_grad_y = _gradient_y_2nd(sf)

    img_grad_x = _gradient_x(img)
    img_grad_y = _gradient_y(img)
    weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x),
                          1, keepdim=True) * beta)
    weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y),
                          1, keepdim=True) * beta)

    smoothness_x = sf_grad_x * weights_x
    smoothness_y = sf_grad_y * weights_y

    return (smoothness_x.abs() + smoothness_y.abs())


def _disp2depth_kitti_K(disp, k_value):

    mask = (disp > 0).float()
    depth = k_value.unsqueeze(1).unsqueeze(
        1).unsqueeze(1) * 0.54 / (disp + (1.0 - mask))

    return depth


def _depth2disp_kitti_K(depth, k_value):

    disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth

    return disp


def _disp2depth(disp, calibration_matrix, mask=None, baseline=3.5677428730839563e+02):
    """Convert disparity to depth map

    Args:
        disp (torch.Tensor): Disparity map
        calibration_matrix (torch.Tensor): Stereo calibration matrix

    Returns:
        torch.Tensor: Depth map
    """
    mask = (disp > 0).float()
    # mask = (disp > 0).float()
    # logging.info(f'calibration_matrix: {calibration_matrix}')
    depth = (baseline * calibration_matrix.squeeze()[0, 0]) / (disp + (1.0 - mask))
    # depth = (baseline * calibration_matrix.squeeze()[0, 0]) / (disp + 1e-8)
    # depth = depth * mask
    # logging.info(
    #     f'avg: {depth.mean()}, min: {depth.min()}, max: {depth.max()}')

    return depth


def _sgbm_disp(img_l, img_r, stereo):
    l1_np = img_l[0].cpu().numpy().transpose(1, 2, 0) * 255.0
    r1_np = img_r[0].cpu().numpy().transpose(1, 2, 0) * 255.0

    l1_np = l1_np.astype(np.uint8)
    r1_np = r1_np.astype(np.uint8)

    return torch.tensor(stereo.compute(l1_np, r1_np).astype(np.float32) / 16.0).cuda().unsqueeze(0).unsqueeze(0)

###############################################
# Loss function
###############################################


class Loss_SceneFlow_SelfSup(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup, self).__init__()

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200

    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        # Photometric loss
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) +
                    _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        # Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(
            disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth, left_occ

    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

        _, _, h_dp, w_dp = sf_f.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        # scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp

        pts1, k1_scale = pixel2pts_ms(
            k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale = pixel2pts_ms(
            k_l2_aug, disp_l2, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp])

        pts2_warp = reconstructPts(coord1, pts2)
        pts1_warp = reconstructPts(coord2, pts1)

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        # Image reconstruction loss
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) +
                     _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) +
                     _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1[occ_map_f].mean()
        loss_im2 = img_diff2[occ_map_b].mean()
        img_diff1[~occ_map_f].detach_()
        img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2

        # Point reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(
            dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(
            dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        loss_pts1 = pts_diff1[occ_map_f].mean()
        loss_pts2 = pts_diff2[occ_map_b].mean()
        pts_diff1[~occ_map_f].detach_()
        pts_diff2[~occ_map_b].detach_()
        loss_pts = loss_pts1 + loss_pts2

        # 3D motion smoothness loss
        loss_3d_s = ((_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() +
                     (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean()) / (2 ** ii)

        # Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * \
            loss_pts + self._sf_3d_sm * loss_3d_s

        return sceneflow_loss, loss_im, loss_pts, loss_3d_s

    def detaching_grad_of_outputs(self, output_dict):

        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert (sf_f.size()[2:4] == sf_b.size()[2:4])
            assert (sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert (sf_f.size()[2:4] == disp_l2.size()[2:4])

            # For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            # Disp Loss
            loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(
                disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(
                disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + \
                (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

            # Sceneflow Loss
            loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b,
                                                                               disp_l1, disp_l2,
                                                                               disp_occ_l1, disp_occ_l2,
                                                                               k_l1_aug, k_l2_aug,
                                                                               img_l1_aug, img_l2_aug,
                                                                               aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]
            loss_sf_2d = loss_sf_2d + loss_im
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict


class Loss_SceneFlow_SemiSupFinetune(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SemiSupFinetune, self).__init__()

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._unsup_loss = Loss_SceneFlow_SelfSup(args)

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        unsup_loss_dict = self._unsup_loss(output_dict, target_dict)
        unsup_loss = unsup_loss_dict['total_loss']

        # Ground Truth
        gt_disp1 = target_dict['target_disp']
        gt_disp1_mask = (target_dict['target_disp_mask'] == 1).float()
        gt_disp2 = target_dict['target_disp2_occ']
        gt_disp2_mask = (target_dict['target_disp2_mask_occ'] == 1).float()
        gt_flow = target_dict['target_flow']
        gt_flow_mask = (target_dict['target_flow_mask'] == 1).float()

        b, _, h_dp, w_dp = gt_disp1.size()

        disp_loss = 0
        flow_loss = 0

        for ii, sf_f in enumerate(output_dict['flow_f_pp']):

            # disp1
            disp_l1 = interpolate2d_as(
                output_dict["disp_l1_pp"][ii], gt_disp1, mode="bilinear") * w_dp
            valid_abs_rel = torch.abs(gt_disp1 - disp_l1) * gt_disp1_mask
            valid_abs_rel[gt_disp1_mask == 0].detach_()
            disp_l1_loss = valid_abs_rel[gt_disp1_mask != 0].mean()

            # Flow Loss
            sf_f_up = interpolate2d_as(sf_f, gt_flow, mode="bilinear")
            out_flow = projectSceneFlow2Flow(
                target_dict['input_k_l1'], sf_f_up, disp_l1)
            valid_epe = _elementwise_robust_epe_char(
                out_flow, gt_flow) * gt_flow_mask

            valid_epe[gt_flow_mask == 0].detach_()
            flow_l1_loss = valid_epe[gt_flow_mask != 0].mean()

            # disp1_next
            out_depth_l1 = _disp2depth_kitti_K(
                disp_l1, target_dict['input_k_l1'][:, 0, 0])
            out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
            out_depth_l1_next = out_depth_l1 + sf_f_up[:, 2:3, :, :]
            disp_l1_next = _depth2disp_kitti_K(
                out_depth_l1_next, target_dict['input_k_l1'][:, 0, 0])

            valid_abs_rel = torch.abs(gt_disp2 - disp_l1_next) * gt_disp2_mask
            valid_abs_rel[gt_disp2_mask == 0].detach_()
            disp_l2_loss = valid_abs_rel[gt_disp2_mask != 0].mean()

            disp_loss = disp_loss + \
                (disp_l1_loss + disp_l2_loss) * self._weights[ii]
            flow_loss = flow_loss + flow_l1_loss * self._weights[ii]

        # finding weight
        u_loss = unsup_loss.detach()
        d_loss = disp_loss.detach()
        f_loss = flow_loss.detach()

        max_val = torch.max(torch.max(f_loss, d_loss), u_loss)

        u_weight = max_val / u_loss
        d_weight = max_val / d_loss
        f_weight = max_val / f_loss

        total_loss = unsup_loss * u_weight + disp_loss * d_weight + flow_loss * f_weight
        loss_dict["unsup_loss"] = unsup_loss
        loss_dict["dp_loss"] = disp_loss
        loss_dict["fl_loss"] = flow_loss
        loss_dict["total_loss"] = total_loss

        return loss_dict


###############################################
# Eval
###############################################

def eval_module_disp_depth(gt_disp, gt_disp_mask, output_disp, gt_depth, output_depth):

    loss_dict = {}
    batch_size = gt_disp.size(0)
    gt_disp_mask_f = gt_disp_mask.float()

    # KITTI disparity metric
    d_valid_epe = _elementwise_epe(output_disp, gt_disp) * gt_disp_mask_f
    d_outlier_epe = (d_valid_epe > 3).float() * \
        ((d_valid_epe / gt_disp) > 0.05).float() * gt_disp_mask_f
    loss_dict["otl"] = (d_outlier_epe.view(
        batch_size, -1).sum(1)).mean() / 91875.68
    loss_dict["otl_img"] = d_outlier_epe

    logging.info("KITTI otl: {}".format(loss_dict["otl"]))

    # MonoDepth metric
    # abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = compute_errors(gt_depth[gt_disp_mask], output_depth[gt_disp_mask])
    abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = compute_errors(
        gt_disp[gt_disp_mask], output_disp[gt_disp_mask])
    loss_dict["abs_rel"] = abs_rel
    loss_dict["sq_rel"] = sq_rel
    loss_dict["rms"] = rms
    loss_dict["log_rms"] = log_rms
    loss_dict["a1"] = a1
    loss_dict["a2"] = a2
    loss_dict["a3"] = a3

    logging.info("MonoDepth absrel: {}".format(loss_dict["abs_rel"]))

    return loss_dict


class Eval_MonoDepth_Eigen(nn.Module):
    def __init__(self):
        super(Eval_MonoDepth_Eigen, self).__init__()

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        # Depth Eval
        gt_depth = target_dict['target_depth']

        out_disp_l1 = interpolate2d_as(
            output_dict["disp_l1_pp"][0], gt_depth, mode="bilinear") * gt_depth.size(3)
        out_depth_l1 = _disp2depth_kitti_K(
            out_disp_l1, target_dict['input_k_l1'][:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
        gt_depth_mask = (gt_depth > 1e-3) * (gt_depth < 80)

        # Compute metrics
        abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = compute_errors(
            gt_depth[gt_depth_mask], out_depth_l1[gt_depth_mask])

        output_dict["out_disp_l_pp"] = out_disp_l1
        output_dict["out_depth_l_pp"] = out_depth_l1
        loss_dict["ab_r"] = abs_rel
        loss_dict["sq_r"] = sq_rel
        loss_dict["rms"] = rms
        loss_dict["log_rms"] = log_rms
        loss_dict["a1"] = a1
        loss_dict["a2"] = a2
        loss_dict["a3"] = a3

        return loss_dict


class Eval_SceneFlow_KITTI_Test(nn.Module):
    def __init__(self):
        super(Eval_SceneFlow_KITTI_Test, self).__init__()

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ##################################################
        # Depth 1
        ##################################################
        input_l1 = target_dict['input_l1']
        intrinsics = target_dict['input_k_l1']
        # logging.info(target_dict['input_k_l1'])

        out_disp_l1 = interpolate2d_as(
            output_dict["disp_l1_pp"][0], input_l1, mode="bilinear") * input_l1.size(3)
        out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
        output_dict["out_disp_l_pp"] = out_disp_l1

        ##################################################
        # Optical Flow Eval
        ##################################################
        out_sceneflow = interpolate2d_as(
            output_dict['flow_f_pp'][0], input_l1, mode="bilinear")
        out_flow = projectSceneFlow2Flow(
            target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])
        output_dict["out_flow_pp"] = out_flow

        ##################################################
        # Depth 2
        ##################################################
        out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
        out_disp_l1_next = _depth2disp_kitti_K(
            out_depth_l1_next, intrinsics[:, 0, 0])
        output_dict["out_disp_l_pp_next"] = out_disp_l1_next

        loss_dict['sf'] = (out_disp_l1_next).sum()

        return loss_dict


class Eval_SceneFlow_KITTI_Train(nn.Module):
    def __init__(self, args):
        super(Eval_SceneFlow_KITTI_Train, self).__init__()

    def forward(self, output_dict, target_dict):
        logging.info("Evaluating SceneFlow KITTI Train")
        # logging.info("output_dict: {}".format(output_dict.keys()))
        # logging.info("target_dict: {}".format(target_dict.keys()))

        loss_dict = {}

        gt_flow = target_dict['target_flow']
        gt_flow_mask = (target_dict['target_flow_mask'] == 1).float()

        gt_disp = target_dict['target_disp']
        gt_disp_mask = (target_dict['target_disp_mask'] == 1).float()

        gt_disp2_occ = target_dict['target_disp2_occ']
        gt_disp2_mask = (target_dict['target_disp2_mask_occ'] == 1).float()

        # logging.info(f'gt_flow_mask: {gt_flow_mask.shape}')
        # logging.info(f'gt_disp_mask: {gt_disp_mask.shape}')
        # logging.info(f'gt_disp2_mask: {gt_disp2_mask.shape}')

        gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

        intrinsics = target_dict['input_k_l1']

        ##################################################
        # Depth 1
        ##################################################

        batch_size, _, _, width = gt_disp.size()

        # logging.info(f'disp_l1_pp shape: {output_dict["disp_l1_pp"][0].shape}')

        out_disp_l1 = interpolate2d_as(
            output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
        # logging.info(f'out_disp_l1 shape: {out_disp_l1.shape}')
        out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
        gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

        dict_disp0_occ = eval_module_disp_depth(
            gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)

        output_dict["out_disp_l_pp"] = out_disp_l1
        output_dict["out_depth_l_pp"] = out_depth_l1

        d0_outlier_image = dict_disp0_occ['otl_img']
        loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
        loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
        loss_dict["d1"] = dict_disp0_occ['otl']

        ##################################################
        # Optical Flow Eval
        ##################################################

        out_sceneflow = interpolate2d_as(
            output_dict['flow_f_pp'][0], gt_flow, mode="bilinear")
        out_flow = projectSceneFlow2Flow(
            target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

        # Flow Eval
        valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
        loss_dict["f_epe"] = (valid_epe.view(
            batch_size, -1).sum(1)).mean() / 91875.68
        output_dict["out_flow_pp"] = out_flow

        flow_gt_mag = torch.norm(
            target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
        flow_outlier_epe = (valid_epe > 3).float(
        ) * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
        loss_dict["f1"] = (flow_outlier_epe.view(
            batch_size, -1).sum(1)).mean() / 91875.68

        ##################################################
        # Depth 2
        ##################################################

        out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
        out_disp_l1_next = _depth2disp_kitti_K(
            out_depth_l1_next, intrinsics[:, 0, 0])
        gt_depth_l1_next = _disp2depth_kitti_K(
            gt_disp2_occ, intrinsics[:, 0, 0])

        dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(
        ), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)

        output_dict["out_disp_l_pp_next"] = out_disp_l1_next
        output_dict["out_depth_l_pp_next"] = out_depth_l1_next

        d1_outlier_image = dict_disp1_occ['otl_img']
        loss_dict["d2"] = dict_disp1_occ['otl']

        ##################################################
        # Scene Flow Eval
        ##################################################

        outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() +
                      d1_outlier_image.bool()).float() * gt_sf_mask
        loss_dict["sf"] = (outlier_sf.view(
            batch_size, -1).sum(1)).mean() / 91873.4

        return loss_dict


###############################################
# Ablation - Loss_SceneFlow_SelfSup
###############################################

class Loss_SceneFlow_SelfSup_NoOcc(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_NoOcc, self).__init__()

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200

    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        # left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        # Photometric loss:
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) +
                    _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_img = img_diff.mean()
        # loss_img = (img_diff[left_occ]).mean()
        # img_diff[~left_occ].detach_()

        # Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(
            disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth  # , left_occ

    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

        # Depth2Pts
        _, _, h_dp, w_dp = sf_f.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        # scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp

        pts1, k1_scale = pixel2pts_ms(
            k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale = pixel2pts_ms(
            k_l2_aug, disp_l2, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp])

        pts2_warp = reconstructPts(coord1, pts2)
        pts1_warp = reconstructPts(coord2, pts1)

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        # occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        # occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        # Image reconstruction loss
        # img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
        # img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) +
                     _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) +
                     _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1.mean()
        loss_im2 = img_diff2.mean()
        # loss_im1 = img_diff1[occ_map_f].mean()
        # loss_im2 = img_diff2[occ_map_b].mean()
        # img_diff1[~occ_map_f].detach_()
        # img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2

        # Point Reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(
            dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(
            dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        loss_pts1 = pts_diff1.mean()
        loss_pts2 = pts_diff2.mean()
        # loss_pts1 = pts_diff1[occ_map_f].mean()
        # loss_pts2 = pts_diff2[occ_map_b].mean()
        # pts_diff1[~occ_map_f].detach_()
        # pts_diff2[~occ_map_b].detach_()
        loss_pts = loss_pts1 + loss_pts2

        # 3D motion smoothness loss
        loss_3d_s = ((_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() +
                     (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean()) / (2 ** ii)

        # Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * \
            loss_pts + self._sf_3d_sm * loss_3d_s

        return sceneflow_loss, loss_im, loss_pts, loss_3d_s

    def detaching_grad_of_outputs(self, output_dict):

        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        # SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert (sf_f.size()[2:4] == sf_b.size()[2:4])
            assert (sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert (sf_f.size()[2:4] == disp_l2.size()[2:4])

            # For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            # Depth Loss
            loss_disp_l1 = self.depth_loss_left_img(
                disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2 = self.depth_loss_left_img(
                disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + \
                (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

            # Sceneflow Loss
            loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b,
                                                                               disp_l1, disp_l2,
                                                                               k_l1_aug, k_l2_aug,
                                                                               img_l1_aug, img_l2_aug,
                                                                               aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]
            loss_sf_2d = loss_sf_2d + loss_im
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict


class Loss_SceneFlow_SelfSup_NoPts(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_NoPts, self).__init__()

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200

    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        # Photometric loss:
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) +
                    _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        # Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(
            disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth, left_occ

    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

        # Depth2Pts
        _, _, h_dp, w_dp = sf_f.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        # scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp

        pts1, k1_scale = pixel2pts_ms(
            k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale = pixel2pts_ms(
            k_l2_aug, disp_l2, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp])

        # pts2_warp = reconstructPts(coord1, pts2)
        # pts1_warp = reconstructPts(coord2, pts1)

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        # Image reconstruction loss
        # img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
        # img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) +
                     _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) +
                     _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1[occ_map_f].mean()
        loss_im2 = img_diff2[occ_map_b].mean()
        img_diff1[~occ_map_f].detach_()
        img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2

        # ## Point Reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        # pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        # pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        # loss_pts1 = pts_diff1[occ_map_f].mean()
        # loss_pts2 = pts_diff2[occ_map_b].mean()
        # pts_diff1[~occ_map_f].detach_()
        # pts_diff2[~occ_map_b].detach_()
        # loss_pts = loss_pts1 + loss_pts2

        # 3D motion smoothness loss
        loss_3d_s = ((_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() +
                     (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean()) / (2 ** ii)

        # Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_sm * \
            loss_3d_s  # + self._sf_3d_pts * loss_pts

        return sceneflow_loss, loss_im, loss_3d_s  # , loss_pts

    def detaching_grad_of_outputs(self, output_dict):

        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        # SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        # loss_sf_3d = 0
        loss_sf_sm = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert (sf_f.size()[2:4] == sf_b.size()[2:4])
            assert (sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert (sf_f.size()[2:4] == disp_l2.size()[2:4])

            # For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            # Depth Loss
            loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(
                disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(
                disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + \
                (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

            # Sceneflow Loss
            loss_sceneflow, loss_im, loss_3d_s = self.sceneflow_loss(sf_f, sf_b,
                                                                     disp_l1, disp_l2,
                                                                     disp_occ_l1, disp_occ_l2,
                                                                     k_l1_aug, k_l2_aug,
                                                                     img_l1_aug, img_l2_aug,
                                                                     aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]
            loss_sf_2d = loss_sf_2d + loss_im
            # loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        # loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict


class Loss_SceneFlow_SelfSup_NoPtsNoOcc(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_NoPtsNoOcc, self).__init__()

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200

    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        # left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        # Photometric loss:
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) +
                    _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_img = img_diff.mean()
        # loss_img = (img_diff[left_occ]).mean()
        # img_diff[~left_occ].detach_()

        # Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(
            disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth  # , left_occ

    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

        # Depth2Pts
        _, _, h_dp, w_dp = sf_f.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        # scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp

        pts1, k1_scale = pixel2pts_ms(
            k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale = pixel2pts_ms(
            k_l2_aug, disp_l2, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp])

        # pts2_warp = reconstructPts(coord1, pts2)
        # pts1_warp = reconstructPts(coord2, pts1)

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        # occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        # occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        # Image reconstruction loss
        # img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
        # img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) +
                     _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) +
                     _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1.mean()
        loss_im2 = img_diff2.mean()
        # loss_im1 = img_diff1[occ_map_f].mean()
        # loss_im2 = img_diff2[occ_map_b].mean()
        # img_diff1[~occ_map_f].detach_()
        # img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2

        # Point Reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        # pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        # pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        # loss_pts1 = pts_diff1.mean()
        # loss_pts2 = pts_diff2.mean()
        # loss_pts1 = pts_diff1[occ_map_f].mean()
        # loss_pts2 = pts_diff2[occ_map_b].mean()
        # pts_diff1[~occ_map_f].detach_()
        # pts_diff2[~occ_map_b].detach_()
        # loss_pts = loss_pts1 + loss_pts2

        # 3D motion smoothness loss
        loss_3d_s = ((_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() +
                     (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean()) / (2 ** ii)

        # Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_sm * \
            loss_3d_s  # + self._sf_3d_pts * loss_pts

        return sceneflow_loss, loss_im, loss_3d_s  # , loss_pts

    def detaching_grad_of_outputs(self, output_dict):

        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        # SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        # loss_sf_3d = 0
        loss_sf_sm = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert (sf_f.size()[2:4] == sf_b.size()[2:4])
            assert (sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert (sf_f.size()[2:4] == disp_l2.size()[2:4])

            # For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            # Depth Loss
            loss_disp_l1 = self.depth_loss_left_img(
                disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2 = self.depth_loss_left_img(
                disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + \
                (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

            # Sceneflow Loss
            loss_sceneflow, loss_im, loss_3d_s = self.sceneflow_loss(sf_f, sf_b,
                                                                     disp_l1, disp_l2,
                                                                     k_l1_aug, k_l2_aug,
                                                                     img_l1_aug, img_l2_aug,
                                                                     aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]
            loss_sf_2d = loss_sf_2d + loss_im
            # loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        # loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict


###############################################
# Ablation - Separate Decoder
###############################################

class Loss_Flow_Only(nn.Module):
    def __init__(self):
        super(Loss_Flow_Only, self).__init__()

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._warping_layer = WarpingLayer_Flow()

    def forward(self, output_dict, target_dict):

        # Loss
        total_loss = 0
        loss_sf_2d = 0
        loss_sf_sm = 0

        for ii, (sf_f, sf_b) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'])):

            # Depth2Pts
            img_l1 = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2 = interpolate2d_as(target_dict["input_l2_aug"], sf_b)

            img_l2_warp = self._warping_layer(img_l2, sf_f)
            img_l1_warp = self._warping_layer(img_l1, sf_b)
            occ_map_f = _adaptive_disocc_detection(sf_b).detach()
            occ_map_b = _adaptive_disocc_detection(sf_f).detach()

            img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) +
                         _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
            img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) +
                         _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
            loss_im1 = img_diff1[occ_map_f].mean()
            loss_im2 = img_diff2[occ_map_b].mean()
            img_diff1[~occ_map_f].detach_()
            img_diff2[~occ_map_b].detach_()
            loss_im = loss_im1 + loss_im2

            loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean(
            ) + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()

            total_loss = total_loss + \
                (loss_im + 10.0 * loss_smooth) * self._weights[ii]

            loss_sf_2d = loss_sf_2d + loss_im
            loss_sf_sm = loss_sf_sm + loss_smooth

        loss_dict = {}
        loss_dict["ofd2"] = loss_sf_2d
        loss_dict["ofs2"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        return loss_dict


class Eval_Flow_Only(nn.Module):
    def __init__(self):
        super(Eval_Flow_Only, self).__init__()

    def upsample_flow_as(self, flow, output_as):
        size_inputs = flow.size()[2:4]
        size_targets = output_as.size()[2:4]
        resized_flow = tf.interpolate(
            flow, size=size_targets, mode="bilinear", align_corners=True)
        # correct scaling of flow
        u, v = resized_flow.chunk(2, dim=1)
        u *= float(size_targets[1] / size_inputs[1])
        v *= float(size_targets[0] / size_inputs[0])
        return torch.cat([u, v], dim=1)

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        im_l1 = target_dict['input_l1']
        batch_size, _, _, _ = im_l1.size()

        gt_flow = target_dict['target_flow']
        gt_flow_mask = target_dict['target_flow_mask']

        # Flow EPE
        out_flow = self.upsample_flow_as(output_dict['flow_f'][0], gt_flow)
        valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask.float()
        loss_dict["epe"] = (valid_epe.view(
            batch_size, -1).sum(1)).mean() / 91875.68

        flow_gt_mag = torch.norm(
            target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
        outlier_epe = (valid_epe > 3).float() * \
            ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
        loss_dict["f1"] = (outlier_epe.view(
            batch_size, -1).sum(1)).mean() / 91875.68

        output_dict["out_flow_pp"] = out_flow

        return loss_dict


class Loss_Disp_Only(nn.Module):
    def __init__(self, args):
        super(Loss_Disp_Only, self).__init__()

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1

    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        # Image loss:
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) +
                    _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        # Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(
            disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth, left_occ

    def detaching_grad_of_outputs(self, output_dict):

        for ii in range(0, len(output_dict['disp_l1'])):
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        # SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_dp_sum = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert (disp_l1.size()[2:4] == disp_l2.size()[2:4])

            # For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], disp_l1)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], disp_l2)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], disp_l1)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], disp_l2)

            # Depth Loss
            loss_disp_l1, _ = self.depth_loss_left_img(
                disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, _ = self.depth_loss_left_img(
                disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + \
                (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

        total_loss = loss_dp_sum

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict


class Eval_Disp_Only(nn.Module):
    def __init__(self):
        super(Eval_Disp_Only, self).__init__()

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        # Depth Eval
        gt_disp = target_dict['target_disp']
        gt_disp_mask = (target_dict['target_disp_mask'] == 1)
        intrinsics = target_dict['input_k_l1']

        out_disp_l1 = interpolate2d_as(
            output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * gt_disp.size(3)
        out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
        gt_depth_pp = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

        output_dict_displ = eval_module_disp_depth(
            gt_disp, gt_disp_mask, out_disp_l1, gt_depth_pp, out_depth_l1)

        output_dict["out_disp_l_pp"] = out_disp_l1
        output_dict["out_depth_l_pp"] = out_depth_l1

        loss_dict["d1"] = output_dict_displ['otl']

        loss_dict["ab"] = output_dict_displ['abs_rel']
        loss_dict["sq"] = output_dict_displ['sq_rel']
        loss_dict["rms"] = output_dict_displ['rms']
        loss_dict["lrms"] = output_dict_displ['log_rms']
        loss_dict["a1"] = output_dict_displ['a1']
        loss_dict["a2"] = output_dict_displ['a2']
        loss_dict["a3"] = output_dict_displ['a3']

        return loss_dict


###############################################
# MonoDepth Experiment
###############################################

class Basis_MonoDepthLoss(nn.Module):
    def __init__(self):
        super(Basis_MonoDepthLoss, self).__init__()
        self.ssim_w = 0.85
        self.disp_gradient_w = 0.1
        self.lr_w = 1.0
        self.n = 4

    def scale_pyramid(self, img_input, depths):
        scaled_imgs = []
        for _, depth in enumerate(depths):
            scaled_imgs.append(interpolate2d_as(img_input, depth))
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = tf.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = tf.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(
            batch_size, height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(
            batch_size, width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        # Disparity is passed in NCHW format with 1 channel
        x_shifts = disp[:, 0, :, :]
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = tf.grid_sample(
            img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros', align_corners=True)

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

        return tf.pad(SSIM_img, pad=(1, 1, 1, 1), mode='constant', value=0)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
                     for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
                     for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i]) for i in range(self.n)]

    def forward(self, disp_l, disp_r, img_l, img_r):

        self.n = len(disp_l)

        # Image pyramid
        img_l_pyramid = self.scale_pyramid(img_l, disp_l)
        img_r_pyramid = self.scale_pyramid(img_r, disp_r)

        # Disocc map
        right_occ = [
            _adaptive_disocc_detection_disp(-disp_l[i]) for i in range(self.n)]
        left_occ = [_adaptive_disocc_detection_disp(
            disp_r[i]) for i in range(self.n)]

        # Image reconstruction loss
        left_est = [self.generate_image_left(
            img_r_pyramid[i], disp_l[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(
            img_l_pyramid[i], disp_r[i]) for i in range(self.n)]

        # L1
        l1_left = [torch.mean((torch.abs(left_est[i] - img_l_pyramid[i])
                               ).mean(dim=1, keepdim=True)[left_occ[i]]) for i in range(self.n)]
        l1_right = [torch.mean((torch.abs(right_est[i] - img_r_pyramid[i])).mean(
            dim=1, keepdim=True)[right_occ[i]]) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean((self.SSIM(left_est[i], img_l_pyramid[i])).mean(
            dim=1, keepdim=True)[left_occ[i]]) for i in range(self.n)]
        ssim_right = [torch.mean((self.SSIM(right_est[i], img_r_pyramid[i])).mean(
            dim=1, keepdim=True)[right_occ[i]]) for i in range(self.n)]

        image_loss_left = [self.ssim_w * ssim_left[i] +
                           (1 - self.ssim_w) * l1_left[i] for i in range(self.n)]
        image_loss_right = [self.ssim_w * ssim_right[i] +
                            (1 - self.ssim_w) * l1_right[i] for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency loss
        right_left_disp = [self.generate_image_left(
            disp_r[i], disp_l[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(
            disp_l[i], disp_r[i]) for i in range(self.n)]

        lr_left_loss = [torch.mean(
            (torch.abs(right_left_disp[i] - disp_l[i]))[left_occ[i]]) for i in range(self.n)]
        lr_right_loss = [torch.mean(
            (torch.abs(left_right_disp[i] - disp_r[i]))[right_occ[i]]) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_l, img_l_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_r, img_r_pyramid)

        disp_left_loss = [torch.mean(
            torch.abs(disp_left_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_right_loss = [torch.mean(
            torch.abs(disp_right_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        # Loss sum
        loss = image_loss + self.disp_gradient_w * \
            disp_gradient_loss + self.lr_w * lr_loss

        return loss


class Loss_MonoDepth(nn.Module):
    def __init__(self):

        super(Loss_MonoDepth, self).__init__()
        self._depth_loss = Basis_MonoDepthLoss()

    def forward(self, output_dict, target_dict):

        loss_dict = {}
        depth_loss = self._depth_loss(
            output_dict['disp_l1'], output_dict['disp_r1'], target_dict['input_l1'], target_dict['input_r1'])
        loss_dict['total_loss'] = depth_loss

        return loss_dict


class Eval_MonoDepth(nn.Module):
    def __init__(self):
        super(Eval_MonoDepth, self).__init__()

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        # Depth Eval
        gt_disp = target_dict['target_disp']
        gt_disp_mask = (target_dict['target_disp_mask'] == 1)
        intrinsics = target_dict['input_k_l1_orig']

        out_disp_l_pp = interpolate2d_as(
            output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * gt_disp.size(3)
        out_depth_l_pp = _disp2depth_kitti_K(
            out_disp_l_pp, intrinsics[:, 0, 0])
        out_depth_l_pp = torch.clamp(out_depth_l_pp, 1e-3, 80)
        gt_depth_pp = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

        output_dict_displ = eval_module_disp_depth(
            gt_disp, gt_disp_mask, out_disp_l_pp, gt_depth_pp, out_depth_l_pp)

        output_dict["out_disp_l_pp"] = out_disp_l_pp
        output_dict["out_depth_l_pp"] = out_depth_l_pp
        loss_dict["ab_r"] = output_dict_displ['abs_rel']
        loss_dict["sq_r"] = output_dict_displ['sq_rel']

        return loss_dict

###############################################


class Eval_Monodepth_MODD2(nn.Module):
    def __init__(self):
        window_size = 1
        min_disp = 0
        num_disp = 160-min_disp
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=-1,
            preFilterCap=0,
            uniquenessRatio=30,
            # uniquenessRatio = 0,
            speckleWindowSize=0,
            speckleRange=0
        )
        super(Eval_Monodepth_MODD2, self).__init__()

    def forward(self, output_dict, target_dict):
        performance_metrics = {}
        cmap = copy.copy(plt.cm.get_cmap('plasma'))
        cmap.set_under('black')

        plt.subplot(3, 3, 1)
        plt.title('left 1')
        plt.imshow(target_dict['input_l1'][0].permute(1, 2, 0).cpu().numpy())
        plt.subplot(3, 3, 2)
        plt.title('right 1')
        plt.imshow(target_dict['input_r1'][0].permute(1, 2, 0).cpu().numpy())
        plt.subplot(3, 3, 3)
        plt.title('left 2')
        plt.show()
        plt.imshow(target_dict['input_l2'][0].permute(1, 2, 0).cpu().numpy())
        plt.subplot(3, 3, 4)
        plt.title('sgbm pred')
        # seg_l1 = target_dict['seg_l1']
        # seg_sky_mask = seg_l1[:, 1, :, :].unsqueeze(0)
        # seg_sky_mask = interpolate2d_as(seg_sky_mask, target_dict['input_l1'], mode="bilinear")
        # seg_no_sky_mask = (seg_sky_mask==0).cuda()

        # Depth Eval
        # logging.info(f'output_dict: {output_dict.keys()}')
        # logging.info(f'target_dict: {target_dict.keys()}')
        # seg_3_channel_no_sky_mask = seg_no_sky_mask.repeat(1, 3, 1, 1).int().float()

        l1_pp = target_dict['input_l1']
        r1_pp = target_dict['input_r1']
        pseudo_gt_disp = _sgbm_disp(l1_pp, r1_pp, self.stereo)
        pseudo_gt_disp = interpolate2d_as(
            pseudo_gt_disp, l1_pp, mode="bilinear")

        pseudo_gt_disp_mask = (pseudo_gt_disp > 0)
        # full_mask = pseudo_gt_disp_mask.logical_and(seg_no_sky_mask)
        full_mask = pseudo_gt_disp_mask
        full_mask = full_mask.int().float()
        pseudo_gt_disp = pseudo_gt_disp * full_mask
        # pseudo_gt_disp = (pseudo_gt_disp - pseudo_gt_disp.mean()) / pseudo_gt_disp.std()
        plt.imshow(pseudo_gt_disp[0, 0].cpu().numpy(), cmap=cmap)

        intrinsics = target_dict['input_k_l1_flip_aug']
        # plt.imshow(pseudo_gt_disp[0, 0].cpu().numpy())

        plt.subplot(3, 3, 5)
        plt.title('self-mono-sf pred')

        out_disp_l_pp = interpolate2d_as(
            output_dict["disp_l1_pp"][0], pseudo_gt_disp, mode="bilinear") * pseudo_gt_disp.size(3)
        out_disp_l_pp_masked = out_disp_l_pp * full_mask
        # out_disp_l_pp_masked = (out_disp_l_pp_masked - out_disp_l_pp_masked.mean()) / out_disp_l_pp_masked.std()
        plt.imshow(out_disp_l_pp_masked[0, 0].cpu().numpy(), cmap=cmap)

        plt.subplot(3, 3, 6)
        plt.title('pred unmasked')
        plt.imshow(out_disp_l_pp[0, 0].cpu().numpy(), cmap=cmap)

        full_depth_mask = full_mask.logical_and(
            out_disp_l_pp_masked > 0).logical_and(pseudo_gt_disp > 0).int().float()

        out_depth_l_pp = _disp2depth(out_disp_l_pp_masked, intrinsics)
        # out_depth_l_pp = torch.clamp(out_depth_l_pp, 1e-3, 80)
        gt_depth_pp = _disp2depth(pseudo_gt_disp, intrinsics)
        # gt_depth_pp = torch.clamp(gt_depth_pp, 1e-3, 80)
        plt.subplot(3, 3, 7)
        plt.title('SGBM depth')
        plt.imshow(gt_depth_pp[0, 0].cpu().numpy(), cmap=cmap, vmin=1)
        plt.subplot(3, 3, 8)
        plt.title('pred depth')
        plt.imshow(out_depth_l_pp[0, 0].cpu().numpy(), cmap=cmap, vmin=1)
        plt.subplot(3, 3, 9)
        plt.title('pred depth unmasked')
        out_depth_sky_mask = _disp2depth(out_disp_l_pp, intrinsics)
        # out_depth_sky_mask = out_depth_sky_mask / out_depth_sky_mask.max()
        plt.imshow(out_depth_sky_mask[0, 0].cpu().numpy(), cmap=cmap, vmin=1)
        plt.show()

        output_dict_displ = eval_module_disp_depth(
            pseudo_gt_disp, pseudo_gt_disp_mask, out_disp_l_pp, gt_depth_pp, out_depth_l_pp)

        output_dict["out_disp_l_pp"] = out_disp_l_pp
        output_dict["out_depth_l_pp"] = out_depth_l_pp
        # logging.info(f'output_dict_displ: {output_dict_displ.keys()}')
        performance_metrics["full_ab_r"] = output_dict_displ['abs_rel']
        performance_metrics["full_sq_r"] = output_dict_displ['sq_rel']
        performance_metrics["full_rms"] = output_dict_displ['rms']
        performance_metrics["full_lrms"] = output_dict_displ['log_rms']
        performance_metrics["full_a1"] = output_dict_displ['a1']
        performance_metrics["full_a2"] = output_dict_displ['a2']
        performance_metrics["full_a3"] = output_dict_displ['a3']
        print(performance_metrics)

        return performance_metrics


###############################################


class Eval_MonoDepth_MODS(nn.Module):
    def __init__(self):
        window_size = 1
        min_disp = 0
        num_disp = 150-min_disp
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=-1,
            preFilterCap=0,
            uniquenessRatio=40,
            # uniquenessRatio = 0,
            speckleWindowSize=0,
            speckleRange=0
        )
        super(Eval_MonoDepth_MODS, self).__init__()

    def forward(self, output_dict, target_dict):
        performance_metrics = {}
        cmap = copy.copy(plt.cm.get_cmap('plasma'))
        cmap.set_under('black')

        vis = True

        gt_disp = target_dict['disp']


        _, _, h, w = target_dict['input_l1'].size()

        seg_l1 = target_dict['seg_l1']
        seg_sky_mask = seg_l1[:, 1, :, :].unsqueeze(0)
        seg_sky_mask = tf.interpolate(seg_sky_mask, [h, w], mode='nearest')
        # seg_sky_mask = interpolate2d_as(seg_sky_mask, target_dict['input_l1'], mode="bilinear")
        seg_no_sky_mask = (seg_sky_mask == 0).cuda()

        seg_water_mask = seg_l1[:, 0, :, :].unsqueeze(0)
        seg_water_mask = tf.interpolate(seg_water_mask, [h, w], mode='nearest')
        # seg_water_mask = interpolate2d_as(seg_water_mask, target_dict['input_l1'], mode="bilinear")
        seg_water_mask = (seg_water_mask == 1).cuda()
        seg_no_water_mask = (seg_water_mask == 0).cuda()

        seg_obstacle_mask = seg_no_sky_mask.logical_and(
            seg_no_water_mask).int().float()

        # plt.imshow(seg_l1[0].cpu().numpy().transpose(1,2,0))
        # plt.show()
        # seg_obstacle_mask = seg_l1[:, 2, :, :].unsqueeze(0)
        # plt.imshow(seg_obstacle_mask[0, 0].cpu().numpy())
        # plt.show()

        # Depth Eval
        # logging.info(f'output_dict: {output_dict.keys()}')
        # logging.info(f'target_dict: {target_dict.keys()}')
        seg_3_channel_no_sky_mask = seg_no_sky_mask.repeat(
            1, 3, 1, 1).int().float()

        l1_pp = target_dict['input_l1'] * seg_3_channel_no_sky_mask
        r1_pp = target_dict['input_r1'] * seg_3_channel_no_sky_mask
        pseudo_gt_disp = torch.Tensor(gt_disp).cuda()
        pseudo_gt_disp = interpolate2d_as(
            pseudo_gt_disp, l1_pp, mode="bilinear")
        pseudo_gt_disp = pseudo_gt_disp * pseudo_gt_disp.size(3)

        # pseudo_gt_disp_mask = (pseudo_gt_disp > 0)
        # seg_obstacle_mask = pseudo_gt_disp_mask.logical_and(
        #     seg_obstacle_mask)
        # seg_water_mask = pseudo_gt_disp_mask.logical_and(
        #     seg_water_mask)

        # OBSTACLE

        # pseudo_gt_disp_obstacle = pseudo_gt_disp * seg_obstacle_mask.int().float()
        # logging.info(
        #     f'pseudo_gt_disp min: {pseudo_gt_disp_obstacle[pseudo_gt_disp_obstacle > 0].min()}, max: {pseudo_gt_disp_obstacle[pseudo_gt_disp_obstacle > 0].max()}, mean: {pseudo_gt_disp_obstacle[pseudo_gt_disp_obstacle > 0].mean()}')
        # pseudo_gt_disp_obstacle = (pseudo_gt_disp_obstacle - pseudo_gt_disp_obstacle.mean()) / pseudo_gt_disp_obstacle.std()


        intrinsics = target_dict['input_k_l1_flip_aug']
        # plt.imshow(pseudo_gt_disp[0, 0].cpu().numpy())
        t = pseudo_gt_disp.size(3) / output_dict["disp_l1_pp"][0].size(3)


        input_l1 = target_dict['input_l1']
        out_disp_l_pp = interpolate2d_as(
            output_dict["disp_l1_pp"][0], input_l1, mode="bilinear") * t * 2 * pseudo_gt_disp.size(3)

        out_disp_l_final = out_disp_l_pp / (2*t)

        # logging.info(f'out_disp_l_pp min: {out_disp_l_pp[out_disp_l_pp > 0].min()}, max: {out_disp_l_pp[out_disp_l_pp > 0].max()}, mean: {out_disp_l_pp[out_disp_l_pp > 0].mean()}')
        # logging.info(f'pseudo_gt_disp min: {pseudo_gt_disp[pseudo_gt_disp > 0].min()}, max: {pseudo_gt_disp[pseudo_gt_disp > 0].max()}, mean: {pseudo_gt_disp[pseudo_gt_disp > 0].mean()}')


        pseudo_gt_disp = pseudo_gt_disp * seg_no_sky_mask.int().float()
        out_disp_l_pp = out_disp_l_pp * seg_no_sky_mask.int().float()
        pseudo_gt_disp = (pseudo_gt_disp - pseudo_gt_disp.mean()) / pseudo_gt_disp.std()
        out_disp_l_pp = (out_disp_l_pp - out_disp_l_pp.mean()) / out_disp_l_pp.std()

        delta = max(pseudo_gt_disp.min(), out_disp_l_pp.min())
        pseudo_gt_disp = pseudo_gt_disp - delta
        out_disp_l_pp = out_disp_l_pp - delta
        pseudo_gt_disp = torch.clamp(pseudo_gt_disp, 0) * 255
        out_disp_l_pp = torch.clamp(out_disp_l_pp, 0) * 255

        # logging.info(f'out_disp_l_pp min: {out_disp_l_pp[out_disp_l_pp > 0].min()}, max: {out_disp_l_pp[out_disp_l_pp > 0].max()}, mean: {out_disp_l_pp[out_disp_l_pp > 0].mean()}')
        # logging.info(f'pseudo_gt_disp min: {pseudo_gt_disp[pseudo_gt_disp > 0].min()}, max: {pseudo_gt_disp[pseudo_gt_disp > 0].max()}, mean: {pseudo_gt_disp[pseudo_gt_disp > 0].mean()}')


        # pseudo_gt_disp = pseudo_gt_disp * 255
        # out_disp_l_pp = out_disp_l_pp * 255

        pseudo_gt_depth = _disp2depth_kitti_K(pseudo_gt_disp, intrinsics[:, 0, 0])
        pseudo_gt_depth = torch.clamp(pseudo_gt_depth, 1e-3, 80)

        gt_depth_mask = (pseudo_gt_depth > 1e-3) * (pseudo_gt_depth < 80)
        gt_depth_mask = gt_depth_mask.logical_and(seg_no_sky_mask)
        gt_water_mask = gt_depth_mask.logical_and(seg_water_mask)
        gt_obstacle_mask = gt_depth_mask.logical_and(seg_obstacle_mask)

        out_depth_l_pp = _disp2depth_kitti_K(out_disp_l_pp, intrinsics[:, 0, 0])
        out_depth_l_pp = torch.clamp(out_depth_l_pp, 1e-3, 80)
        # plt.subplot(1,2,1)
        # plt.imshow(pseudo_gt_depth[0,0].cpu().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(out_depth_l_pp[0,0].cpu().numpy())
        # plt.show()




        # pseudo_gt_disp = (pseudo_gt_disp - pseudo_gt_disp.min()) / (pseudo_gt_disp.max())
        # out_disp_l_pp = (out_disp_l_pp - out_disp_l_pp.min()) / (out_disp_l_pp.max())
        # delta = min(pseudo_gt_disp.min(), out_disp_l_pp.min())
        # pseudo_gt_disp = pseudo_gt_disp - delta
        # out_disp_l_pp = out_disp_l_pp - delta
        # out_disp_l_pp = interpolate2d_as(
        #     output_dict["disp_l1_pp"][0], pseudo_gt_disp, mode="bilinear") * pseudo_gt_disp.size(3)

        # plt.subplot(1,2,1)
        # plt.imshow(pseudo_gt_disp[0,0].cpu().numpy(), cmap=cmap)
        # plt.subplot(1,2,2)
        # plt.imshow(out_disp_l_pp[0,0].cpu().numpy(), cmap=cmap)
        # plt.show()
        abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = compute_errors(pseudo_gt_depth[gt_depth_mask], out_depth_l_pp[gt_depth_mask])
        w_abs_rel, w_sq_rel, w_rms, w_log_rms, w_a1, w_a2, w_a3 = compute_errors(pseudo_gt_depth[gt_water_mask], out_depth_l_pp[gt_water_mask])
        o_abs_rel, o_sq_rel, o_rms, o_log_rms, o_a1, o_a2, o_a3 = compute_errors(pseudo_gt_depth[gt_obstacle_mask], out_depth_l_pp[gt_obstacle_mask])

        output_dict["out_disp_l_pp"] = out_disp_l_final

        out_sceneflow = interpolate2d_as(
            output_dict['flow_f_pp'][0], input_l1, mode="bilinear")
        out_flow = projectSceneFlow2Flow(
            target_dict['input_k_l1'], out_sceneflow, out_disp_l_final)
        output_dict["out_flow_pp"] = out_flow

        # out_disp_l_pp_obstacle = out_disp_l_pp * seg_obstacle_mask.int().float()
        # # out_disp_l_pp_obstacle = (out_disp_l_pp_obstacle - out_disp_l_pp_obstacle.mean()) / out_disp_l_pp_obstacle.std()

        # o_abs_rel, o_sq_rel, o_rms, o_log_rms, o_a1, o_a2, o_a3 = compute_errors(pseudo_gt_disp_obstacle[seg_obstacle_mask], out_disp_l_pp_obstacle[seg_obstacle_mask])

        # # logging.info(f'out_disp_l_pp_obstacle: abs_rel: {o_abs_rel}, sq_rel: {o_sq_rel}, rms: {o_rms}, log_rms: {o_log_rms}, a1: {o_a1}, a2: {o_a2}, a3: {o_a3}')

        # ###############################
        # ### WATER

        # pseudo_gt_disp_water = pseudo_gt_disp * seg_water_mask.int().float()
        # # logging.info(
        # #     f'pseudo_gt_disp min: {pseudo_gt_disp_water[pseudo_gt_disp_water > 0].min()}, max: {pseudo_gt_disp_water[pseudo_gt_disp_water > 0].max()}, mean: {pseudo_gt_disp_water[pseudo_gt_disp_water > 0].mean()}')

        # out_disp_l_pp_water = out_disp_l_pp * seg_water_mask.int().float()

        # w_abs_rel, w_sq_rel, w_rms, w_log_rms, w_a1, w_a2, w_a3 = compute_errors(pseudo_gt_disp_water[seg_water_mask], out_disp_l_pp_water[seg_water_mask])

        # # logging.info(f'out_disp_l_pp_water: abs_rel: {w_abs_rel}, sq_rel: {w_sq_rel}, rms: {w_rms}, log_rms: {w_log_rms}, a1: {w_a1}, a2: {w_a2}, a3: {w_a3}')


        ###############################



        # out_depth_l_pp = _disp2depth(out_disp_l_pp_masked, intrinsics)
        # # out_depth_l_pp = torch.clamp(out_depth_l_pp, 1e-3, 80)
        # gt_depth_pp = _disp2depth(pseudo_gt_disp, intrinsics)
        # # gt_depth_pp = torch.clamp(gt_depth_pp, 1e-3, 80)

        # if vis:
        #     plt.subplot(3, 3, 7)
        #     plt.title('SGBM depth')
        #     plt.imshow(gt_depth_pp[0, 0].cpu().numpy(), cmap=cmap, vmin=1)
        #     plt.subplot(3, 3, 8)
        #     plt.title('pred depth')
        #     plt.imshow(out_depth_l_pp[0, 0].cpu().numpy(), cmap=cmap, vmin=1)
        #     plt.subplot(3, 3, 9)
        #     plt.title('pred depth unmasked')
        # out_depth_sky_mask = _disp2depth(
        #     out_disp_l_pp * seg_no_sky_mask.float().int(), intrinsics)
        # # out_depth_sky_mask = out_depth_sky_mask / out_depth_sky_mask.max()

        # if vis:
        #     plt.imshow(
        #         out_depth_sky_mask[0, 0].cpu().numpy(), cmap=cmap, vmin=1)
        #     plt.show()

        # output_dict_displ = eval_module_disp_depth(
        #     pseudo_gt_disp, full_depth_mask.bool(), out_disp_l_pp, gt_depth_pp, out_depth_l_pp)

        # output_dict["out_disp_l_pp"] = out_disp_l_pp
        # output_dict["out_depth_l_pp"] = out_depth_l_pp
        # # logging.info(f'output_dict_displ: {output_dict_displ.keys()}')
        # performance_metrics["ab_r"] = output_dict_displ['abs_rel']
        # performance_metrics["sq_r"] = output_dict_displ['sq_rel']
        # performance_metrics["rms"] = output_dict_displ['rms']
        # performance_metrics["lrms"] = output_dict_displ['log_rms']
        # performance_metrics["a1"] = output_dict_displ['a1']
        # performance_metrics["a2"] = output_dict_displ['a2']
        # performance_metrics["a3"] = output_dict_displ['a3']
        # performance_metrics['sgbm_mean'] = pseudo_gt_disp[pseudo_gt_disp > 0].mean(
        # )
        # performance_metrics['out_disp_l_pp_mean'] = out_disp_l_pp[out_disp_l_pp > 0].mean(
        # )
        # print(performance_metrics)
        performance_metrics['o_abs_rel'] = o_abs_rel
        performance_metrics['o_sq_rel'] = o_sq_rel
        performance_metrics['o_rms'] = o_rms
        performance_metrics['o_log_rms'] = o_log_rms
        performance_metrics['o_a1'] = o_a1
        performance_metrics['o_a2'] = o_a2
        performance_metrics['o_a3'] = o_a3
        performance_metrics['w_abs_rel'] = w_abs_rel
        performance_metrics['w_sq_rel'] = w_sq_rel
        performance_metrics['w_rms'] = w_rms
        performance_metrics['w_a1'] = w_a1
        performance_metrics['w_log_rms'] = w_log_rms
        performance_metrics['w_a2'] = w_a2
        performance_metrics['w_a3'] = w_a3

        performance_metrics['full_abs_rel'] = abs_rel
        performance_metrics['full_sq_rel'] = sq_rel
        performance_metrics['full_rms'] = rms
        performance_metrics['full_log_rms'] = log_rms
        performance_metrics['full_a1'] = a1
        performance_metrics['full_a2'] = a2
        performance_metrics['full_a3'] = a3

        if vis:
            plt.subplot(3, 3, 1)
            plt.title('left 1')
            plt.imshow(target_dict['input_l1']
                       [0].permute(1, 2, 0).cpu().numpy())

            plt.subplot(3, 3, 2)
            plt.title('right 1')
            plt.imshow(target_dict['input_r1']
                       [0].permute(1, 2, 0).cpu().numpy())

            plt.subplot(3, 3, 3)
            plt.title('left 2')
            plt.imshow(target_dict['input_l2']
                       [0].permute(1, 2, 0).cpu().numpy())

            plt.subplot(3, 3, 4)
            # plt.title('sgbm pred obstacle')
            # vis_pseudo_gt_disp_obstacle = (pseudo_gt_disp_obstacle - \
            #     pseudo_gt_disp_obstacle.min()) / pseudo_gt_disp_obstacle.max()
            # vis_pseudo_gt_disp_obstacle = vis_pseudo_gt_disp_obstacle * 255
            # plt.imshow(pseudo_gt_disp_obstacle[0, 0].cpu().numpy(), cmap=cmap, vmin=1)
            plt.imshow(pseudo_gt_disp[0,0].cpu().numpy(), cmap=cmap, vmin=0.01)

            plt.subplot(3, 3, 5)
            # plt.title('self-mono-sf pred obstacle')
            # vis_out_disp_l_pp_obstacle = (out_disp_l_pp_obstacle - \
            #     out_disp_l_pp_obstacle.min()) / out_disp_l_pp_obstacle.max()
            # vis_out_disp_l_pp_obstacle = vis_out_disp_l_pp_obstacle * 255
            # plt.imshow(out_disp_l_pp_obstacle[0, 0].cpu().numpy(), cmap=cmap, vmin=1)
            plt.imshow(out_disp_l_pp[0,0].cpu().numpy(), cmap=cmap, vmin=0.01)

            plt.subplot(3, 3, 6)
            plt.title('difference')
            diff = torch.abs(pseudo_gt_disp - out_disp_l_pp)
            vis_diff = (diff - diff.min()) / diff.max()
            vis_diff = vis_diff * 255
            plt.imshow(diff[0, 0].cpu().numpy(), cmap=cmap, vmin=0.01)

            # plt.subplot(3, 3, 7)
            # plt.title('sgbm pred water')
            # plt.imshow(pseudo_gt_disp_water[0, 0].cpu().numpy(), cmap=cmap, vmin=1)

            # plt.subplot(3, 3, 8)
            # plt.title('self-mono-sf pred water')
            # plt.imshow(out_disp_l_pp_water[0, 0].cpu().numpy(), cmap=cmap, vmin=1)

            # plt.subplot(3, 3, 9)
            # plt.title('difference')
            # diff = torch.abs(pseudo_gt_disp_water - out_disp_l_pp_water)
            # plt.imshow(diff[0, 0].cpu().numpy(), cmap=cmap, vmin=1)
            plt.show()

        return performance_metrics


class Eval_SceneFlow_MODS_Test(nn.Module):
    """Test on MODS
    Performance metrics:
        1. Inlier ratio
        2. RMSE
        3. Abs Rel
        4. Sq Rel
    """

    def __init__(self, args):
        super(Eval_SceneFlow_MODS_Test, self).__init__()

    def upsample_flow_as(self, flow, output_as):
        size_inputs = flow.size()[2:4]
        size_targets = output_as.size()[2:4]
        resized_flow = tf.interpolate(
            flow, size=size_targets, mode="bilinear", align_corners=True)
        # correct scaling of flow
        u, v = resized_flow.chunk(2, dim=1)
        u *= float(size_targets[1] / size_inputs[1])
        v *= float(size_targets[0] / size_inputs[0])
        return torch.cat([u, v], dim=1)

    def forward(self, output_dict, target_dict):
        # logging.info("Evaluating on MODS")
        # logging.info("output_dict: {}".format(output_dict.keys()))
        # logging.info("target_dict: {}".format(target_dict.keys()))
        vis = False

        loss_dict = {}

        input_l1 = target_dict['input_l1']
        input_r1 = target_dict['input_r1']
        input_k_l1 = target_dict['input_k_l1']
        input_k_r1 = target_dict['input_k_r1']
        input_l2 = target_dict['input_l2']
        input_r2 = target_dict['input_r2']
        input_k_l2 = target_dict['input_k_l2']
        input_k_r2 = target_dict['input_k_r2']

        out_flow = self.upsample_flow_as(output_dict['flow_f'][0], input_l1)
        output_dict['out_flow_pp'] = out_flow

        # plt.imshow(input_l1[0].permute(1,2,0).cpu().numpy())
        # plt.show()

        # logging.info(f'input_l1: {input_l1.shape}')
        # logging.info(f'basename: {target_dict["basename"]}')

        # predict disparity with SGM, only use confidence > 0.5
        # then compare to output disparity from model
        window_size = 3
        min_disp = 0
        num_disp = 80-min_disp
        # stereo = cv2.StereoSGBM_create(
        #     minDisparity=min_disp,
        #     numDisparities=num_disp,
        #     blockSize = 11,
        #     P1 = 8*3*window_size**2,
        #     P2 = 32*3*window_size**2,
        #     disp12MaxDiff = 3,
        #     uniquenessRatio = 15,
        #     speckleWindowSize = 100,
        #     speckleRange = 2,
        # )
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            uniquenessRatio=50,
            disp12MaxDiff=1,
        )

        l1_np = input_l1[0].permute(1, 2, 0).cpu().numpy() * 255.0
        l1_np = l1_np.astype(np.uint8)
        r1_np = input_r1[0].permute(1, 2, 0).cpu().numpy() * 255.0
        r1_np = r1_np.astype(np.uint8)
        disp = stereo.compute(l1_np, r1_np).astype(np.float32) / 16.0
        if vis:
            plt.subplot(2, 2, 1)
            plt.title('left')
            plt.imshow(l1_np.astype(np.float32)/255.0)
            plt.subplot(2, 2, 2)
            plt.title('right')
            plt.imshow(r1_np.astype(np.float32)/255.0)

        disp_tensor = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0).cuda()

        batch_size, _, _, width = disp_tensor.size()
        out_disp_l1 = interpolate2d_as(
            output_dict["disp_l1_pp"][0], disp_tensor, mode="bilinear") * width
        out_disp_l1 = torch.clamp(out_disp_l1, 1e-3, 80)
        out_disp_l1 = out_disp_l1 * (disp_tensor > 0).float()

        # # clear figure
        # plt.clf()
        # plt.subplot(2, 2, 1)
        # plt.title('sgm disp')
        # # histogram of original disparity
        # plt.hist(disp.flatten(), bins=100)
        # plt.subplot(2, 2, 2)
        # plt.title('model disp')
        # # histogram of model disparity
        # plt.hist(out_disp_l1_n.flatten(), bins=100)
        # plt.show()

        disp_mask = disp > 0
        disp_mask_torch = torch.from_numpy(
            disp_mask).unsqueeze(0).unsqueeze(0).cuda()

        disp_mask_f = disp_mask_torch.float()

        disp_torch = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0).cuda()
        disp_torch = torch.clamp(disp_torch, 1e-3, 80)

        # out_disp_l1 = out_disp_l1 / out_disp_l1.max()

        # logging.info(f'before standardization: model disp max: {out_disp_l1.max()}, min: {out_disp_l1.min()}, mean: {out_disp_l1.mean()}')
        # logging.info(f'before standardization: SGM disp max: {disp_torch.max()}, min: {disp_torch.min()}, mean: {disp_torch.mean()}')
        # standardize both
        out_disp_l1 = (out_disp_l1 - out_disp_l1.mean()) / out_disp_l1.std()
        disp_torch = (disp_torch - disp_torch.mean()) / disp_torch.std()

        ##################################################
        # Depth 1
        ##################################################

        intrinsics = target_dict['input_k_l1']
        translation_matrix = target_dict['input_t'].flatten()
        out_depth_mine = _disp2depth(
            out_disp_l1, intrinsics, translation_matrix[0])
        # logging.info(f'out_depth_mine: {out_depth_mine.shape}')
        out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)

        gt_depth = _disp2depth_kitti_K(disp_torch, intrinsics[:, 0, 0])

        performance_metrics = eval_module_disp_depth(
            disp_torch, disp_mask_torch, out_disp_l1, gt_depth, out_depth_l1)
        # logging.info(f'metrics: {performance_metrics}')
        # logging.info(f'after: model disp max: {out_disp_l1.max()}, min: {out_disp_l1.min()}, mean: {out_disp_l1.mean()}')
        # logging.info(f'after: SGM disp max: {disp_torch.max()}, min: {disp_torch.min()}, mean: {disp_torch.mean()}')

        # VISUALIZATION

        if vis:
            plt.subplot(2, 2, 3)
            plt.title('sgm disp')
            plt.imshow(disp_torch.cpu().numpy()[0, 0])

            plt.subplot(2, 2, 4)
            plt.title('model disp')
            plt.imshow(out_disp_l1.cpu().numpy()[0, 0])

            plt.show()

        # END VISUALIZATION

        # KITTI disparity metric
        d_valid_epe = _elementwise_epe(out_disp_l1, disp_torch) * disp_mask_f
        # For this benchmark, we consider a pixel to be correctly estimated if the disparity or flow end-point error is <3px or <5%.
        d_outlier_epe = (d_valid_epe > 3).float(
        ) * ((d_valid_epe / disp_torch) > 0.05).float() * disp_mask_f
        loss_dict["otl"] = (d_outlier_epe.view(batch_size, -1).sum(1)).mean()
        # logging.info(f'otl: {loss_dict["otl"]}')

        # loss_dict["otl_img"] = d_outlier_epe
        # if vis:
        #     plt.subplot(3, 2, 5)
        #     plt.title('outlier')
        #     plt.imshow(d_outlier_epe.cpu().numpy()[0,0])
        #     plt.show()

        # cv2.imshow('disp', disp)
        # cv2.waitKey(0)

        # gt_flow = target_dict['target_flow']
        # gt_flow_mask = (target_dict['target_flow_mask']==1).float()

        # gt_disp = target_dict['target_disp']
        # gt_disp_mask = (target_dict['target_disp_mask']==1).float()

        # gt_disp2_occ = target_dict['target_disp2_occ']
        # gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()
        return loss_dict
