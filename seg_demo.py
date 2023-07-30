import logging
import torch
import os
from core import commandline, runtime, logger, tools, configuration as config
from datasets import MaSTr1325_Full, Mods_Full
import torch.nn.functional as tf
from torch.utils.data import DataLoader, random_split
from argparse import Namespace
from models import model_monosceneflow as msf, model_segmentation as mseg
from augmentations import NoAugmentation, Augmentation_Resize_Only
from utils.sceneflow_util import projectSceneFlow2Flow
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from utils.interpolation import interpolate2d_as
from collections import OrderedDict
import numpy as np
from utils.flow import flow_to_png_middlebury

torch.manual_seed(0)
Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Sky, Building, Pole, Unlabelled])


VALIDATION_BATCH_SIZE = 1
DATASET_SELECTION = 'mods'
IMG_SIZE = [960, 1280]


def main():

    # Change working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    gpuargs = {"num_workers": 12, "pin_memory": True}
    if DATASET_SELECTION == 'mods':
        full_dataset = Mods_Full(
            args=Namespace(),
            root='../data/mods/sequences',
            flip_augmentations=False,
            preprocessing_crop=False,
            crop_size=[768, 1024],
            # crop_size=[928, 1248],
            num_examples=-1,
        )
        valid = full_dataset
    elif DATASET_SELECTION == 'mastr':
        full_dataset = MaSTr1325_Full(
            args=Namespace(),
            root='../data/mastr1325/MaSTr1325_images_512x384',
            # root='/storage/datasets/MaSTr1325/images',
            flip_augmentations=False,
            preprocessing_crop=True,
            crop_size=[384, 512],
            num_examples=-1,
        )
        train, valid = random_split(full_dataset, [1100, 189])

    # create train dataloader
    inf_loader = DataLoader(
        valid,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **gpuargs)
    print(f'len(inf_loader): {len(inf_loader)}')

    msf_model = msf.MonoSceneFlow(
        args=Namespace(evaluation=True, finetuning=False))
    msf_model.requires_grad_(False)
    msf_model.eval()
    msf_model.cuda()

    # load from checkpoint
    checkpoint_path = 'experiments/noteworthy/modd2_fulldata_fullres_1/checkpoint_latest.ckpt'
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    # remove '_model.' from key names
    state_dict = OrderedDict([(k.replace('_model.', ''), v)
                             for k, v in state_dict.items()])
    msf_model.load_state_dict(state_dict)
    print(f'Loaded checkpoint from {checkpoint_path}')

    seg_model = mseg.ModelSegmentation(args=Namespace())
    seg_model.load_state_dict(torch.load('seg_model_upscaled.pt'))
    seg_model.cuda()

    augmentation = Augmentation_Resize_Only(
        args=Namespace(), photometric=False, imgsize=IMG_SIZE)

    # focal loss
    focal_loss_fn = smp.losses.FocalLoss(mode='multiclass')

    # jaccard loss
    jaccard_loss_fn = smp.losses.JaccardLoss(
        mode='multiclass', classes=[0, 1, 2, 3])

    optimizer = torch.optim.AdamW([
        dict(params=seg_model.parameters(), lr=0.0001),
    ])

    def calculate_loss(pred, target):
        focal_loss = focal_loss_fn(pred, target)
        jaccard_loss = jaccard_loss_fn(pred, target)
        return 0.5 * focal_loss + 0.5 * jaccard_loss

    def upsample_flow_as(flow, output_as):
        size_inputs = flow.size()[2:4]
        size_targets = output_as.size()[2:4]
        resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
        # correct scaling of flow
        u, v = resized_flow.chunk(2, dim=1)
        u *= float(size_targets[1] / size_inputs[1])
        v *= float(size_targets[0] / size_inputs[0])
        return torch.cat([u, v], dim=1)

    seg_model.eval()
    with torch.inference_mode():
        for batch_idx, sample in enumerate(inf_loader):
            input_keys = list(filter(lambda x: "input" in x, sample.keys()))
            target_keys = list(filter(lambda x: "target" in x, sample.keys()))
            tensor_keys = input_keys + target_keys

            for key, value in sample.items():
                if key in tensor_keys:
                    sample[key] = value.cuda(non_blocking=True)
            augmented_sample = augmentation(sample)

            msf_out = msf_model(augmented_sample)

            seg_in = torch.cat((msf_out["flow_f"][0], msf_out["flow_b"]
                                [0], msf_out["disp_l1"][0], msf_out["disp_l2"][0]), dim=1)
            seg_out = seg_model(seg_in)

            if DATASET_SELECTION == 'mods':
                # ground truth reading not yet implemented for mods
                vis = labelVisualize(4, COLOR_DICT, seg_out.cpu().numpy())

                plt.subplot(2, 2, 1)
                plt.title('input')
                plt.imshow(sample["input_l1_aug"].cpu().numpy()[
                           0].transpose(1, 2, 0))

                plt.subplot(2, 2, 2)
                out_flow = msf_out["flow_f"][0].cpu().numpy()
                flow_f_rgb = flow_to_png_middlebury(out_flow[0, ...])
                plt.imshow(flow_f_rgb)

                plt.subplot(2, 2, 4)
                plt.title('prediction')
                plt.imshow(vis[0])
                plt.show()

            elif DATASET_SELECTION == 'mastr':
                target = sample["ann_l1"].squeeze(1).cuda()
                target[target == 4] = 3
                vis = labelVisualize(4, COLOR_DICT, seg_out.cpu().numpy())

                plt.subplot(2, 2, 1)
                plt.title('input')
                plt.imshow(sample["input_l1_aug"].cpu().numpy()[
                           0].transpose(1, 2, 0))

                plt.subplot(2, 2, 2)

                input_l1 = augmented_sample['input_l1_aug']
                flow_f_pp = msf_out["flow_f_pp"][0]
                disp_l1_pp = msf_out["disp_l1_pp"][0]

                print(f'msf shape: {flow_f_pp.shape}')
                print(f'input_l1 shape: {input_l1.shape}')
                print(f'disp_l1 shape: {disp_l1_pp.shape}')
                # exit()

                out_sceneflow = upsample_flow_as(flow_f_pp, input_l1)
                out_disp = interpolate2d_as(disp_l1_pp, input_l1, mode="bilinear") * input_l1.size(3)
                out_flow_pp = projectSceneFlow2Flow(intrinsic=augmented_sample['input_k_l1_aug'], sceneflow=out_sceneflow, disp=out_disp)
                out_flow_pp = out_flow_pp.data.cpu().numpy()
                flow_f_rgb = flow_to_png_middlebury(out_flow_pp[0, ...])
                # flow_f_rgb = flow_to_png_middlebury(out_sceneflow[0, ...].data.cpu().numpy())
                plt.imshow(flow_f_rgb)

                plt.subplot(2, 2, 3)
                plt.title('ground truth')
                plt.imshow(target.cpu().numpy()[0])

                plt.subplot(2, 2, 4)
                plt.title('prediction')
                plt.imshow(vis[0])
                # save plot to file
                # plt.savefig(f'experiments/seg_predictions/prediction_{batch_idx}.png')
                plt.show()
                # dataset labels are 0, 1, 2, 4
                # target = target + 1
                # loss = calculate_loss(seg_out, target)
                # print(f'loss: {loss.item()}')


def labelVisualize(num_class, color_dict, img):
    img_out = np.argmax(img, axis=1)
    return img_out


if __name__ == "__main__":
    main()
