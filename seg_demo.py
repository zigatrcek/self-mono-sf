import logging
import torch
import os
from core import commandline, runtime, logger, tools, configuration as config
from datasets import MaSTr1325_Full, Mods_Full
import torch.nn.functional as tf
from torch.utils.data import DataLoader, random_split
from argparse import Namespace
from models import model_monosceneflow as msf, model_segmentation as mseg, model_upsample as mups
from augmentations import NoAugmentation, Augmentation_Resize_Only
from utils.sceneflow_util import projectSceneFlow2Flow
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from utils.interpolation import interpolate2d_as
from collections import OrderedDict
import numpy as np
from utils.flow import flow_to_png_middlebury
import yaml
import sys
import json
import time

path_to_config = sys.argv[1]

config = yaml.safe_load(open(path_to_config))
MODS_ROOT = config['mods_root']
MSF_MODEL_PATH = config['msf_model_path']
UPSAMPLE_MODEL_PATH = config['upsample_model_path']
SEG_MODEL_PATH = config['seg_model_path']
IMG_SIZE = config['img_size']
# currently only supports val size 1
VALIDATION_BATCH_SIZE = config['validation_batch_size']
VALIDATION_BATCH_SIZE = 1
SEG_MODEL = config['seg_model']
RANDOM_SEED = config['random_seed'] if 'random_seed' in config else False
SEG_HEAD = config['seg_head']
DIR = config['dir']
SAVE_PREDICTIONS = config['save_predictions']

timestamp = time.strftime('%m-%d_%H-%M', time.localtime())
experiment_dir = os.path.join(DIR, f'{SEG_MODEL}_{SEG_HEAD}_{timestamp}')
os.makedirs(experiment_dir, exist_ok=True)

IN_CHANNELS_DICT = {
    'corr': 8 + 80,
    'no_corr': 8,
    'pyramid': 8 + 3 + 96,
}

if RANDOM_SEED:
    torch.manual_seed(RANDOM_SEED)



def main():

    # Change working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    gpuargs = {"num_workers": 12, "pin_memory": True}
    full_dataset = Mods_Full(
        args=Namespace(),
        root=MODS_ROOT,
        flip_augmentations=False,
        preprocessing_crop=False,
        crop_size=[768, 1024],
        # crop_size=[928, 1248],
        num_examples=-1,
    )
    valid = full_dataset

    # create train dataloader
    valid_loader = DataLoader(
        valid,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **gpuargs)
    print(f'len(valid_loader): {len(valid_loader)}')

    msf_model = msf.MonoSceneFlow(
        args=Namespace(evaluation=True, finetuning=False))
    msf_model.requires_grad_(False)
    msf_model.eval()
    msf_model.cuda()

    # load from checkpoint
    checkpoint_path = MSF_MODEL_PATH
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    # remove '_model.' from key names
    state_dict = OrderedDict([(k.replace('_model.', ''), v)
                             for k, v in state_dict.items()])
    msf_model.load_state_dict(state_dict)
    print(f'Loaded checkpoint from {checkpoint_path}')

    upsample_model = None
    if SEG_MODEL == 'corr':
        upsample_model = mups.ModelUpsample()
    elif SEG_MODEL == 'pyramid':
        upsample_model = mups.ModelUpsample(num_layers=6, reduce_dim=False)

    if upsample_model is not None:
        upsample_model.load_state_dict(torch.load(UPSAMPLE_MODEL_PATH))
        upsample_model.cuda()
        upsample_model.eval()

    if SEG_HEAD == 'unet':
        seg_model = mseg.ModelSegmentation(args=Namespace())
    elif SEG_HEAD == 'conv':
        seg_model = mseg.ModelConvSegmentation(in_channels=IN_CHANNELS_DICT[SEG_MODEL], out_channels=4)

    seg_model.load_state_dict(torch.load(SEG_MODEL_PATH))
    seg_model.cuda()
    seg_model.eval()


    augmentation = Augmentation_Resize_Only(
        args=Namespace(), photometric=False, imgsize=IMG_SIZE)


    def upsample_flow_as(flow, output_as):
        size_inputs = flow.size()[2:4]
        size_targets = output_as.size()[2:4]
        resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
        # correct scaling of flow
        u, v = resized_flow.chunk(2, dim=1)
        u *= float(size_targets[1] / size_inputs[1])
        v *= float(size_targets[0] / size_inputs[0])
        return torch.cat([u, v], dim=1)

    with torch.inference_mode():
        for batch_idx, sample in enumerate(valid_loader):
            input_keys = list(filter(lambda x: "input" in x, sample.keys()))
            target_keys = list(filter(lambda x: "target" in x, sample.keys()))
            tensor_keys = input_keys + target_keys

            for key, value in sample.items():
                if key in tensor_keys:
                    sample[key] = value.cuda(non_blocking=True)
            augmented_sample = augmentation(sample)

            msf_out = msf_model(augmented_sample)


            if SEG_MODEL == 'corr':
                upsample_in = []
                for x in msf_out["corr_f"][::-1]:
                    upsample_in.append(x.clone())
                    del x
                upsampled = upsample_model(upsample_in)

                upsampled.append(msf_out["flow_f"][0])
                upsampled.append(msf_out["flow_b"][0])
                upsampled.append(msf_out["disp_l1"][0])
                upsampled.append(msf_out["disp_l2"][0])

                seg_in = torch.cat((
                    upsampled
                ), dim=1)
                # print(f'seg_in.shape: {seg_in.shape}')
            elif SEG_MODEL == 'pyramid':
                upsample_in = []
                pyramid = msf_out["x1_pyramid"]
                orig_img = pyramid[-1]
                for x in msf_out["x1_pyramid"][:-1]:
                    upsample_in.append(x.clone())
                    del x
                upsampled = upsample_model(upsample_in)
                upsampled.append(msf_out["flow_f"][0])
                upsampled.append(msf_out["flow_b"][0])
                upsampled.append(msf_out["disp_l1"][0])
                upsampled.append(msf_out["disp_l2"][0])
                upsampled.append(orig_img)
                seg_in = torch.cat((
                    upsampled
                ), dim=1)

            elif SEG_MODEL == 'no_corr':
                seg_in = torch.cat((msf_out["flow_f"][0], msf_out["flow_b"]
                                [0], msf_out["disp_l1"][0], msf_out["disp_l2"][0]), dim=1)

            seg_out = seg_model(seg_in)
            prediction = torch.argmax(seg_out, dim=1, keepdim=True).cpu().numpy()


            plt.subplot(2, 1, 1)
            plt.title('input')
            plt.imshow(sample["input_l1_aug"].cpu().numpy()[
                        0].transpose(1, 2, 0))


            plt.subplot(2, 1, 2)
            plt.title('prediction')
            plt.imshow(prediction[0, 0])
            plt.show()

            if SAVE_PREDICTIONS:
                prediction = prediction[0, 0]
                prediction = prediction.astype(np.uint8)
                # prediction = np.where(prediction == 0, 0, 255)
                # print(sample.keys())
                os.makedirs(os.path.join(experiment_dir, 'predictions', f'{sample["sequence"][0]}'), exist_ok=True)
                plt.imsave(os.path.join(experiment_dir, 'predictions', f'{sample["sequence"][0]}', f'{sample["img_name"][0]}m.png'), prediction)





if __name__ == "__main__":
    main()
