import logging
import torch
import os
from core import commandline, runtime, logger, tools, configuration as config
from datasets import MaSTr1325_Train_mnsf, MaSTr1325_Valid_mnsf
from torch.utils.data import DataLoader, random_split
from argparse import Namespace
from models import model_monosceneflow as msf, model_segmentation as mseg
from augmentations import NoAugmentation
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

torch.manual_seed(0)
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Unlabelled])


VALIDATION_BATCH_SIZE = 1



def main():

    # Change working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    gpuargs = {"num_workers": 12, "pin_memory": True}
    full_dataset = MaSTr1325_Train_mnsf(
        args=Namespace(),
        root='../data/mastr1325/MaSTr1325_images_512x384',
        flip_augmentations=False,
        preprocessing_crop=True,
        crop_size=[384, 512],
        num_examples=-1,
    )
    train, valid = random_split(full_dataset,[1100,189])

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
    state_dict = OrderedDict([(k.replace('_model.', ''), v) for k, v in state_dict.items()])
    msf_model.load_state_dict(state_dict)
    print(f'Loaded checkpoint from {checkpoint_path}')

    seg_model = mseg.ModelSegmentation(args=Namespace())
    seg_model.load_state_dict(torch.load('seg_model_seed_0.pt'))
    seg_model.cuda()

    augmentation = NoAugmentation(args=Namespace())

    # focal loss
    focal_loss_fn = smp.losses.FocalLoss(mode='multiclass')

    # jaccard loss
    jaccard_loss_fn = smp.losses.JaccardLoss(mode='multiclass', classes=[0, 1, 2, 3])

    optimizer = torch.optim.AdamW([
        dict(params=seg_model.parameters(), lr=0.0001),
    ])

    def calculate_loss(pred, target):
        focal_loss = focal_loss_fn(pred, target)
        jaccard_loss = jaccard_loss_fn(pred, target)
        return 0.5 * focal_loss + 0.5 * jaccard_loss


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

            target = sample["ann_l1"].squeeze(1).cuda()
            target[target == 4] = 3
            vis = labelVisualize(4, COLOR_DICT, seg_out.cpu().numpy())

            plt.subplot(2, 2, 1)
            plt.title('input')
            plt.imshow(sample["input_l1_aug"].cpu().numpy()[0].transpose(1, 2, 0))


            plt.subplot(2, 2, 3)
            plt.title('ground truth')
            plt.imshow(target.cpu().numpy()[0])

            plt.subplot(2, 2, 4)
            plt.title('prediction')
            plt.imshow(vis[0])
            plt.show()
            # dataset labels are 0, 1, 2, 4
            # target = target + 1
            loss = calculate_loss(seg_out, target)
            print(f'loss: {loss.item()}')



def labelVisualize(num_class,color_dict,img):
    img_out = np.argmax(img, axis=1)
    print(f'img_out shape: {img_out.shape}')
    return img_out


if __name__ == "__main__":
    main()
