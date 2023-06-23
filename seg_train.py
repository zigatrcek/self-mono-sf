import logging
import torch
import os
from core import commandline, runtime, logger, tools, configuration as config
from datasets import MaSTr1325_Train_mnsf, MaSTr1325_Valid_mnsf
from torch.utils.data import DataLoader
from argparse import Namespace
from models import model_monosceneflow as msf, model_segmentation as mseg
from augmentations import NoAugmentation
import segmentation_models_pytorch as smp

TRAINING_BATCH_SIZE = 4
VALIDATION_BATCH_SIZE = 1

def main():

    # Change working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    gpuargs = {"num_workers": 12, "pin_memory": True}
    train_dataset = MaSTr1325_Train_mnsf(
        args=Namespace(),
        root='../data/mastr1325/MaSTr1325_images_512x384',
        flip_augmentations=False,
        preprocessing_crop=True,
        crop_size=[384, 512],
        num_examples=-1,
    )

    # create train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **gpuargs)
    print(f'len(train_loader): {len(train_loader)}')

    # create validation dataloader
    # valid_dataset = MaSTr1325_Valid_mnsf(
    #     args=Namespace(),
    #     root='../data/mastr1325/MaSTr1325_images_512x384',
    #     flip_augmentations=False,
    #     preprocessing_crop=True,
    #     crop_size=[384, 512],
    #     num_examples=-1,
    # )
    # valid_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=VALIDATION_BATCH_SIZE,
    #     shuffle=False,
    #     drop_last=False,
    #     **gpuargs)


    msf_model = msf.MonoSceneFlow(args=Namespace(evaluation=True, finetuning=False))
    msf_model.requires_grad_(False)
    msf_model.eval()
    msf_model.cuda()


    mseg_model = mseg.ModelSegmentation(args=Namespace())
    mseg_model.train()
    mseg_model.cuda()

    augmentation = NoAugmentation(args=Namespace())

    # focal loss
    focal_loss_fn = smp.losses.FocalLoss(mode='multiclass')

    # jaccard loss
    jaccard_loss_fn = smp.losses.JaccardLoss(mode='multiclass')

    optimizer = torch.optim.AdamW([
        dict(params=mseg_model.parameters(), lr=0.0001),
    ])

    def calculate_loss(pred, target):
        focal_loss = focal_loss_fn(pred, target)
        print(f'focal_loss: {focal_loss}')
        jaccard_loss = jaccard_loss_fn(pred, target)
        print(f'jaccard_loss: {jaccard_loss}')
        return 0.5 * focal_loss + 0.5 * jaccard_loss




    for batch_idx, sample in enumerate(train_loader):
        input_keys = list(filter(lambda x: "input" in x, sample.keys()))
        target_keys = list(filter(lambda x: "target" in x, sample.keys()))
        tensor_keys = input_keys + target_keys

        for key, value in sample.items():
            if key in tensor_keys:
                sample[key] = value.cuda(non_blocking=True)
        augmented_sample = augmentation(sample)
        with torch.inference_mode():
            msf_out = msf_model(augmented_sample)
        seg_in = torch.cat((msf_out["flow_f"][0], msf_out["flow_b"][0], msf_out["disp_l1"][0], msf_out["disp_l2"][0]), dim=1)
        print(f'seg_in: {seg_in.shape}')
        seg_out = mseg_model(seg_in)
        print(f'seg_out: {seg_out.shape}')
        # optimize
        annotated = sample["ann_l1"].squeeze(1).cuda()
        annotated[annotated == 4] = 3
        annotated = annotated + 1
        loss = calculate_loss(seg_out, annotated)
        print(sample['ann_l1'].unique())

        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()







if __name__ == "__main__":
    main()
