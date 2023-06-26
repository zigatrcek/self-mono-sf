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
import time

TRAINING_BATCH_SIZE = 4
VALIDATION_BATCH_SIZE = 1
EPOCHS = 30

torch.manual_seed(0)


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
    train_loader = DataLoader(
        train,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **gpuargs)
    print(f'len(train_loader): {len(train_loader)}')


    valid_loader = DataLoader(
        valid,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **gpuargs)

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
    seg_model.train()
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

    min_valid_loss = float('inf')

    for e in range(EPOCHS):
        start_time = time.time()
        training_loss = 0.0
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


            seg_in = torch.cat((msf_out["flow_f"][0], msf_out["flow_b"]
                            [0], msf_out["disp_l1"][0], msf_out["disp_l2"][0]), dim=1)
            seg_out = seg_model(seg_in)
            # optimize
            target = sample["ann_l1"].squeeze(1).cuda()
            # dataset labels are 0, 1, 2, 4
            target[target == 4] = 3
            # target = target + 1
            loss = calculate_loss(seg_out, target)
            # print(f'Batch: {batch_idx} Loss: {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        end_train_time = time.time()
        seg_model.eval()
        validation_loss = 0.0
        for batch_idx, sample in enumerate(valid_loader):
            input_keys = list(filter(lambda x: "input" in x, sample.keys()))
            target_keys = list(filter(lambda x: "target" in x, sample.keys()))
            tensor_keys = input_keys + target_keys

            for key, value in sample.items():
                if key in tensor_keys:
                    sample[key] = value.cuda(non_blocking=True)
            augmented_sample = augmentation(sample)


            with torch.inference_mode():
                msf_out = msf_model(augmented_sample)


            seg_in = torch.cat((msf_out["flow_f"][0], msf_out["flow_b"]
                            [0], msf_out["disp_l1"][0], msf_out["disp_l2"][0]), dim=1)
            seg_out = seg_model(seg_in)

            target = sample["ann_l1"].squeeze(1).cuda()
            # dataset labels are 0, 1, 2, 4
            target[target == 4] = 3
            # target = target + 1
            loss = calculate_loss(seg_out, target)
            validation_loss += loss.item()
            # print(f'Batch: {batch_idx} Loss: {loss.item()}')
        end_valid_time = time.time()
        if validation_loss < min_valid_loss:
            min_valid_loss = validation_loss
            print(f'Saving model with validation loss {min_valid_loss}')
            torch.save(seg_model.state_dict(), 'seg_model.pt')
        print(f'Epoch: {e} Training Loss: {training_loss / len(train_loader)} Validation Loss: {validation_loss / len(valid_loader)} Time: {end_valid_time - start_time} Train Time: {end_train_time - start_time} Valid Time: {end_valid_time - end_train_time}')
        seg_model.train()


if __name__ == "__main__":
    main()
