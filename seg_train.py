import torch
import os
from datasets import MaSTr1325_Full
from torch.utils.data import DataLoader, random_split
from argparse import Namespace
from models import model_monosceneflow as msf, model_segmentation as mseg, model_upsample as mups
from augmentations import NoAugmentation, Augmentation_Resize_Only
import segmentation_models_pytorch as smp
from collections import OrderedDict
import time
from torch.nn import Upsample as upsample

TRAINING_BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 1
EPOCHS = 30
IMG_SIZE = [960, 1280]
IN_CHANNELS_DICT = {
    'corr': 88,
    'no_corr': 8,
}
SEG_MODEL = 'corr'

torch.manual_seed(0)


def main():

    # Change working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    gpuargs = {"num_workers": 12, "pin_memory": True}
    full_dataset = MaSTr1325_Full(
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
    # checkpoint_path = 'experiments/noteworthy/modd2_fulldata_fullres_1/checkpoint_latest.ckpt'
    checkpoint_path = '/home/bogosort/diploma/self-mono-sf/checkpoints/modb_raw/checkpoint_latest.ckpt'
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    # remove '_model.' from key names
    state_dict = OrderedDict([(k.replace('_model.', ''), v) for k, v in state_dict.items()])
    msf_model.load_state_dict(state_dict)
    print(f'Loaded checkpoint from {checkpoint_path}')

    upsample_model = mups.ModelUpsample()
    upsample_model.train()
    upsample_model.cuda()

    seg_model = mseg.ModelSegmentation(args=Namespace(in_channels=IN_CHANNELS_DICT[SEG_MODEL]))
    seg_model.train()
    seg_model.cuda()

    parameters = list(upsample_model.parameters()) + list(seg_model.parameters())


    augmentation = Augmentation_Resize_Only(args=Namespace(), photometric=False, imgsize=IMG_SIZE)
    # augmentation = NoAugmentation(args=Namespace())

    # focal loss
    focal_loss_fn = smp.losses.FocalLoss(mode='multiclass')

    # jaccard loss
    jaccard_loss_fn = smp.losses.JaccardLoss(mode='multiclass', classes=[0, 1, 2, 3])

    optimizer = torch.optim.AdamW([
        dict(params=parameters, lr=0.0001),
    ])

    def calculate_loss(pred, target):
        focal_loss = focal_loss_fn(pred, target)
        jaccard_loss = jaccard_loss_fn(pred, target)
        return 0.5 * focal_loss + 0.5 * jaccard_loss

    min_valid_loss = float('inf')

    upsampler = upsample(IMG_SIZE, mode='nearest', align_corners=None)

    def transform_target(target):
        target = target.type(torch.float).cuda()
        target = upsampler(target).squeeze(1).type(torch.int64)
        # dataset labels are 0, 1, 2, 4
        target[target == 4] = 3
        return target


    for e in range(EPOCHS):
        start_time = time.time()
        training_loss = 0.0
        for batch_idx, sample in enumerate(train_loader):
            batch_start_time = time.time()
            input_keys = list(filter(lambda x: "input" in x, sample.keys()))
            target_keys = list(filter(lambda x: "target" in x, sample.keys()))
            tensor_keys = input_keys + target_keys

            for key, value in sample.items():
                if key in tensor_keys:
                    sample[key] = value.cuda(non_blocking=True)
            augmented_sample = augmentation(sample)


            with torch.inference_mode():
                msf_out = msf_model(augmented_sample)


            if SEG_MODEL == 'corr':
                upsample_in = []
                for x in msf_out["corr_f"][::-1]:
                    upsample_in.append(x.clone())
                    del x
                upsampled = upsample_model(upsample_in)
                # print(f' Len upsampled: {len(upsampled)}')
                # for x in upsampled:
                #     print(f'Shape upsampled: {x.shape}')

                upsampled.append(msf_out["flow_f"][0])
                upsampled.append(msf_out["flow_b"][0])
                upsampled.append(msf_out["disp_l1"][0])
                upsampled.append(msf_out["disp_l2"][0])

                seg_in = torch.cat((
                    upsampled
                ), dim=1)
                # print(f'seg_in.shape: {seg_in.shape}')
            elif SEG_MODEL == 'no_corr':
                seg_in = torch.cat((msf_out["flow_f"][0], msf_out["flow_b"]
                                [0], msf_out["disp_l1"][0], msf_out["disp_l2"][0]), dim=1)

            seg_out = seg_model(seg_in)
            # print(f'seg_out.shape: {seg_out.shape}')
            # optimize
            target = transform_target(sample["ann_l1"])

            loss = calculate_loss(seg_out, target)
            # print(f'Batch: {batch_idx} Loss: {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            batch_end_time = time.time()
            print(f'Epoch: {e} Batch: {batch_idx} Loss: {loss.item()} Time: {batch_end_time - batch_start_time}')
        end_train_time = time.time()
        upsample_model.eval()
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


            if SEG_MODEL == 'corr':
                upsample_in = []
                for x in msf_out["corr_f"][::-1]:
                    upsample_in.append(x.clone())
                    del x
                upsampled = upsample_model(upsample_in)
                # print(f' Len upsampled: {len(upsampled)}')
                # for x in upsampled:
                #     print(f'Shape upsampled: {x.shape}')

                upsampled.append(msf_out["flow_f"][0])
                upsampled.append(msf_out["flow_b"][0])
                upsampled.append(msf_out["disp_l1"][0])
                upsampled.append(msf_out["disp_l2"][0])

                seg_in = torch.cat((
                    upsampled
                ), dim=1)
                # print(f'seg_in.shape: {seg_in.shape}')
            elif SEG_MODEL == 'no_corr':
                seg_in = torch.cat((msf_out["flow_f"][0], msf_out["flow_b"]
                                [0], msf_out["disp_l1"][0], msf_out["disp_l2"][0]), dim=1)
            seg_out = seg_model(seg_in)

            target = transform_target(sample["ann_l1"])

            loss = calculate_loss(seg_out, target)
            validation_loss += loss.item()
            # print(f'Batch: {batch_idx} Loss: {loss.item()}')
        end_valid_time = time.time()
        if validation_loss < min_valid_loss:
            min_valid_loss = validation_loss
            print(f'Saving model with validation loss {min_valid_loss}')
            torch.save(seg_model.state_dict(), 'seg_model.pt')
        print(f'Epoch: {e} Training Loss: {training_loss / len(train_loader)} Validation Loss: {validation_loss / len(valid_loader)} Time: {end_valid_time - start_time} Train Time: {end_train_time - start_time} Valid Time: {end_valid_time - end_train_time}')
        upsample_model.train()
        seg_model.train()


if __name__ == "__main__":
    main()
