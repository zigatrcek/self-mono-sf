#!/bin/bash

# experiments and datasets meta
MODD2_HOME="../data/mastr1325/MaSTr1325_images_512x384"
EXPERIMENTS_HOME="experiments"

# model
MODEL=MonoSceneFlow_Seg

# save path
ALIAS="-modd2_seg-"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL$ALIAS$TIME"
CHECKPOINT=""
MSF_MODEL="experiments/noteworthy/modd2_fulldata_fullres_1/checkpoint_latest.ckpt"

# Loss and Augmentation
Train_Dataset=MaSTr1325_Train_mnsf
Train_Augmentation=NoAugmentation
Train_Loss_Function=Loss_SceneFlow_SelfSup

Valid_Dataset=MaSTr1325_Valid_mnsf
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Loss_SceneFlow_SelfSup

# --checkpoint=$CHECKPOINT \


# training configuration
python ../main.py \
--batch_size=4 \
--batch_size_val=1 \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[23, 39, 47, 54]" \
--model=$MODEL \
--num_workers=16 \
--optimizer=Adam \
--optimizer_lr=2e-4 \
--save=$SAVE_PATH \
--total_epochs=20 \
--training_augmentation=$Train_Augmentation \
--training_augmentation_photometric=False \
--training_dataset=$Train_Dataset \
--training_dataset_root=$MODD2_HOME \
--training_dataset_flip_augmentations=False \
--training_dataset_preprocessing_crop=False \
--training_dataset_num_examples=-1 \
--training_key=total_loss \
--training_loss=$Train_Loss_Function \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$MODD2_HOME \
--validation_dataset_preprocessing_crop=False \
--validation_key=total_loss \
--validation_loss=$Valid_Loss_Function \
--validation_dataset_num_examples=-1 \
--msf_pretrained=$MSF_MODEL \
