#!/bin/bash

# DATASETS_HOME
# MODD2_HOME="data/modd2/rectified_video_data"
MODD2_HOME="data/kitti_data"
CHECKPOINT="checkpoints/full_model_kitti_ft/checkpoint_kitti_ft.ckpt"

# model
MODEL=MonoSceneFlow_fullmodel

# Valid_Dataset=MODD2_Visualisation_mnsf
Valid_Dataset=KITTI_Raw_KittiSplit_Valid_mnsf
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_SceneFlow_KITTI_Test

# training configuration
SAVE_PATH="eval/visualisation/kitti/full_model_kitti_ft"
python ../main.py \
--batch_size=1 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--model=$MODEL \
--evaluation=True \
--num_workers=4 \
--save=$SAVE_PATH \
--start_epoch=1 \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$MODD2_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=sf \
--validation_dataset_num_examples=10 \
--save_disp=True \
--save_disp2=True \
--save_flow=True
