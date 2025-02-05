#!/bin/bash

# DATASETS_HOME
KITTI_HOME="data/modd2/rectified_video_data"
CHECKPOINT="checkpoints/test_run_modd2/checkpoint_best.ckpt"

# model
MODEL=MonoSceneFlow_fullmodel

Valid_Dataset=MODD2_Valid_mnsf
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_SceneFlow_KITTI_Test

# training configuration
SAVE_PATH="eval/monosf_modd2_selfsup_modd2_test"
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
--validation_dataset_root=$KITTI_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=sf \
--validation_dataset_num_examples=100 \
# --save_disp=True \
# --save_disp2=True \
# --save_flow=True
