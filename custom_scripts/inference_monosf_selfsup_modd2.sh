#!/bin/bash

# DATASETS_HOME
MODD2_HOME="/storage/private/student-vicos/modd2/video_data"
CHECKPOINT="/home/ziga/self-mono-sf/checkpoints/modd2/checkpoint_best.ckpt"

# model
MODEL=MonoSceneFlow_fullmodel

# Valid_Dataset=MODD2_Inference_mnsf
Valid_Dataset=MODD2_Visualisation_mnsf
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_Monodepth_MODD2

# training configuration
SAVE_PATH="eval/inference/modd2_sgbm_test"
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
--validation_key=otl \
--validation_dataset_num_examples=-1 \
--save_disp=False \
--save_disp2=False \
--save_flow=False \
--save_depth=False
