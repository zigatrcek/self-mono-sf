#!/bin/bash

# DATASETS_HOME
DATASET_HOME="/storage/datasets/modb_raw/sequences"
CHECKPOINT="/home/ziga/self-mono-sf/checkpoints/modb_raw/checkpoint_latest.ckpt"
# CHECKPOINT="/home/ziga/self-mono-sf/checkpoints/modd2_fulldata_fullres_20_epochs/checkpoint_best.ckpt"

# model
MODEL=MonoSceneFlow_fullmodel

Valid_Dataset=Mods_Test
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_MonoDepth_MODS

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
--validation_dataset_root=$DATASET_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=otl \
--validation_dataset_num_examples=-1 \
# --save_flow=True
# --save_disp=True \
# --save_disp2=True \
