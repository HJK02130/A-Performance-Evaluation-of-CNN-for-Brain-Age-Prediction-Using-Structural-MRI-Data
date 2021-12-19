#!/usr/bin/env sh

# DEFINE
DATA_ROOT_PATH=""
DEVICE="cuda"

################################################################################
#### Commands for training with less sample (Table 4)
#### we show the command for 2d-slice set network with mean operation
#### for other models just use the command from table 1 section and add the
#### data.num_sample parameter and increase number of epochs and patience
################################################################################
python3 -m src.scripts.main -c config/config.py \
  --exp_name 2d_slice_mean_n=5000 \
  -r result/2d_slice_mean_n=5000 \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" \
  --data.train_num_sample 5000 \
  --train.max_epoch 145 --train.patience 145


