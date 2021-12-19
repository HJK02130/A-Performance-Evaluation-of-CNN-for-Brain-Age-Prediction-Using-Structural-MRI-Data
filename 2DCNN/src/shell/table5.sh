#!/usr/bin/env sh

# DEFINE
DATA_ROOT_PATH=""
DEVICE="cuda"

################################################################################
#### Commands for training with slice along other dimensions (Table 5)
#### we show the command for 2d-slice set network with mean operation and along
#### coronal axis.
#### for other models just use the command from table 1 section and add the
#### data.frame_dim parameter,
#  1 for sagittal (default)
#  2 for coronal
#  3 for axial
################################################################################
python3 -m src.scripts.main -c config/config.py \
  --exp_name 2d_slice_mean_frame_dim_2 \
  -r result/2d_slice_mean_frame_dim_2 \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" \
  --data.frame_dim 2


