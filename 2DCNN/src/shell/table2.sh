#!/usr/bin/env sh

# DEFINE
DATA_ROOT_PATH=""
DEVICE="cuda"

############################## Table 2 #########################################
#### This commands Assumes that you have computed the best models for the architecture
#### or table 1 commands are already run
#### We show commands for k=2,
#### for other values of k just modify the frame_keep_fraction parameter
#### k=2  -> 0.5
#### k=4  -> 0.25
#### k=5  -> 0.2
#### k=10 -> 0.1
################################################################################

# 3d cnn with 50% frames (k=2), with imputation
python3 -m src.scripts.main -c config/config.py \
  --exp_name 3d_cnn_eval_k=2_imputed \
  -r /tmp \
  --mode test \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/brain_age_3d.py \
  --data.root_path "$DATA_ROOT_PATH" \
  --data.frame_keep_style ordered --data.frame_keep_fraction 0.5 \
  --data.impute fill \
  --statefile result/3d_cnn/run_0001/best_model.pt

# 2d lstm with 50% frames (k=2), without imputation
python3 -m src.scripts.main -c config/config.py \
  --exp_name 2d_slice_lstm_eval_k=2 \
  -r /tmp \
  --mode test \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/brain_age_slice_lstm.py \
  --data.root_path "$DATA_ROOT_PATH" \
  --data.frame_keep_style ordered --data.frame_keep_fraction 0.5 \
  --data.impute drop \
  --statefile result/2d_slice_lstm/run_0001/best_model.pt

# 2d lstm with 50% frames (k=2), with imputation
python3 -m src.scripts.main -c config/config.py \
  --exp_name 2d_slice_lstm_eval_k=2_imputed \
  -r /tmp \
  --mode test \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/brain_age_slice_lstm.py \
  --data.root_path "$DATA_ROOT_PATH" \
  --data.frame_keep_style ordered --data.frame_keep_fraction 0.5 \
  --data.impute fill \
  --statefile result/2d_slice_lstm/run_0001/best_model.pt

# 2D-slice-attention with 50% frames (k=2), without imputation
python3 -m src.scripts.main -c config/config.py \
  --exp_name 2d_slice_attention_eval_k=2 \
  -r /tmp \
  --mode test \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "attention" \
  --data.root_path "$DATA_ROOT_PATH" \
  --data.frame_keep_style ordered --data.frame_keep_fraction 0.5 \
  --data.impute drop \
  --statefile result/2d_slice_attention/run_0001/best_model.pt

# 2D-slice-mean with 50% frames (k=2), without imputation
python3 -m src.scripts.main -c config/config.py \
  --exp_name 2d_slice_mean_eval_k=2 \
  -r /tmp \
  --mode test \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" \
  --data.frame_keep_style ordered --data.frame_keep_fraction 0.5 \
  --data.impute drop \
  --statefile result/2d_slice_attention/run_0001/best_model.pt

# 2D-slice-max with 50% frames (k=2), without imputation
python3 -m src.scripts.main -c config/config.py \
  --exp_name 2d_slice_max_eval_k=2 \
  -r /tmp \
  --mode test \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "max" \
  --data.root_path "$DATA_ROOT_PATH" \
  --data.frame_keep_style ordered --data.frame_keep_fraction 0.5 \
  --data.impute drop \
  --statefile result/2d_slice_attention/run_0001/best_model.pt
