#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

task="inference"
name="zero-shot"
bz="1"
lr="0.00007"
TRAIN_FILE=$1
VALID_FILE=$2
TEST_FILE=$3

python finetunes/finetune_seq2seq/finetune_seq2seq.py \
    --log_gradient_norm="True" \
    --config_path="ProDMM_seq2seq_ckpt/" \
    --tokenizer_path="AI4Protein/prolm_1.5B" \
    --pretrain_task="seq2seq" \
    --train_file="$TRAIN_FILE" \
    --valid_file="$VALID_FILE" \
    --test_file="$TEST_FILE" \
    --train_batch_size="1" \
    --eval_batch_size="1" \
    --num_workers="0" \
    --seed="42" \
    --max_steps="-1" \
    --max_epochs="10000" \
    --accumulate_grad_batches="1" \
    --lr=${lr} \
    --adam_beta1="0.9" \
    --adam_beta2="0.999" \
    --adam_epsilon="1e-7" \
    --gradient_clip_value="1.0" \
    --gradient_clip_algorithm="norm" \
    --precision="bf16-mixed" \
    --weight_decay="0.01" \
    --scheduler_type="linear" \
    --val_check_interval="20" \
    --save_model_dir="finetune_checkpoint/${task}/${name}_lr${lr}_bz${bz}" \
    --save_model_name="${name}_lr${lr}_bz${bz}_${formatted_time}" \
    --log_steps="1" \
    --logger="wandb" \
    --logger_project="prolm" \
    --logger_run_name="${name}_lr${lr}_bz${bz}_${formatted_time}" \
    --wandb_entity="ai4bio-llm" \
    --devices=1 \
    --nodes=1 \
    --accelerator="gpu" \
    --strategy="auto" \
    --warmup_steps="2000" \
    --warmup_max_steps="10000" \
    --save_interval "20" \
    --pretrain_ckpt "ProDMM_seq2seq_ckpt" \
    --frozen_encoder true \
    --num_training_data 202 \
    --patience 10 \
    --cds_gen true \


