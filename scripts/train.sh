#!/bin/bash

python3 shurale/cli/train.py \
  --use_gradient_checkpointing True \
  --deepspeed_stage 0 \
  --stabilize False \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --use_flash_attention_2 False \
  --load_in_4bit True \
  --apply_lora True \
  --raw_lora_target_modules all \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --warmup_steps 1000 \
  --save_total_limit 0 \
  --push_to_hub True \
  --hub_model_id BobaZooba/Shurale7B-v1-LoRA \
  --hub_private_repo True \
  --report_to_wandb True \
  --logging_steps 1 \
  --num_train_epochs 3 \
  --save_steps 1000 \
  --max_steps 10050 \
  --save_safetensors True \
  --use_gradient_checkpointing True \
  --max_length 2048 \
  --prepare_model_for_kbit_training True \
  --label_smoothing_factor 0.1 \
  --path_to_env_file ./.env
