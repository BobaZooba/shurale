#!/bin/bash

python3 shurale/cli/fuse.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --lora_hub_model_id BobaZooba/Shurale7B-v1-LoRA \
  --hub_model_id BobaZooba/Shurale7B-v1 \
  --hub_private_repo True \
  --force_fp16 True \
  --fused_model_local_path ./fused_model/ \
  --max_shard_size 1GB \
  --path_to_env_file ./.env
