#!/bin/bash

python3 shurale/cli/prepare.py \
  --dataset_key soda \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --path_to_env_file ./.env
