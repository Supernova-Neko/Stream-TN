#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_machines 1 \
    --num_processes 2 \
    train_stream.py \
        --config "config/train_stream.jsonl" \
        --project_name "StreamTn_v3_1_datav4_seed46" \
        --exp_dir "exp/StreamTn_v3_1_datav4_seed46" \
        --seed 46