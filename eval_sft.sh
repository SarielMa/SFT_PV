#!/bin/bash

# BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
# LORA_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/runs/pv_sft_Qwen2.5-1.5B-Instruct_qlora/lora_adapter"

# lm_eval --model vllm \
#   --model_args "pretrained=${BASE_MODEL},tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_len=8096,enable_lora=True,lora_modules=pv=${LORA_DIR},max_lora_rank=16" \
#   --tasks PvExtraction_full \
#   --num_fewshot 0 \
#   --batch_size auto \
#   --output_path results/PV_qlora_qwen1p5b \
#   --log_samples \
#   --apply_chat_template \
#   --include_path /home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner

lm_eval --model vllm \
  --model_args "pretrained=/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/runs/pv_sft_Qwen2.5-1.5B-Instruct_merged_epoch30,tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_len=8096" \
  --tasks PvExtraction_full \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path results/PV_qlora_qwen1p5b_merged_epoch30 \
  --log_samples \
  --apply_chat_template \
  --include_path /home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner
