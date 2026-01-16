

python merge_lora.py \
  --base /home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/runs/pv_sft_Qwen2.5-1.5B-Instruct_merged_epoch10 \
  --adapter ./dpo_runs/Qwen2.5-1.5B-Instruct \
  --out ./dpo_runs/Qwen2.5-1.5B-Instruct-merged \
  --dtype bf16

