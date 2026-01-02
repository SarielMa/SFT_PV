python merge_lora.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --adapter /home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/runs/pv_sft_Qwen2.5-1.5B-Instruct_qlora_epoch3/lora_adapter \
  --out /home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/runs/pv_sft_Qwen2.5-1.5B-Instruct_merged_epoch3 \
  --dtype bf16

