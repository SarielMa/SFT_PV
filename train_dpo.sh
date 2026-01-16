python train_dpo.py \
    --model_name "/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/runs/pv_sft_Qwen2.5-1.5B-Instruct_merged_epoch10" \
    --train_data_path "./dpo_data" \
    --valid_data_path "./dpo_data" \
    --output_dir "./dpo_runs/Qwen2.5-1.5B-Instruct" \
    --num_gpus 1
