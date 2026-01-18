python extract_sft_table.py \
  --runs_root ./runs_pv \
  --epochs 3 \
  --shot 0 \
  --out_csv PV_SFT_epoch3_shot0.csv \
  --models meta-llama/Llama-3.3-70B-Instruct meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-1.5B-Instruct


# python extract_dpo_table.py \
#   --dpo_root ./dpo_pipeline_outputs \
#   --epochs 3 \
#   --shot 0 \
#   --out_csv PV_DPO_epoch3_shot0.csv \
#   --models meta-llama/Llama-3.3-70B-Instruct meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.2-3B-Instruct
