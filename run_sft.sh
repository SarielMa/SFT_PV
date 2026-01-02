module load CUDA/12.6

python sft.py \
  --dataset_path /home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test/ \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir ./runs/pv_sft_Qwen2.5-1.5B-Instruct_qlora_epoch30 \
  --use_qlora --bf16 --do_eval \
  --max_length 8192 --batch_size 1 --grad_accum 16 --epochs 30 --lr 2e-4

