python eval_vllm_prf.py \
  --model "/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/runs/pv_sft_Qwen2.5-1.5B-Instruct_merged_epoch10" \
  --data  "/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/test/" \
  --tp 1 \
  --temperature 0.0 \
  --max_tokens 8192 \
  --batch_size 1 \
  --out_json "metrics_qwen1p5b.json" \
  --out_pred_jsonl "pred_dump_qwen1p5b.jsonl"
