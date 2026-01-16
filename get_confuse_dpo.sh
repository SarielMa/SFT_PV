python infer_vllm_and_confusion.py \
  --model "/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/runs/pv_sft_Qwen2.5-1.5B-Instruct_merged_epoch10" \
  --data  "/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test/" \
  --out_code_csv "code_confusion_summary_qwen1p5b.csv" \
  --out_subcode_csv "subcode_confusion_summary_qwen1p5b.csv" \
  --tp 1 \
  --max_tokens 8192 \
  --temperature 0.0 \
  --out_pred_jsonl "pred_dump_qwen1p5b.jsonl"
