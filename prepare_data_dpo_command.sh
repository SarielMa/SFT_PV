#!/usr/bin/env bash

python prepare_dpo_data.py \
  --input_dir "/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test/" \
  --output_dir "/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/dpo_data/" \
  --code_confusion_file "/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/code_confusion_summary_qwen1p5b.csv" \
  --subcode_confusion_file "/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/subcode_confusion_summary_qwen1p5b.csv" \
  --negatives_per_sample 1 \
  --seed 42 \
  --print_samples 3
