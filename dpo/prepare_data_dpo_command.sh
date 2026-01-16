!python /nfs/roberts/scratch/pi_sjf37/gp528/FinBen/Llama_eightb_train/prepare_targeted_dpo_data_revised.py \
    --input_dir /home/gp528/eppc_dataset_local_train/train/Annotator\ Train\ Valid\ Dataset\ Full\ Dataset \
    --output_dir /home/gp528/eppc_dpo_dataset_revised_v1_70b/valid \
    --code_confusion_file /home/gp528/eppc_dataset_local/test/code_confusion_summary_benchmark_annotator_finetuned_70b_full_3.csv\
    --subcode_confusion_file /home/gp528/eppc_dataset_local/test/subcode_confusion_summary_benchmark_annotator_finetuned_70b_full_3.csv \
    --negatives_per_sample 1 \
    --seed 42 \
    --print_samples 3