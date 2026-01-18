# SFT_PV

```bash
runs_pv/
└── <ModelName>/                     # e.g. Qwen2.5-1.5B-Instruct
    └── epoch<EPOCHS>/               # e.g. epoch3
        ├── sft_adapter/             # LoRA / QLoRA training outputs
        │   ├── checkpoint-*/        # HF Trainer checkpoints (ignored by git)
        │   └── lora_adapter/         # final adapter (adapter_config.json here)
        │
        ├── merged/                  # ⬅ FULL HF MODEL (base + LoRA merged)
        │   ├── config.json
        │   ├── model.safetensors
        │   └── tokenizer.json
        │
        ├── lm_eval_results/          # (optional) SFT-only eval outputs
        │   ├── PvExtraction_full_results.json
        │   └── PvExtraction_full_samples.jsonl
        │
        └── logs/
            ├── sft.log
            ├── merge.log
            └── eval_PvExtraction_full.log

```

```bash
dpo_pipeline_outputs/
└── <ModelName>_epoch<EPOCHS>_sftMerged/
    ├── confusion/                   # confusion matrices from SFT model
    │   ├── code_confusion_summary.csv
    │   └── subcode_confusion_summary.csv
    │
    ├── pred/                        # raw model predictions
    │   └── pred_dump.jsonl
    │
    ├── dpo_data/                    # prepared DPO preference dataset
    │
    ├── dpo_runs/
    │   ├── dpo_<ModelName>_epoch<EPOCHS>_sftMerged/
    │   │   ├── checkpoint-*/        # HF checkpoints (ignored by git)
    │   │   └── adapter/             # DPO LoRA adapter
    │   │
    │   └── dpo_<ModelName>_epoch<EPOCHS>_sftMerged-merged/
    │       ├── config.json
    │       └── model.safetensors
    │
    ├── lm_eval_results/             # DPO-evaluated results
    │   ├── PvExtraction_full_results.json
    │   └── PvExtraction_full_samples.jsonl
    │
    └── logs/
        ├── infer.log
        ├── prepare_dpo.log
        ├── train_dpo.log
        ├── merge.log
        └── eval.log
```bash
